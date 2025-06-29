import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from apify_client import ApifyClient
from .. import config
from ..utils.image_utils import download_image, is_landscape, get_image_metadata, create_storage_structure
from ..utils.gcs_storage import GCSStorage
from .image_filter import ImageContentFilter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InstagramScraper:
    """
    Class for scraping Instagram posts and downloading images.
    Wraps around the functional Instagram scraping utilities.
    """
    
    def __init__(self, username: str = None, output_dir: str = 'data'):
        """
        Initialize the Instagram scraper.
        
        Args:
            username: Instagram username to scrape. If None, uses profiles from config.
            output_dir: Directory to store downloaded images and metadata.
        """
        self.username = username
        self.output_dir = output_dir
        
        # Create storage directories
        self.storage_paths = create_storage_structure(output_dir)
        
        # Determine profile URLs based on username or config
        if username:
            self.profile_urls = [f"https://www.instagram.com/{username}/"]
        else:
            self.profile_urls = config.INSTAGRAM_TARGET_PROFILES
            
        if not self.profile_urls:
            logger.warning("No Instagram profile URLs provided or configured.")
            
        # Initialize Apify client if token is available
        if not config.APIFY_API_TOKEN:
            logger.error("APIFY_API_TOKEN not found. Please set it in your .env file.")
            self.apify_client = None
        else:
            self.apify_client = initialize_apify_client()
            
        logger.info(f"Instagram scraper initialized for profiles: {self.profile_urls}")
    
    def scrape_user_media(self, limit: int = 10) -> List[str]:
        """
        Scrape media from the user's Instagram profile.
        
        Args:
            limit: Maximum number of posts to retrieve.
            
        Returns:
            List of paths to downloaded images.
        """
        if not self.apify_client:
            logger.error("Apify client not initialized. Cannot scrape Instagram.")
            return []
            
        # Scrape posts using the functional approach
        posts = process_instagram_posts(
            profile_urls=self.profile_urls,
            max_posts=limit,
            landscape_only=True,
            base_dir=self.output_dir,
            use_gcs=config.USE_GCS
        )
        
        # Extract image paths from processed posts
        image_paths = [post['local_path'] for post in posts if 'local_path' in post]
        
        logger.info(f"Scraped {len(image_paths)} images from Instagram")
        return image_paths

# Validation check for Apify token
if not config.APIFY_API_TOKEN:
    logger.error("APIFY_API_TOKEN not found. Please set it in your .env file.")
    # Depending on execution context, might want to raise an exception or exit

def initialize_apify_client():
    """Initializes and returns the ApifyClient with the API token."""
    if not config.APIFY_API_TOKEN:
        raise ValueError("APIFY_API_TOKEN is not configured.")
    return ApifyClient(config.APIFY_API_TOKEN)

def extract_hashtags(caption: str) -> List[str]:
    """
    Extract hashtags from a post caption.
    
    Args:
        caption: The post caption text.
        
    Returns:
        A list of hashtags found in the caption.
    """
    if not caption:
        return []
        
    # Split by spaces and filter for words starting with #
    words = caption.split()
    hashtags = [word.strip('#') for word in words if word.startswith('#')]
    
    return hashtags

def extract_post_metadata(post: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract useful metadata from an Instagram post.
    
    Args:
        post: The Instagram post data from Apify.
        
    Returns:
        A dictionary of metadata extracted from the post.
    """
    metadata = {
        'post_id': post.get('id'),
        'shortcode': post.get('shortCode'),
        'timestamp': post.get('timestamp'),
        'caption': post.get('caption', ''),
        'hashtags': extract_hashtags(post.get('caption', '')),
        'likes': post.get('likesCount', 0),
        'comments': post.get('commentsCount', 0),
        'owner_username': post.get('ownerUsername', ''),
        'owner_id': post.get('ownerId', ''),
        'url': post.get('url', ''),
        'location': post.get('locationName', ''),
        'is_video': post.get('isVideo', False),
    }
    
    return metadata

def run_instagram_scraper_for_profiles(client: ApifyClient, profile_urls: List[str], max_posts_per_profile: int = 100):
    """
    Runs the apify/instagram-scraper Actor to fetch posts from a list of Instagram profile URLs.

    Args:
        client: An initialized ApifyClient instance.
        profile_urls: A list of Instagram profile URLs to scrape.
        max_posts_per_profile: Maximum number of recent posts to retrieve from each profile.

    Returns:
        The Actor run object containing details about the scraping run, or None if an error occurs.
    """
    if not profile_urls:
        print("No profile URLs provided to scrape.")
        return None

    print(f"Starting Instagram scrape for profiles: {', '.join(profile_urls)}. Max posts per profile: {max_posts_per_profile}")
    
    actor_input = {
        "directUrls": profile_urls,
        "resultsType": "posts", # We want the post details
        "resultsLimit": max_posts_per_profile, # Max posts from each URL provided in directUrls
        # "searchLimit" and "searchType" are not primary for direct URL scraping of profiles.
        # Other useful params could be: "isUserTaggedFeedURL", "isUserReelFeedURL", "onlyPostsNewerThan"
    }

    try:
        # Ensure you have the correct Actor ID if it's not the generic one or if you're using a specific version
        actor = client.actor("apify/instagram-scraper") 
        run = actor.call(run_input=actor_input, timeout_secs=300) # Wait up to 5 minutes for potentially larger scrapes
        print(f"Scraping run for profiles finished. Run ID: {run.get('id')}, Dataset ID: {run.get('defaultDatasetId')}")
        return run
    except Exception as e:
        print(f"Error running Instagram scraper for profiles {profile_urls}: {e}")
        return None

def get_scraped_data(client: ApifyClient, run_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches items from the dataset produced by an Actor run.

    Args:
        client: An initialized ApifyClient instance.
        run_id: The ID of the Actor run.

    Returns:
        A list of items from the dataset, or None if an error occurs.
    """
    if not run_id:
        logger.error("No run_id provided to fetch scraped data.")
        return None
        
    logger.info(f"Fetching dataset for run ID: {run_id}...")
    try:
        dataset_items = client.dataset(run_id).list_items().items
        logger.info(f"Successfully fetched {len(dataset_items)} items from dataset {run_id}.")
        return dataset_items
    except Exception as e:
        logger.error(f"Error fetching dataset items for run ID {run_id}: {e}")
        return None
        
def download_images_from_posts(posts: List[Dict[str, Any]], 
                               base_dir: str = 'data',
                               min_landscape_ratio: float = 1.2,
                               landscape_only: bool = True,
                               use_gcs: bool = True) -> List[Dict[str, Any]]:
    """
    Download images from Instagram posts, filter for landscape orientation if specified,
    and store metadata.
    
    Args:
        posts: List of Instagram post data from Apify.
        base_dir: Base directory for local storage.
        min_landscape_ratio: Minimum width/height ratio to consider as landscape.
        landscape_only: Whether to filter for landscape images only.
        use_gcs: Whether to upload images to Google Cloud Storage.
        
    Returns:
        A list of processed post dictionaries with local paths and metadata.
    """
    # Create storage structure
    storage_paths = create_storage_structure(base_dir)
    
    # Initialize GCS client if needed
    gcs = GCSStorage() if use_gcs else None
    if use_gcs and not gcs.is_available():
        logger.warning("GCS client not available. Falling back to local storage only.")
        use_gcs = False
    
    processed_posts = []
    
    for i, post in enumerate(posts):
        try:
            # This function should only be called with photo posts, but double-check anyway
            if post.get('isVideo', False):
                logger.info(f"Skipping video post: {post.get('shortCode')}")
                continue
                
            # Get image URL - prefer displayUrl for highest quality
            image_url = post.get('displayUrl')
            if not image_url and 'images' in post and post['images']:
                # Fallback to first image in images array
                image_url = post['images'][0]
                
            if not image_url:
                logger.warning(f"No image URL found for post {post.get('shortCode')}")
                continue
                
            # Extract post metadata
            post_metadata = extract_post_metadata(post)
            shortcode = post.get('shortCode', f"unknown_{i}")
            
            # Generate local filename and path
            local_filename = f"{post_metadata['owner_username']}_{shortcode}.jpg"
            local_path = os.path.join(storage_paths['original'], local_filename)
            
            # Download image
            image_data = download_image(image_url, local_path)
            if not image_data:
                logger.warning(f"Failed to download image for post {shortcode}")
                continue
                
            # Check if landscape orientation
            landscape = is_landscape(image_data, min_landscape_ratio)
            post_metadata['is_landscape'] = landscape
            
            # Skip if not landscape and we only want landscape images
            if landscape_only and not landscape:
                logger.info(f"Skipping non-landscape image for post {shortcode}")
                # Delete the downloaded file
                if os.path.exists(local_path):
                    os.remove(local_path)
                continue
                
            # Extract image metadata
            image_metadata = get_image_metadata(image_data)
            post_metadata['image_metadata'] = image_metadata
            post_metadata['local_path'] = local_path
            
            # Save metadata to JSON file
            metadata_path = os.path.join(storage_paths['metadata'], f"{shortcode}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(post_metadata, f, indent=2)
                
            # Upload to GCS if configured
            if use_gcs:
                # Upload image
                gcs_image_path = f"images/original/{local_filename}"
                if gcs.upload_file(local_path, gcs_image_path):
                    post_metadata['gcs_path'] = gcs_image_path
                
                # Upload metadata
                gcs_metadata_path = f"metadata/{shortcode}_metadata.json"
                gcs.upload_file(metadata_path, gcs_metadata_path)
                
            # Add to processed posts
            processed_posts.append(post_metadata)
            logger.info(f"Successfully processed post {shortcode}")
            
        except Exception as e:
            logger.error(f"Error processing post {post.get('shortCode', 'unknown')}: {e}")
            continue
            
    logger.info(f"Downloaded {len(processed_posts)} images out of {len(posts)} posts.")
    return processed_posts

def process_instagram_posts(profile_urls: List[str] = None, 
                            max_posts: int = 100,
                            landscape_only: bool = True, 
                            min_landscape_ratio: float = 1.2,
                            base_dir: str = 'data',
                            use_gcs: bool = False,
                            content_filter_terms: List[str] = None,
                            use_content_filter: bool = False) -> List[Dict[str, Any]]:
    """
    Complete workflow to scrape Instagram posts, download images, and process metadata.
    
    Args:
        profile_urls: List of Instagram profile URLs to scrape. Defaults to config value.
        max_posts: Maximum number of posts to fetch per profile.
        landscape_only: Whether to filter for landscape images only.
        min_landscape_ratio: Minimum width/height ratio to consider as landscape.
        base_dir: Base directory for local storage.
        use_gcs: Whether to upload images to Google Cloud Storage.
        content_filter_terms: List of content terms to filter by (e.g. 'sunset', 'mountains').
                              Defaults to config.CV_CONTENT_DESCRIPTIONS_FILTER if None.
        use_content_filter: Whether to use Google Vision API for content filtering.
        
    Returns:
        A list of processed posts with image paths and metadata.
    """
    # Use config profiles if none provided
    if not profile_urls:
        profile_urls = config.INSTAGRAM_TARGET_PROFILES
        
    if not profile_urls:
        logger.error("No Instagram profile URLs provided or configured.")
        return []
    
    logger.info(f"Using Instagram profile URLs: {profile_urls}")
    
    # Initialize Apify client
    try:
        client = initialize_apify_client()
        logger.info(f"Apify client initialized with token: {config.APIFY_API_TOKEN[:5]}...")
    except Exception as e:
        logger.error(f"Failed to initialize Apify client: {e}")
        return []
    
    # Run scraper
    logger.info(f"Starting Instagram scraping process for profiles: {profile_urls}")
    scraper_run = run_instagram_scraper_for_profiles(client, profile_urls, max_posts)
    
    if not scraper_run:
        logger.error("Scraping failed: No run object returned.")
        return []
        
    if not scraper_run.get('defaultDatasetId'):
        logger.error(f"Scraping did not produce a dataset. Run details: {scraper_run}")
        return []
        
    # Get scraped data
    dataset_id = scraper_run['defaultDatasetId']
    logger.info(f"Fetching data from dataset ID: {dataset_id}")
    posts = get_scraped_data(client, dataset_id)
    
    if not posts:
        logger.error("No posts found in scraped data.")
        return []
        
    logger.info(f"Retrieved {len(posts)} posts from Instagram.")
    
    # Filter out video posts at the API level
    photo_posts = [post for post in posts if not post.get('isVideo', False)]
    logger.info(f"Filtered {len(posts)} posts to {len(photo_posts)} photo posts (excluded {len(posts) - len(photo_posts)} videos).")
    
    # Before filtering, check if we have images in the photo posts
    image_count = sum(1 for post in photo_posts if post.get('displayUrl') or (post.get('images') and post['images']))
    logger.info(f"Found {image_count} images in {len(photo_posts)} photo posts.")
    
    if image_count == 0:
        logger.error("No images found in the retrieved posts.")
        return []
        
    # Download and process images
    logger.info(f"Starting to download and process {image_count} images with settings: landscape_only={landscape_only}, min_ratio={min_landscape_ratio}")
    processed_posts = download_images_from_posts(
        photo_posts, 
        base_dir=base_dir,
        min_landscape_ratio=min_landscape_ratio,
        landscape_only=landscape_only,
        use_gcs=use_gcs
    )
    
    # Apply content filtering if requested
    if use_content_filter and processed_posts:
        # Initialize content filter
        content_filter = ImageContentFilter(use_google_vision=True)
        
        # Set content filter terms
        if content_filter_terms:
            content_filter.content_filters = content_filter_terms
        elif config.CV_CONTENT_DESCRIPTIONS_FILTER:
            content_filter.content_filters = config.CV_CONTENT_DESCRIPTIONS_FILTER
            
        if not content_filter.content_filters:
            logger.warning("No content filter terms provided. Skipping content filtering.")
        else:
            logger.info(f"Applying content filtering with terms: {content_filter.content_filters}")
            
            # Filter posts by content
            content_filtered_posts = []
            for post in processed_posts:
                image_path = post.get('local_path')
                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"Missing local path for post {post.get('shortcode')}. Skipping content filtering.")
                    continue
                    
                try:
                    # Analyze image content
                    meets_criteria, matched_filters = content_filter.meets_content_criteria(image_path=image_path)
                    
                    # Add content analysis to post metadata
                    post['content_filter_results'] = {
                        'meets_criteria': meets_criteria,
                        'matched_filters': matched_filters
                    }
                    
                    # Keep post if it meets content criteria
                    if meets_criteria:
                        content_filtered_posts.append(post)
                        logger.info(f"Post {post.get('shortcode')} meets content criteria: {matched_filters}")
                    else:
                        logger.info(f"Post {post.get('shortcode')} does not meet content criteria.")
                        # Optionally delete local file if it doesn't meet criteria
                        # if os.path.exists(image_path):
                        #     os.remove(image_path)
                except Exception as e:
                    logger.error(f"Error applying content filter to post {post.get('shortcode')}: {e}")
                    # Keep the post even if filtering fails
                    content_filtered_posts.append(post)
            
            logger.info(f"Content filtering complete. Kept {len(content_filtered_posts)} out of {len(processed_posts)} posts.")
            processed_posts = content_filtered_posts
    
    logger.info(f"Finished processing. Got {len(processed_posts)} valid posts.")
    
    # Create a detailed log of the results
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, 'scraping_results.log'), 'w') as log_file:
        log_file.write(f"Instagram Scraping Results\n")
        log_file.write(f"==========================\n")
        log_file.write(f"Profiles scraped: {', '.join(profile_urls)}\n")
        log_file.write(f"Total posts retrieved: {len(posts)}\n")
        log_file.write(f"Images found: {image_count}\n")
        log_file.write(f"Images processed: {len(processed_posts)}\n")
        log_file.write(f"Landscape filtering: {landscape_only} (min ratio: {min_landscape_ratio})\n\n")
        
        if processed_posts:
            log_file.write("Successfully processed images:\n")
            for post in processed_posts:
                log_file.write(f"- {post.get('owner_username')}/{post.get('shortcode')}: {post.get('local_path')}\n")
        else:
            log_file.write("No images were successfully processed.\n")
            log_file.write("Possible reasons:\n")
            log_file.write("1. No images meet the landscape criteria (try setting landscape_only=False)\n")
            log_file.write("2. Image download failed (check network connection)\n")
            log_file.write("3. Profile may be private or have no posts\n")
    
    return processed_posts

# Example usage (for testing this module directly)
if __name__ == '__main__':
    import sys
    import os
    import importlib

    # Determine project root (assuming this script is in src/phase1_acquisition)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Add project root to sys.path to allow importing 'src.config'
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # Load .env file from project root
    dotenv_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(dotenv_path):
        from dotenv import load_dotenv
        print(f"Loading .env file from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Environment variables might not be set.")

    # Now import src.config - it should pick up .env variables if loaded
    try:
        from src import config as test_config
        # If .env was loaded after config.py was first implicitly compiled,
        # we might need to reload it to ensure it sees the .env variables.
        importlib.reload(test_config)
    except ImportError as e:
        print(f"Error importing src.config: {e}. Ensure script is run from project root or PYTHONPATH is set.")
        sys.exit(1)

    if not test_config.APIFY_API_TOKEN:
        print("APIFY_API_TOKEN not set. Please ensure it's in your .env file in the project root and loaded correctly.")
    else:
        print("APIFY_API_TOKEN loaded successfully.")
        
        # Test the full process
        print("\n--- Testing full Instagram scraping and image download process ---")
        processed_posts = process_instagram_posts(
            max_posts=5,  # Limit to 5 posts per profile for testing
            landscape_only=True,
            base_dir=os.path.join(PROJECT_ROOT, 'data'),
            use_gcs=False  # Set to True to test GCS upload if configured
        )
        
        if processed_posts:
            print(f"Successfully processed {len(processed_posts)} posts.")
            print("\nFirst processed post details:")
            first_post = processed_posts[0]
            print(f"  Username: {first_post.get('owner_username')}")
            print(f"  Post URL: {first_post.get('url')}")
            print(f"  Local path: {first_post.get('local_path')}")
            print(f"  Is landscape: {first_post.get('is_landscape')}")
            print(f"  Hashtags: {first_post.get('hashtags')}")
            if 'image_metadata' in first_post:
                img_meta = first_post['image_metadata']
                print(f"  Image dimensions: {img_meta.get('width')}x{img_meta.get('height')}")
                print(f"  Aspect ratio: {img_meta.get('aspect_ratio')}")
        else:
            print("No posts were processed. Check the logs for errors.")
