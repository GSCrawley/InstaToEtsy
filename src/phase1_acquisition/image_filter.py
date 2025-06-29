import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import json
import requests
from io import BytesIO

# Use Google Vision API for content detection if available
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

from .. import config
from ..utils.image_utils import download_image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageContentFilter:
    """Class for analyzing images and filtering based on content."""
    
    def __init__(self, use_google_vision: bool = True):
        """
        Initialize the image content filter.
        
        Args:
            use_google_vision: Whether to use Google Vision API for content detection.
                               Falls back to simpler methods if API is not available.
        """
        self.use_google_vision = use_google_vision and GOOGLE_VISION_AVAILABLE
        self.vision_client = None
        
        # Initialize Google Vision client if available and configured
        if self.use_google_vision:
            try:
                if config.GOOGLE_APPLICATION_CREDENTIALS:
                    credentials = service_account.Credentials.from_service_account_file(
                        config.GOOGLE_APPLICATION_CREDENTIALS
                    )
                    self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                    logger.info("Google Vision API client initialized successfully.")
                else:
                    logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Vision API will not be used.")
                    self.use_google_vision = False
            except Exception as e:
                logger.error(f"Error initializing Google Vision API client: {e}")
                self.use_google_vision = False
        
        # Get content filter list from config
        self.content_filters = config.CV_CONTENT_DESCRIPTIONS_FILTER or []
        if not self.content_filters:
            logger.warning("No content filters specified in config. All images will pass content filtering.")
            
        logger.info(f"Image content filter initialized. Using Google Vision: {self.use_google_vision}")
        logger.info(f"Content filters: {self.content_filters}")
        
    def analyze_image(self, image_path: str = None, image_url: str = None, image_data: bytes = None) -> Dict[str, Any]:
        """
        Analyze an image to detect its content.
        
        Args:
            image_path: Path to the local image file.
            image_url: URL of the image to analyze.
            image_data: Binary image data.
            
        Returns:
            A dictionary containing analysis results including:
            - labels: List of content labels detected in the image
            - objects: List of objects detected in the image
            - colors: List of dominant colors in the image
            - text: Text detected in the image
            - safe_search: Safe search annotations if available
        """
        # Ensure we have image data to analyze
        if image_data is None:
            if image_path:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
            elif image_url:
                image_data = download_image(image_url)
                
        if not image_data:
            logger.error("No valid image data provided for analysis")
            return {}
            
        # Analyze with Google Vision if available
        if self.use_google_vision and self.vision_client:
            return self._analyze_with_google_vision(image_data)
        else:
            # Fallback to basic analysis using PIL
            return self._analyze_basic(image_data)
            
    def _analyze_with_google_vision(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze image using Google Vision API.
        
        Args:
            image_data: Binary image data.
            
        Returns:
            Dictionary of analysis results.
        """
        try:
            # Create image object
            image = vision.Image(content=image_data)
            
            # Request features
            features = [
                vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
                vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION),
                vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
                vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
                vision.Feature(type_=vision.Feature.Type.SAFE_SEARCH_DETECTION)
            ]
            
            # Send request
            response = self.vision_client.annotate_image({
                'image': image,
                'features': features
            })
            
            # Process response
            results = {}
            
            # Extract labels
            if response.label_annotations:
                results['labels'] = [
                    {
                        'description': label.description,
                        'score': label.score,
                        'topicality': label.topicality
                    } 
                    for label in response.label_annotations
                ]
                
            # Extract objects
            if response.localized_object_annotations:
                results['objects'] = [
                    {
                        'name': obj.name,
                        'score': obj.score,
                        'bounding_poly': [
                            {'x': vertex.x, 'y': vertex.y} 
                            for vertex in obj.bounding_poly.normalized_vertices
                        ]
                    }
                    for obj in response.localized_object_annotations
                ]
                
            # Extract colors
            if response.image_properties_annotation:
                results['colors'] = [
                    {
                        'color': {
                            'red': color.color.red,
                            'green': color.color.green,
                            'blue': color.color.blue
                        },
                        'score': color.score,
                        'pixel_fraction': color.pixel_fraction
                    }
                    for color in response.image_properties_annotation.dominant_colors.colors
                ]
                
            # Extract text
            if response.text_annotations:
                results['text'] = response.text_annotations[0].description if response.text_annotations else ""
                results['text_annotations'] = [
                    {
                        'description': text.description,
                        'bounding_poly': [
                            {'x': vertex.x, 'y': vertex.y}
                            for vertex in text.bounding_poly.vertices
                        ]
                    }
                    for text in response.text_annotations[1:]  # Skip the first one which is the full text
                ]
                
            # Extract safe search
            if response.safe_search_annotation:
                results['safe_search'] = {
                    'adult': vision.SafeSearchAnnotation.Likelihood.Name(response.safe_search_annotation.adult),
                    'medical': vision.SafeSearchAnnotation.Likelihood.Name(response.safe_search_annotation.medical),
                    'spoof': vision.SafeSearchAnnotation.Likelihood.Name(response.safe_search_annotation.spoof),
                    'violence': vision.SafeSearchAnnotation.Likelihood.Name(response.safe_search_annotation.violence),
                    'racy': vision.SafeSearchAnnotation.Likelihood.Name(response.safe_search_annotation.racy)
                }
                
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing image with Google Vision: {e}")
            return {}
            
    def _analyze_basic(self, image_data: bytes) -> Dict[str, Any]:
        """
        Basic image analysis using PIL.
        
        Args:
            image_data: Binary image data.
            
        Returns:
            Dictionary with basic analysis results.
        """
        try:
            # Open image with PIL
            img = Image.open(BytesIO(image_data))
            
            # Extract basic properties
            results = {
                'format': img.format,
                'mode': img.mode,
                'width': img.width,
                'height': img.height,
                'aspect_ratio': img.width / img.height
            }
            
            # Extract dominant colors (simplified)
            img_small = img.resize((100, 100))
            if img_small.mode != 'RGB':
                img_small = img_small.convert('RGB')
                
            pixels = np.array(img_small)
            pixels = pixels.reshape(-1, 3)
            
            # Simple clustering to find dominant colors
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5)
            kmeans.fit(pixels)
            
            # Get the colors
            colors = []
            for center in kmeans.cluster_centers_:
                r, g, b = center.astype(int)
                colors.append({
                    'color': {'red': int(r), 'green': int(g), 'blue': int(b)},
                    'score': 0.0,  # Not available in this simple analysis
                    'pixel_fraction': 0.0  # Not available in this simple analysis
                })
                
            results['colors'] = colors
            
            # We don't have advanced capabilities like object detection without additional libraries
            results['labels'] = []
            results['objects'] = []
            results['text'] = ""
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing basic image analysis: {e}")
            return {}
            
    def meets_content_criteria(self, image_path: str = None, image_url: str = None, 
                              image_data: bytes = None, analysis: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        """
        Check if an image meets the content criteria based on the content filters.
        
        Args:
            image_path: Path to the local image file.
            image_url: URL of the image.
            image_data: Binary image data.
            analysis: Pre-computed image analysis results.
            
        Returns:
            A tuple containing:
            - Boolean indicating if the image meets criteria
            - List of matched content labels explaining why it matches or doesn't match
        """
        # Get analysis if not provided
        if not analysis:
            analysis = self.analyze_image(image_path, image_url, image_data)
            
        if not analysis:
            logger.warning("No analysis results available. Cannot determine if image meets criteria.")
            return False, ["No analysis results available"]
            
        # If no content filters specified, all images pass
        if not self.content_filters:
            return True, []
            
        # Extract all labels and object names from the analysis
        all_labels = []
        
        # Add labels
        if 'labels' in analysis:
            all_labels.extend([label['description'].lower() for label in analysis['labels']])
            
        # Add object names
        if 'objects' in analysis:
            all_labels.extend([obj['name'].lower() for obj in analysis['objects']])
            
        # Check if any of the labels match our content filters
        matched_filters = []
        for content_filter in self.content_filters:
            content_filter_lower = content_filter.lower()
            for label in all_labels:
                if content_filter_lower in label or label in content_filter_lower:
                    matched_filters.append(f"{label} matches {content_filter}")
                    
        # If we have any matches, the image meets our criteria
        meets_criteria = len(matched_filters) > 0
        
        if meets_criteria:
            logger.info(f"Image meets content criteria. Matched filters: {matched_filters}")
        else:
            logger.info(f"Image does not meet content criteria. No matches found.")
            
        return meets_criteria, matched_filters
        
    def save_analysis_to_file(self, analysis: Dict[str, Any], output_path: str) -> bool:
        """
        Save image analysis results to a JSON file.
        
        Args:
            analysis: Image analysis results.
            output_path: Path to save the JSON file.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
                
            logger.info(f"Analysis saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving analysis to {output_path}: {e}")
            return False

# ImageFilter class to match the import in main.py
class ImageFilter:
    """
    Wrapper class for ImageContentFilter that provides the functionality expected by main.py.
    """
    
    def __init__(self):
        """Initialize the image filter with default settings."""
        self.content_filter = ImageContentFilter()
        logger.info("ImageFilter initialized")
    
    def filter_images(self, image_paths: List[str],
                     min_width: int = 1200,
                     min_height: int = 1200,
                     aspect_ratio_range: Tuple[float, float] = (0.5, 2.0),
                     prefer_landscape: bool = True) -> List[str]:
        """
        Filter images based on dimensions, aspect ratio, and optionally content.
        
        Args:
            image_paths: List of paths to images.
            min_width: Minimum image width in pixels.
            min_height: Minimum image height in pixels.
            aspect_ratio_range: Tuple of (min_ratio, max_ratio) for filtering by aspect ratio.
            prefer_landscape: Whether to prioritize landscape-oriented images.
            
        Returns:
            Filtered list of image paths that meet the criteria.
        """
        filtered_paths = []
        
        for path in image_paths:
            try:
                # Open image to check dimensions
                with Image.open(path) as img:
                    width, height = img.size
                    aspect_ratio = width / height
                    
                    # Check dimensions
                    if width < min_width or height < min_height:
                        logger.info(f"Image {path} rejected: dimensions {width}x{height} below minimum {min_width}x{min_height}")
                        continue
                    
                    # Check aspect ratio
                    min_ratio, max_ratio = aspect_ratio_range
                    if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                        logger.info(f"Image {path} rejected: aspect ratio {aspect_ratio:.2f} outside range {min_ratio}-{max_ratio}")
                        continue
                    
                    # Check content if content filters are configured
                    if self.content_filter.content_filters:
                        meets_criteria, _ = self.content_filter.meets_content_criteria(image_path=path)
                        if not meets_criteria:
                            logger.info(f"Image {path} rejected: does not meet content criteria")
                            continue
                    
                    # Add to filtered list
                    filtered_paths.append(path)
                    logger.info(f"Image {path} accepted: dimensions {width}x{height}, aspect ratio {aspect_ratio:.2f}")
                    
            except Exception as e:
                logger.error(f"Error processing image {path}: {e}")
                continue
        
        # Sort filtered paths - put landscape images first if preferred
        if prefer_landscape and filtered_paths:
            filtered_paths.sort(key=lambda p: Image.open(p).size[0] / Image.open(p).size[1], reverse=True)
            
        logger.info(f"Filtered {len(image_paths)} images to {len(filtered_paths)} that meet criteria")
        return filtered_paths

# Example usage
if __name__ == "__main__":
    import sys
    import os
    
    # Determine project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Add project root to sys.path
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        
    # Load .env file
    dotenv_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(dotenv_path):
        from dotenv import load_dotenv
        print(f"Loading .env file from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
        
    # Get some sample images for testing
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'original')
    os.makedirs(data_dir, exist_ok=True)
    
    # Create filter
    content_filter = ImageContentFilter()
    
    # Test with local images if available
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if image_files:
        print(f"Testing with {len(image_files)} local images")
        for image_file in image_files[:3]:  # Test first 3 images
            image_path = os.path.join(data_dir, image_file)
            print(f"\nAnalyzing {image_file}...")
            
            # Analyze image
            analysis = content_filter.analyze_image(image_path=image_path)
            
            # Check if meets criteria
            meets_criteria, matched_filters = content_filter.meets_content_criteria(analysis=analysis)
            
            # Print results
            print(f"Image meets criteria: {meets_criteria}")
            if matched_filters:
                print(f"Matched filters: {matched_filters}")
                
            # Print some analysis details
            if 'labels' in analysis:
                print(f"Top 5 labels: {[label['description'] for label in analysis['labels'][:5]]}")
            if 'colors' in analysis:
                print(f"Dominant colors: {len(analysis['colors'])}")
            
            # Save analysis
            output_path = os.path.join(PROJECT_ROOT, 'data', 'metadata', f"{os.path.splitext(image_file)[0]}_analysis.json")
            content_filter.save_analysis_to_file(analysis, output_path)
    else:
        # Test with a sample URL if no local images
        test_url = "https://source.unsplash.com/1600x900/?landscape,nature"
        print(f"No local images found. Testing with URL: {test_url}")
        
        # Analyze image
        analysis = content_filter.analyze_image(image_url=test_url)
        
        # Check if meets criteria
        meets_criteria, matched_filters = content_filter.meets_content_criteria(analysis=analysis)
        
        # Print results
        print(f"Image meets criteria: {meets_criteria}")
        if matched_filters:
            print(f"Matched filters: {matched_filters}")
            
        # Print some analysis details
        if 'labels' in analysis:
            print(f"Top 5 labels: {[label['description'] for label in analysis['labels'][:5]]}")
        if 'colors' in analysis:
            print(f"Dominant colors: {len(analysis['colors'])}")
            
        # Save analysis
        output_path = os.path.join(PROJECT_ROOT, 'data', 'metadata', "sample_analysis.json")
        content_filter.save_analysis_to_file(analysis, output_path)
