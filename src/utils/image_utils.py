import os
import requests
from PIL import Image, ExifTags
from io import BytesIO
import logging
from typing import Tuple, Dict, Optional, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_image(url: str, save_path: Optional[str] = None) -> Optional[bytes]:
    """
    Download an image from a URL.
    
    Args:
        url: The URL of the image to download.
        save_path: Optional path to save the image to. If None, image is not saved to disk.
        
    Returns:
        The image data as bytes if successful, None otherwise.
    """
    try:
        logger.info(f"Downloading image from {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Get the image data
        image_data = response.content
        
        # Save the image if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(image_data)
            logger.info(f"Image saved to {save_path}")
            
        return image_data
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return None

def get_image_dimensions(image_data: bytes) -> Optional[Tuple[int, int]]:
    """
    Get the dimensions of an image from its binary data.
    
    Args:
        image_data: The image data as bytes.
        
    Returns:
        A tuple of (width, height) if successful, None otherwise.
    """
    try:
        img = Image.open(BytesIO(image_data))
        return img.size
    except Exception as e:
        logger.error(f"Error getting image dimensions: {e}")
        return None

def is_landscape(image_data: bytes, min_ratio: float = 1.2) -> bool:
    """
    Check if an image is in landscape orientation.
    
    Args:
        image_data: The image data as bytes.
        min_ratio: The minimum width-to-height ratio to consider as landscape.
                  Default is 1.2 (20% wider than tall).
        
    Returns:
        True if the image is in landscape orientation, False otherwise.
    """
    dimensions = get_image_dimensions(image_data)
    if not dimensions:
        return False
        
    width, height = dimensions
    return width / height >= min_ratio

def get_image_metadata(image_data: bytes) -> Dict[str, Any]:
    """
    Extract metadata from an image.
    
    Args:
        image_data: The image data as bytes.
        
    Returns:
        A dictionary containing metadata about the image.
    """
    metadata = {}
    try:
        img = Image.open(BytesIO(image_data))
        metadata['format'] = img.format
        metadata['mode'] = img.mode
        metadata['width'], metadata['height'] = img.size
        metadata['aspect_ratio'] = metadata['width'] / metadata['height']
        
        # Extract EXIF data if available
        if hasattr(img, '_getexif') and img._getexif():
            exif = {
                ExifTags.TAGS[k]: v
                for k, v in img._getexif().items()
                if k in ExifTags.TAGS
            }
            metadata['exif'] = exif
    except Exception as e:
        logger.error(f"Error extracting image metadata: {e}")
    
    return metadata

def create_storage_structure(base_dir: str) -> Dict[str, str]:
    """
    Create a directory structure for storing images and metadata.
    
    Args:
        base_dir: The base directory to create the structure in.
        
    Returns:
        A dictionary mapping directory names to their paths.
    """
    structure = {
        'original': os.path.join(base_dir, 'original'),
        'processed': os.path.join(base_dir, 'processed'),
        'metadata': os.path.join(base_dir, 'metadata'),
        'temp': os.path.join(base_dir, 'temp')
    }
    
    for dir_path in structure.values():
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
        
    return structure
