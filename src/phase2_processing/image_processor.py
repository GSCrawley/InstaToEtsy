import os
import logging
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
from pathlib import Path

from .. import config
from ..utils.image_utils import get_image_metadata
from ..utils.gcs_storage import GCSStorage

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print size presets (width, height in inches)
PRINT_SIZES = {
    'small': {
        '8x10': (8, 10),
        '11x14': (11, 14),
        '12x16': (12, 16)
    },
    'medium': {
        '16x20': (16, 20),
        '18x24': (18, 24),
        '20x24': (20, 24)
    },
    'large': {
        '24x36': (24, 36),
        '30x40': (30, 40),
        '36x48': (36, 48)
    }
}

# Material presets with DPI requirements
MATERIAL_PRESETS = {
    'fine_art_paper': {
        'min_dpi': 300,
        'recommended_dpi': 360,
        'color_profile': 'sRGB',
        'format': 'TIFF'
    },
    'canvas': {
        'min_dpi': 240,
        'recommended_dpi': 300,
        'color_profile': 'sRGB',
        'format': 'TIFF'
    },
    'photo_paper': {
        'min_dpi': 300,
        'recommended_dpi': 300,
        'color_profile': 'sRGB',
        'format': 'JPEG'
    },
    'metal': {
        'min_dpi': 240,
        'recommended_dpi': 300,
        'color_profile': 'sRGB',
        'format': 'TIFF'
    },
    'acrylic': {
        'min_dpi': 300,
        'recommended_dpi': 360,
        'color_profile': 'sRGB',
        'format': 'TIFF'
    }
}

class ImageProcessor:
    """Class for processing and enhancing images for high-quality printing."""
    
    def __init__(self, use_gcs: bool = True):
        """
        Initialize the image processor.
        
        Args:
            use_gcs: Whether to use Google Cloud Storage for storing processed images.
        """
        self.use_gcs = use_gcs
        self.gcs = GCSStorage() if use_gcs else None
        
        if use_gcs and not self.gcs.is_available():
            logger.warning("GCS client not available. Falling back to local storage only.")
            self.use_gcs = False
            
        # Default enhancement parameters
        self.default_params = {
            'brightness': 1.0,    # 1.0 is original
            'contrast': 1.1,      # Slight contrast boost
            'color': 1.05,        # Slight color boost
            'sharpness': 1.2,     # Moderate sharpening
            'saturation': 1.05,   # Slight saturation boost
        }
        
        logger.info(f"Image processor initialized. Using GCS: {self.use_gcs}")
        
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load an image from a file path.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            A PIL Image object or None if loading fails.
        """
        try:
            img = Image.open(image_path)
            return img
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {e}")
            return None
            
    def enhance_image(self, img: Image.Image, params: Dict[str, float] = None) -> Image.Image:
        """
        Enhance an image using various adjustments.
        
        Args:
            img: PIL Image object to enhance.
            params: Dictionary of enhancement parameters:
                   - brightness: Brightness factor (1.0 is original)
                   - contrast: Contrast factor (1.0 is original)
                   - color: Color factor (1.0 is original)
                   - sharpness: Sharpness factor (1.0 is original)
                   - saturation: Saturation factor (1.0 is original)
                   
        Returns:
            Enhanced PIL Image object.
        """
        # Use default params if none provided
        if params is None:
            params = self.default_params
            
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Apply brightness adjustment
        if 'brightness' in params:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(params['brightness'])
            
        # Apply contrast adjustment
        if 'contrast' in params:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(params['contrast'])
            
        # Apply color adjustment
        if 'color' in params:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(params['color'])
            
        # Apply sharpness adjustment
        if 'sharpness' in params:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(params['sharpness'])
            
        # Apply saturation adjustment (requires converting to HSV and back)
        if 'saturation' in params:
            # PIL doesn't have direct saturation adjustment
            # For more advanced saturation control, we would use OpenCV or similar
            pass
            
        return img
        
    def resize_for_print(self, img: Image.Image, 
                        print_size: Tuple[int, int], 
                        dpi: int = 300, 
                        fit_method: str = 'contain') -> Image.Image:
        """
        Resize an image for printing at a specific size and DPI.
        
        Args:
            img: PIL Image object to resize.
            print_size: Tuple of (width, height) in inches.
            dpi: Target dots per inch (resolution).
            fit_method: How to fit the image:
                       - 'contain': Resize to fit within the dimensions, maintaining aspect ratio.
                       - 'cover': Resize to cover the dimensions, maintaining aspect ratio (may crop).
                       - 'stretch': Stretch/squash to exactly match dimensions.
                       
        Returns:
            Resized PIL Image object.
        """
        # Calculate pixel dimensions based on print size and DPI
        target_width_px = int(print_size[0] * dpi)
        target_height_px = int(print_size[1] * dpi)
        
        # Get current image dimensions
        orig_width, orig_height = img.size
        orig_aspect = orig_width / orig_height
        target_aspect = target_width_px / target_height_px
        
        if fit_method == 'contain':
            # Resize to fit within the dimensions, maintaining aspect ratio
            if orig_aspect > target_aspect:
                # Image is wider than target aspect, constrain by width
                new_width = target_width_px
                new_height = int(new_width / orig_aspect)
            else:
                # Image is taller than target aspect, constrain by height
                new_height = target_height_px
                new_width = int(new_height * orig_aspect)
                
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create a blank canvas of the target size
            canvas = Image.new('RGB', (target_width_px, target_height_px), color='white')
            
            # Paste the resized image centered on the canvas
            paste_x = (target_width_px - new_width) // 2
            paste_y = (target_height_px - new_height) // 2
            canvas.paste(resized_img, (paste_x, paste_y))
            
            return canvas
            
        elif fit_method == 'cover':
            # Resize to cover the dimensions, maintaining aspect ratio (may crop)
            if orig_aspect > target_aspect:
                # Image is wider than target aspect, constrain by height
                new_height = target_height_px
                new_width = int(new_height * orig_aspect)
            else:
                # Image is taller than target aspect, constrain by width
                new_width = target_width_px
                new_height = int(new_width / orig_aspect)
                
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Calculate crop coordinates
            left = (new_width - target_width_px) // 2
            top = (new_height - target_height_px) // 2
            right = left + target_width_px
            bottom = top + target_height_px
            
            # Crop to target size
            cropped_img = resized_img.crop((left, top, right, bottom))
            return cropped_img
            
        elif fit_method == 'stretch':
            # Simply stretch/squash to the target dimensions
            return img.resize((target_width_px, target_height_px), Image.LANCZOS)
            
        else:
            logger.warning(f"Unknown fit method: {fit_method}. Using 'contain'.")
            # Default to contain method
            return self.resize_for_print(img, print_size, dpi, 'contain')
            
    def apply_borders(self, img: Image.Image, 
                     border_width: Union[int, Tuple[int, int, int, int]] = 0, 
                     border_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        Apply borders to an image.
        
        Args:
            img: PIL Image object.
            border_width: Border width in pixels. Can be a single int for equal borders,
                         or a tuple of (top, right, bottom, left) for different sides.
            border_color: RGB tuple for border color.
            
        Returns:
            PIL Image with borders.
        """
        # Get image dimensions
        width, height = img.size
        
        # Convert single int to tuple if needed
        if isinstance(border_width, int):
            border_width = (border_width, border_width, border_width, border_width)
            
        # Calculate new dimensions
        new_width = width + border_width[1] + border_width[3]
        new_height = height + border_width[0] + border_width[2]
        
        # Create new image with border color
        bordered_img = Image.new('RGB', (new_width, new_height), border_color)
        
        # Paste original image onto the new image with borders
        bordered_img.paste(img, (border_width[3], border_width[0]))
        
        return bordered_img
        
    def convert_to_print_format(self, img: Image.Image, 
                              format_name: str = 'TIFF', 
                              quality: int = 95) -> Tuple[bytes, str]:
        """
        Convert an image to a print-ready format.
        
        Args:
            img: PIL Image object.
            format_name: Target format ('TIFF', 'PNG', 'JPEG', etc.).
            quality: Quality level for formats that support it.
            
        Returns:
            Tuple of (image_data_bytes, file_extension).
        """
        # Ensure format is uppercase
        format_name = format_name.upper()
        
        # Map format to extension
        format_extensions = {
            'TIFF': '.tiff',
            'JPEG': '.jpg',
            'PNG': '.png',
            'BMP': '.bmp'
        }
        
        # Get file extension
        file_ext = format_extensions.get(format_name, f'.{format_name.lower()}')
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        
        # Handle format-specific save options
        if format_name == 'JPEG':
            img.save(img_byte_arr, format=format_name, quality=quality)
        elif format_name == 'TIFF':
            img.save(img_byte_arr, format=format_name, compression='tiff_lzw')
        elif format_name == 'PNG':
            img.save(img_byte_arr, format=format_name, compress_level=int(quality / 10))
        else:
            img.save(img_byte_arr, format=format_name)
            
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue(), file_ext
        
    def generate_print_variants(self, img: Image.Image, 
                               metadata: Dict[str, Any],
                               size_categories: List[str] = None,
                               materials: List[str] = None,
                               fit_method: str = 'contain',
                               base_dir: str = 'data',
                               base_filename: str = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Generate print variants for different sizes and materials.
        
        Args:
            img: PIL Image object.
            metadata: Original image metadata.
            size_categories: List of size categories to include ('small', 'medium', 'large').
            materials: List of materials to generate variants for.
            fit_method: How to fit the image ('contain', 'cover', 'stretch').
            base_dir: Base directory for output files.
            base_filename: Base filename for output files.
            
        Returns:
            Dictionary of generated variants with paths and metadata.
        """
        # Default values if not provided
        if size_categories is None:
            size_categories = ['small', 'medium', 'large']
            
        if materials is None:
            materials = list(MATERIAL_PRESETS.keys())
            
        if base_filename is None:
            base_filename = f"processed_image_{hash(img)}"
            
        # Create processed image directory if it doesn't exist
        processed_dir = os.path.join(base_dir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # Dictionary to store results
        results = {}
        
        # Track the best variant for each size
        best_variants = {}
        
        # Process each size category
        for size_cat in size_categories:
            results[size_cat] = {}
            
            if size_cat not in PRINT_SIZES:
                logger.warning(f"Unknown size category: {size_cat}. Skipping.")
                continue
                
            # Process each size in the category
            for size_name, size_inches in PRINT_SIZES[size_cat].items():
                results[size_cat][size_name] = {}
                
                # Process each material
                for material in materials:
                    if material not in MATERIAL_PRESETS:
                        logger.warning(f"Unknown material: {material}. Skipping.")
                        continue
                        
                    # Get material settings
                    mat_settings = MATERIAL_PRESETS[material]
                    dpi = mat_settings['recommended_dpi']
                    format_name = mat_settings['format']
                    
                    # Enhanced image (apply basic enhancements)
                    enhanced_img = self.enhance_image(img)
                    
                    # Resize for print
                    resized_img = self.resize_for_print(enhanced_img, size_inches, dpi, fit_method)
                    
                    # Convert to print format
                    img_data, file_ext = self.convert_to_print_format(resized_img, format_name)
                    
                    # Generate output filename
                    output_filename = f"{base_filename}_{size_name}_{material}{file_ext}"
                    output_path = os.path.join(processed_dir, output_filename)
                    
                    # Save locally
                    with open(output_path, 'wb') as f:
                        f.write(img_data)
                        
                    # Upload to GCS if enabled
                    gcs_path = None
                    if self.use_gcs:
                        gcs_path = f"processed/{output_filename}"
                        self.gcs.upload_file(output_path, gcs_path)
                        
                    # Calculate print resolution
                    actual_width, actual_height = resized_img.size
                    actual_width_inches, actual_height_inches = size_inches
                    actual_dpi_w = actual_width / actual_width_inches
                    actual_dpi_h = actual_height / actual_height_inches
                    
                    # Store variant details
                    variant_details = {
                        'local_path': output_path,
                        'gcs_path': gcs_path,
                        'size_inches': size_inches,
                        'size_pixels': (actual_width, actual_height),
                        'dpi': (actual_dpi_w, actual_dpi_h),
                        'material': material,
                        'format': format_name,
                        'fit_method': fit_method
                    }
                    
                    results[size_cat][size_name][material] = variant_details
                    
                    # Track best variant for this size (prefer higher DPI and better materials)
                    size_key = f"{size_cat}_{size_name}"
                    if size_key not in best_variants:
                        best_variants[size_key] = variant_details
                    else:
                        # Simple heuristic: prefer higher DPI
                        current_dpi = min(best_variants[size_key]['dpi'])
                        new_dpi = min(variant_details['dpi'])
                        if new_dpi > current_dpi:
                            best_variants[size_key] = variant_details
                            
        # Save metadata with variants
        metadata_path = os.path.join(base_dir, 'metadata', f"{base_filename}_print_variants.json")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        metadata_dict = {
            'original_metadata': metadata,
            'variants': results,
            'best_variants': best_variants
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
            
        # Upload metadata to GCS
        if self.use_gcs:
            gcs_metadata_path = f"metadata/{base_filename}_print_variants.json"
            self.gcs.upload_file(metadata_path, gcs_metadata_path)
            
        return results
        
    def process_image(self, image_path: str, 
                     size_categories: List[str] = None,
                     materials: List[str] = None,
                     fit_method: str = 'contain',
                     enhancement_params: Dict[str, float] = None,
                     base_dir: str = 'data') -> Dict[str, Any]:
        """
        Process an image through the full pipeline.
        
        Args:
            image_path: Path to the input image.
            size_categories: List of size categories to include.
            materials: List of materials to generate variants for.
            fit_method: How to fit the image.
            enhancement_params: Custom enhancement parameters.
            base_dir: Base directory for output files.
            
        Returns:
            Dictionary with processing results and variant information.
        """
        # Load image
        logger.info(f"Processing image: {image_path}")
        img = self.load_image(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return {'success': False, 'error': 'Failed to load image'}
            
        # Get original metadata
        original_metadata = get_image_metadata(img)
        
        # Get base filename
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Apply enhancements
        enhanced_img = self.enhance_image(img, enhancement_params)
        
        # Generate print variants
        variants = self.generate_print_variants(
            enhanced_img,
            original_metadata,
            size_categories,
            materials,
            fit_method,
            base_dir,
            base_filename
        )
        
        # Build result
        result = {
            'success': True,
            'original_path': image_path,
            'original_metadata': original_metadata,
            'variants': variants
        }
        
        logger.info(f"Successfully processed image: {image_path}")
        
        return result
        
    def batch_process_images(self, image_paths: List[str], 
                           size_categories: List[str] = None,
                           materials: List[str] = None,
                           fit_method: str = 'contain',
                           enhancement_params: Dict[str, float] = None,
                           base_dir: str = 'data') -> Dict[str, Any]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of paths to input images.
            size_categories: List of size categories to include.
            materials: List of materials to generate variants for.
            fit_method: How to fit the image.
            enhancement_params: Custom enhancement parameters.
            base_dir: Base directory for output files.
            
        Returns:
            Dictionary with processing results for each image.
        """
        results = {}
        successful = 0
        failed = 0
        
        for path in image_paths:
            try:
                result = self.process_image(
                    path,
                    size_categories,
                    materials,
                    fit_method,
                    enhancement_params,
                    base_dir
                )
                
                results[path] = result
                if result.get('success', False):
                    successful += 1
                else:
                    failed += 1
                    
                logger.info(f"Processed {successful + failed}/{len(image_paths)} images")
            except Exception as e:
                logger.error(f"Error processing image {path}: {e}")
                results[path] = {
                    'success': False,
                    'error': str(e)
                }
                failed += 1
                
        # Create summary
        summary = {
            'total': len(image_paths),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(image_paths) if len(image_paths) > 0 else 0
        }
        
        # Save batch processing summary
        timestamp = int(time.time())
        summary_path = os.path.join(base_dir, 'metadata', f"batch_processing_summary_{timestamp}.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump({
                'summary': summary,
                'results': results
            }, f, indent=2)
            
        logger.info(f"Batch processing complete. Success rate: {summary['success_rate']:.2%}")
        
        return {
            'summary': summary,
            'results': results,
            'summary_path': summary_path
        }
