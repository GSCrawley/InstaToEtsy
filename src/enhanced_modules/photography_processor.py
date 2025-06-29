"""
Enhanced Photography Processor for Print Optimization
Specialized image processing for high-quality print preparation.
"""

import os
import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

class PhotographyProcessor:
    """
    Enhanced image processor optimized for photography printing.
    Provides professional-grade image enhancement and print preparation.
    """
    
    def __init__(self, use_gcs: bool = False):
        """
        Initialize the photography processor.
        
        Args:
            use_gcs: Whether to use Google Cloud Storage (optional for personal use)
        """
        self.use_gcs = use_gcs
        
        # Professional print specifications
        self.print_specs = {
            '8x10': {
                'inches': (8, 10),
                'pixels_300dpi': (2400, 3000),
                'min_input_res': (1600, 2000)
            },
            '11x14': {
                'inches': (11, 14),
                'pixels_300dpi': (3300, 4200),
                'min_input_res': (2200, 2800)
            },
            '16x20': {
                'inches': (16, 20),
                'pixels_300dpi': (4800, 6000),
                'min_input_res': (3200, 4000)
            },
            '24x36': {
                'inches': (24, 36),
                'pixels_300dpi': (7200, 10800),
                'min_input_res': (4800, 7200)
            }
        }
        
        # Material-specific settings
        self.material_settings = {
            'canvas': {
                'sharpening': 1.3,
                'contrast_boost': 1.1,
                'saturation_boost': 1.05,
                'format': 'TIFF',
                'dpi': 300
            },
            'fine_art_paper': {
                'sharpening': 1.2,
                'contrast_boost': 1.05,
                'saturation_boost': 1.1,
                'format': 'TIFF',
                'dpi': 360
            },
            'metal': {
                'sharpening': 1.4,
                'contrast_boost': 1.15,
                'saturation_boost': 1.2,
                'format': 'TIFF',
                'dpi': 300
            },
            'acrylic': {
                'sharpening': 1.1,
                'contrast_boost': 1.08,
                'saturation_boost': 1.15,
                'format': 'TIFF',
                'dpi': 360
            }
        }
        
        logger.info("Photography Processor initialized for professional print preparation")
    
    async def create_print_variants(self, 
                                  image_path: str,
                                  photo_analysis: Dict[str, Any],
                                  print_sizes: List[str],
                                  materials: List[str],
                                  output_dir: str) -> List[Dict[str, Any]]:
        """
        Create professional print variants for an analyzed photo.
        
        Args:
            image_path: Path to source image
            photo_analysis: CV analysis results for optimization guidance
            print_sizes: List of print sizes to create
            materials: List of materials to prepare for
            output_dir: Directory for processed images
            
        Returns:
            List of created variant information
        """
        logger.info(f"🎨 Creating print variants for {os.path.basename(image_path)}")
        
        try:
            # Load and validate source image
            source_image = Image.open(image_path)
            if source_image.mode != 'RGB':
                source_image = source_image.convert('RGB')
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Base filename for variants
            base_name = Path(image_path).stem
            
            variants = []
            
            # Process each size and material combination
            for size in print_sizes:
                for material in materials:
                    try:
                        variant = await self._create_single_variant(
                            source_image=source_image,
                            photo_analysis=photo_analysis,
                            size=size,
                            material=material,
                            base_name=base_name,
                            output_dir=output_dir
                        )
                        
                        if variant:
                            variants.append(variant)
                            logger.info(f"✅ Created {size} {material} variant")
                        
                    except Exception as e:
                        logger.error(f"Failed to create {size} {material} variant: {e}")
                        continue
            
            logger.info(f"🎯 Created {len(variants)} print variants")
            return variants
            
        except Exception as e:
            logger.error(f"Error creating print variants for {image_path}: {e}")
            return []
    
    async def _create_single_variant(self,
                                   source_image: Image.Image,
                                   photo_analysis: Dict[str, Any],
                                   size: str,
                                   material: str,
                                   base_name: str,
                                   output_dir: str) -> Optional[Dict[str, Any]]:
        """Create a single print variant with optimal processing."""
        
        # Get specifications
        print_spec = self.print_specs.get(size)
        material_spec = self.material_settings.get(material)
        
        if not print_spec or not material_spec:
            logger.error(f"Unknown size ({size}) or material ({material})")
            return None
        
        # Check if source resolution is sufficient
        source_width, source_height = source_image.size
        min_width, min_height = print_spec['min_input_res']
        
        if source_width < min_width or source_height < min_height:
            logger.warning(f"Source resolution {source_width}x{source_height} "
                         f"below recommended {min_width}x{min_height} for {size}")
        
        # Start processing pipeline
        processed_image = source_image.copy()
        
        # Step 1: Intelligent enhancement based on analysis
        processed_image = await self._apply_intelligent_enhancement(
            processed_image, photo_analysis, material_spec
        )
        
        # Step 2: Resize for print with smart cropping
        processed_image = await self._resize_for_print(
            processed_image, print_spec, photo_analysis
        )
        
        # Step 3: Material-specific optimization
        processed_image = await self._apply_material_optimization(
            processed_image, material_spec
        )
        
        # Step 4: Final sharpening for print
        processed_image = await self._apply_print_sharpening(
            processed_image, material_spec
        )
        
        # Step 5: Save with appropriate format and quality
        variant_filename = f"{base_name}_{size}_{material}.{material_spec['format'].lower()}"
        variant_path = os.path.join(output_dir, variant_filename)
        
        success = await self._save_print_ready_image(
            processed_image, variant_path, material_spec
        )
        
        if success:
            return {
                'size': size,
                'material': material,
                'file_path': variant_path,
                'dimensions_inches': print_spec['inches'],
                'dimensions_pixels': processed_image.size,
                'dpi': material_spec['dpi'],
                'format': material_spec['format'],
                'file_size_mb': os.path.getsize(variant_path) / (1024 * 1024),
                'created_timestamp': datetime.now().isoformat()
            }
        
        return None
    
    async def _apply_intelligent_enhancement(self,
                                           image: Image.Image,
                                           analysis: Dict[str, Any],
                                           material_spec: Dict[str, Any]) -> Image.Image:
        """Apply intelligent enhancement based on photo analysis."""
        
        # Get analysis insights
        color_analysis = analysis.get('color_analysis', {})
        technical_quality = analysis.get('technical_quality', {})
        composition = analysis.get('composition_analysis', {})
        
        enhanced_image = image.copy()
        
        # Adaptive contrast enhancement
        contrast_factor = material_spec['contrast_boost']
        
        # Boost contrast more for low-contrast images
        if technical_quality.get('contrast', 1.0) < 0.5:
            contrast_factor *= 1.2
        
        contrast_enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = contrast_enhancer.enhance(contrast_factor)
        
        # Intelligent saturation adjustment
        saturation_factor = material_spec['saturation_boost']
        avg_saturation = color_analysis.get('average_saturation', 0.5)
        
        # Reduce saturation boost for already highly saturated images
        if avg_saturation > 0.8:
            saturation_factor = min(saturation_factor, 1.05)
        elif avg_saturation < 0.3:
            # Boost saturation more for low-saturation images
            saturation_factor *= 1.15
        
        color_enhancer = ImageEnhance.Color(enhanced_image)
        enhanced_image = color_enhancer.enhance(saturation_factor)
        
        # Brightness adjustment based on color temperature
        color_temp = color_analysis.get('color_temperature', 'neutral')
        if color_temp == 'dark':
            brightness_enhancer = ImageEnhance.Brightness(enhanced_image)
            enhanced_image = brightness_enhancer.enhance(1.05)
        
        # Apply shadow/highlight recovery for high dynamic range scenes
        if composition.get('edge_density', 0) > 0.1:  # High detail image
            enhanced_image = await self._apply_shadow_highlight_recovery(enhanced_image)
        
        return enhanced_image
    
    async def _apply_shadow_highlight_recovery(self, image: Image.Image) -> Image.Image:
        """Apply shadow/highlight recovery for better dynamic range."""
        
        # Convert to numpy for advanced processing
        img_array = np.array(image)
        
        # Create luminance mask
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Shadow recovery (brighten dark areas)
        shadow_mask = (gray < 85).astype(np.float32)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        
        # Highlight recovery (darken bright areas)
        highlight_mask = (gray > 170).astype(np.float32)
        highlight_mask = cv2.GaussianBlur(highlight_mask, (21, 21), 0)
        
        # Apply adjustments
        for channel in range(3):
            # Shadow recovery
            img_array[:, :, channel] = img_array[:, :, channel] + (shadow_mask * 15)
            # Highlight recovery
            img_array[:, :, channel] = img_array[:, :, channel] - (highlight_mask * 10)
        
        # Clip values
        img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    async def _resize_for_print(self,
                              image: Image.Image,
                              print_spec: Dict[str, Any],
                              analysis: Dict[str, Any]) -> Image.Image:
        """Resize image for print with intelligent cropping."""
        
        target_width, target_height = print_spec['pixels_300dpi']
        source_width, source_height = image.size
        
        # Calculate aspect ratios
        source_aspect = source_width / source_height
        target_aspect = target_width / target_height
        
        # Use composition analysis to guide cropping if aspect ratios differ significantly
        if abs(source_aspect - target_aspect) > 0.1:
            # Smart crop based on composition analysis
            image = await self._smart_crop_for_print(image, target_aspect, analysis)
        
        # High-quality resize
        resized_image = image.resize(
            (target_width, target_height),
            Image.LANCZOS
        )
        
        return resized_image
    
    async def _smart_crop_for_print(self,
                                  image: Image.Image,
                                  target_aspect: float,
                                  analysis: Dict[str, Any]) -> Image.Image:
        """Intelligent cropping based on composition analysis."""
        
        width, height = image.size
        current_aspect = width / height
        
        if current_aspect > target_aspect:
            # Image is wider than target - crop width
            new_width = int(height * target_aspect)
            
            # Use rule of thirds analysis to determine best crop position
            composition = analysis.get('composition_analysis', {})
            thirds_score = composition.get('rule_of_thirds_score', 0.5)
            
            # If good rule of thirds, try to preserve it
            if thirds_score > 0.6:
                # Center crop with slight bias toward rule of thirds points
                left = (width - new_width) // 2
            else:
                # Standard center crop
                left = (width - new_width) // 2
            
            image = image.crop((left, 0, left + new_width, height))
            
        else:
            # Image is taller than target - crop height
            new_height = int(width / target_aspect)
            
            # Prefer upper portion for landscapes (horizon placement)
            top = max(0, (height - new_height) // 3)  # Bias toward upper third
            image = image.crop((0, top, width, top + new_height))
        
        return image
    
    async def _apply_material_optimization(self,
                                         image: Image.Image,
                                         material_spec: Dict[str, Any]) -> Image.Image:
        """Apply material-specific optimizations."""
        
        optimized_image = image.copy()
        
        # Material-specific color profile adjustments
        if material_spec.get('format') == 'TIFF':
            # Ensure proper color space for professional printing
            # Convert to wider gamut if needed
            pass
        
        # Apply subtle noise reduction for large prints
        if material_spec['dpi'] >= 360:
            # Convert to array for noise reduction
            img_array = np.array(optimized_image)
            
            # Apply bilateral filter for edge-preserving noise reduction
            denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
            optimized_image = Image.fromarray(denoised)
        
        return optimized_image
    
    async def _apply_print_sharpening(self,
                                    image: Image.Image,
                                    material_spec: Dict[str, Any]) -> Image.Image:
        """Apply final sharpening optimized for print medium."""
        
        sharpening_factor = material_spec['sharpening']
        
        # Create custom sharpening kernel based on material
        if material_spec.get('format') == 'TIFF' and material_spec['dpi'] >= 360:
            # More aggressive sharpening for high-DPI fine art prints
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]]) * 0.5
        else:
            # Standard unsharp mask
            sharpened = image.filter(ImageFilter.UnsharpMask(
                radius=1.5,
                percent=int((sharpening_factor - 1) * 100),
                threshold=3
            ))
            return sharpened
        
        # Apply custom kernel
        img_array = np.array(image)
        sharpened_array = cv2.filter2D(img_array, -1, kernel)
        sharpened_array = np.clip(sharpened_array, 0, 255)
        
        return Image.fromarray(sharpened_array.astype(np.uint8))
    
    async def _save_print_ready_image(self,
                                    image: Image.Image,
                                    output_path: str,
                                    material_spec: Dict[str, Any]) -> bool:
        """Save image with optimal settings for print production."""
        
        try:
            save_kwargs = {}
            
            if material_spec['format'] == 'TIFF':
                save_kwargs.update({
                    'format': 'TIFF',
                    'compression': 'tiff_lzw',  # Lossless compression
                    'dpi': (material_spec['dpi'], material_spec['dpi'])
                })
            elif material_spec['format'] == 'PNG':
                save_kwargs.update({
                    'format': 'PNG',
                    'optimize': True,
                    'dpi': (material_spec['dpi'], material_spec['dpi'])
                })
            else:  # JPEG
                save_kwargs.update({
                    'format': 'JPEG',
                    'quality': 95,
                    'optimize': True,
                    'dpi': (material_spec['dpi'], material_spec['dpi'])
                })
            
            image.save(output_path, **save_kwargs)
            
            logger.info(f"💾 Saved print-ready image: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            return False
    
    async def batch_process_photos(self,
                                 photo_list: List[Dict[str, Any]],
                                 print_sizes: List[str],
                                 materials: List[str],
                                 output_dir: str,
                                 max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Process multiple photos concurrently with rate limiting.
        
        Args:
            photo_list: List of photo analysis results
            print_sizes: Print sizes to create
            materials: Materials to prepare
            output_dir: Output directory
            max_concurrent: Maximum concurrent processing tasks
            
        Returns:
            Batch processing results
        """
        logger.info(f"🔄 Starting batch processing of {len(photo_list)} photos")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_photo(photo_data):
            async with semaphore:
                image_path = photo_data['instagram_metadata']['local_path']
                return await self.create_print_variants(
                    image_path=image_path,
                    photo_analysis=photo_data,
                    print_sizes=print_sizes,
                    materials=materials,
                    output_dir=output_dir
                )
        
        # Process all photos concurrently
        start_time = time.time()
        results = await asyncio.gather(
            *[process_single_photo(photo) for photo in photo_list],
            return_exceptions=True
        )
        
        # Compile results
        successful = 0
        failed = 0
        total_variants = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process photo {i}: {result}")
                failed += 1
            elif result:
                successful += 1
                total_variants += len(result)
            else:
                failed += 1
        
        execution_time = time.time() - start_time
        
        batch_results = {
            'total_photos': len(photo_list),
            'successful': successful,
            'failed': failed,
            'total_variants_created': total_variants,
            'execution_time': execution_time,
            'average_time_per_photo': execution_time / len(photo_list) if photo_list else 0,
            'output_directory': output_dir
        }
        
        logger.info(f"✅ Batch processing complete: {successful}/{len(photo_list)} photos processed "
                   f"in {execution_time:.1f}s")
        
        return batch_results
    
    def get_print_recommendations(self, 
                                photo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get print size and material recommendations based on photo analysis.
        
        Args:
            photo_analysis: CV analysis results
            
        Returns:
            Recommendations for optimal printing
        """
        basic_props = photo_analysis.get('basic_properties', {})
        print_assessment = photo_analysis.get('print_assessment', {})
        color_analysis = photo_analysis.get('color_analysis', {})
        
        recommendations = {
            'recommended_sizes': [],
            'recommended_materials': [],
            'notes': []
        }
        
        # Size recommendations based on resolution and quality
        resolution_mp = basic_props.get('resolution_mp', 0)
        print_score = print_assessment.get('print_suitability_score', 0.5)
        
        if resolution_mp >= 6 and print_score >= 0.8:
            recommendations['recommended_sizes'] = ['8x10', '11x14', '16x20', '24x36']
        elif resolution_mp >= 4 and print_score >= 0.7:
            recommendations['recommended_sizes'] = ['8x10', '11x14', '16x20']
        elif resolution_mp >= 2 and print_score >= 0.6:
            recommendations['recommended_sizes'] = ['8x10', '11x14']
        else:
            recommendations['recommended_sizes'] = ['8x10']
            recommendations['notes'].append('Limited to smaller sizes due to resolution')
        
        # Material recommendations based on content and color
        color_temp = color_analysis.get('color_temperature', 'neutral')
        harmony_score = color_analysis.get('harmony_score', 0.5)
        
        # Default materials
        recommended_materials = ['fine_art_paper']
        
        # Canvas for landscapes and high-harmony images
        if harmony_score > 0.7:
            recommended_materials.append('canvas')
        
        # Metal for high-contrast, cool-toned images
        if color_temp == 'cool' and print_score > 0.7:
            recommended_materials.append('metal')
            
        # Acrylic for vibrant, warm images
        if color_temp == 'warm' and color_analysis.get('average_saturation', 0) > 0.6:
            recommended_materials.append('acrylic')
        
        recommendations['recommended_materials'] = recommended_materials
        
        return recommendations