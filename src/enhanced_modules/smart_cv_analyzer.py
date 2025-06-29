"""
Smart Computer Vision Analyzer for Photography
Enhanced analysis system optimized for photography business automation.
"""

import os
import logging
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image, ImageStat, ImageFilter
import cv2
from sklearn.cluster import KMeans
import asyncio
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

try:
    from google.cloud import vision
    from google.oauth2 import service_account
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    logger.warning("Google Vision API not available. Using local CV analysis only.")

class SmartCVAnalyzer:
    """
    Advanced computer vision analyzer optimized for photography business.
    Combines multiple analysis techniques for comprehensive image assessment.
    """
    
    def __init__(self, use_google_vision: bool = True):
        """
        Initialize the smart CV analyzer.
        
        Args:
            use_google_vision: Whether to use Google Vision API for enhanced analysis
        """
        self.use_google_vision = use_google_vision and GOOGLE_VISION_AVAILABLE
        self.vision_client = None
        
        # Initialize Google Vision if available
        if self.use_google_vision:
            try:
                from src import config
                if hasattr(config, 'GOOGLE_APPLICATION_CREDENTIALS') and config.GOOGLE_APPLICATION_CREDENTIALS:
                    credentials = service_account.Credentials.from_service_account_file(
                        config.GOOGLE_APPLICATION_CREDENTIALS
                    )
                    self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                    logger.info("Google Vision API initialized successfully")
                else:
                    logger.warning("Google Vision credentials not configured")
                    self.use_google_vision = False
            except Exception as e:
                logger.error(f"Failed to initialize Google Vision: {e}")
                self.use_google_vision = False
        
        # Photography-specific categories and keywords
        self.photography_categories = {
            'landscape': [
                'landscape', 'mountain', 'valley', 'horizon', 'countryside', 'scenic',
                'nature', 'outdoor', 'vista', 'panorama', 'terrain', 'field'
            ],
            'sunset': [
                'sunset', 'sunrise', 'golden hour', 'dusk', 'dawn', 'twilight',
                'orange sky', 'red sky', 'evening', 'morning', 'silhouette'
            ],
            'water': [
                'ocean', 'sea', 'lake', 'river', 'waterfall', 'stream', 'pond',
                'reflection', 'waves', 'beach', 'coast', 'shore', 'water'
            ],
            'mountains': [
                'mountain', 'peak', 'summit', 'range', 'hill', 'ridge', 'alpine',
                'cliff', 'rock formation', 'canyon', 'gorge'
            ],
            'forest': [
                'forest', 'trees', 'woods', 'jungle', 'pine', 'oak', 'birch',
                'leaves', 'foliage', 'canopy', 'woodland', 'grove'
            ],
            'urban': [
                'city', 'building', 'architecture', 'street', 'bridge', 'skyline',
                'urban', 'downtown', 'structure', 'tower', 'skyscraper'
            ],
            'weather': [
                'clouds', 'storm', 'rain', 'snow', 'fog', 'mist', 'lightning',
                'dramatic sky', 'overcast', 'clear sky', 'partly cloudy'
            ]
        }
        
        # Quality assessment criteria
        self.quality_weights = {
            'sharpness': 0.25,
            'composition': 0.20,
            'color_harmony': 0.20,
            'contrast': 0.15,
            'resolution': 0.10,
            'print_suitability': 0.10
        }
        
        logger.info("Smart CV Analyzer initialized")
    
    async def analyze_photography(self, 
                                image_path: str,
                                target_categories: List[str] = None,
                                assess_print_quality: bool = True) -> Dict[str, Any]:
        """
        Comprehensive photography analysis combining multiple techniques.
        
        Args:
            image_path: Path to the image file
            target_categories: Categories to specifically analyze for
            assess_print_quality: Whether to assess print suitability
            
        Returns:
            Comprehensive analysis results
        """
        logger.info(f"🔍 Analyzing photography: {os.path.basename(image_path)}")
        
        try:
            # Load image
            pil_image = Image.open(image_path)
            cv_image = cv2.imread(image_path)
            
            # Run parallel analysis
            analysis_tasks = [
                self._analyze_basic_properties(pil_image),
                self._analyze_composition(cv_image),
                self._analyze_color_harmony(pil_image),
                self._assess_technical_quality(cv_image),
            ]
            
            if self.use_google_vision:
                analysis_tasks.append(self._analyze_with_google_vision(image_path))
            else:
                analysis_tasks.append(self._analyze_content_locally(cv_image))
            
            # Execute analysis in parallel
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Combine results
            basic_props = results[0] if not isinstance(results[0], Exception) else {}
            composition = results[1] if not isinstance(results[1], Exception) else {}
            color_harmony = results[2] if not isinstance(results[2], Exception) else {}
            technical_quality = results[3] if not isinstance(results[3], Exception) else {}
            content_analysis = results[4] if not isinstance(results[4], Exception) else {}
            
            # Calculate category matches
            category_scores = self._calculate_category_scores(
                content_analysis, target_categories or []
            )
            
            # Assess print suitability
            print_assessment = {}
            if assess_print_quality:
                print_assessment = self._assess_print_suitability(
                    basic_props, technical_quality, composition
                )
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_score(
                composition, color_harmony, technical_quality, print_assessment
            )
            
            # Compile comprehensive results
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'basic_properties': basic_props,
                'composition_analysis': composition,
                'color_analysis': color_harmony,
                'technical_quality': technical_quality,
                'content_analysis': content_analysis,
                'category_scores': category_scores,
                'print_assessment': print_assessment,
                'overall_score': overall_score,
                'quality_breakdown': self._get_quality_breakdown(
                    composition, color_harmony, technical_quality, print_assessment
                ),
                # Legacy compatibility
                'labels': content_analysis.get('labels', []),
                'confidence': overall_score
            }
            
            logger.info(f"✅ Analysis complete: Overall score {overall_score:.2f}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'overall_score': 0.0
            }
    
    async def _analyze_basic_properties(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze basic image properties."""
        return {
            'width': image.width,
            'height': image.height,
            'aspect_ratio': image.width / image.height,
            'format': image.format,
            'mode': image.mode,
            'resolution_mp': (image.width * image.height) / 1_000_000,
            'is_landscape': image.width > image.height,
            'is_high_res': min(image.width, image.height) >= 1200
        }
    
    async def _analyze_composition(self, image: np.ndarray) -> Dict[str, Any]:
        """Advanced composition analysis using computer vision."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Rule of thirds analysis
        thirds_score = self._analyze_rule_of_thirds(gray)
        
        # Leading lines detection
        lines_score = self._detect_leading_lines(gray)
        
        # Symmetry analysis
        symmetry_score = self._analyze_symmetry(gray)
        
        # Edge density (complexity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Overall composition score
        composition_score = (
            thirds_score * 0.4 +
            lines_score * 0.3 +
            symmetry_score * 0.2 +
            min(edge_density * 10, 1.0) * 0.1  # Normalize edge density
        )
        
        return {
            'rule_of_thirds_score': thirds_score,
            'leading_lines_score': lines_score,
            'symmetry_score': symmetry_score,
            'edge_density': edge_density,
            'composition_score': composition_score
        }
    
    def _analyze_rule_of_thirds(self, gray_image: np.ndarray) -> float:
        """Analyze adherence to rule of thirds."""
        height, width = gray_image.shape
        
        # Define rule of thirds grid
        third_h, third_w = height // 3, width // 3
        
        # Calculate gradient magnitude to find interesting features
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Check intersection points
        intersections = [
            (third_w, third_h), (2*third_w, third_h),
            (third_w, 2*third_h), (2*third_w, 2*third_h)
        ]
        
        total_interest = 0
        for x, y in intersections:
            # Sample area around intersection
            region = magnitude[max(0, y-20):min(height, y+20), 
                             max(0, x-20):min(width, x+20)]
            if region.size > 0:
                total_interest += np.mean(region)
        
        # Normalize score
        max_possible = np.max(magnitude) * 4
        return min(total_interest / max_possible if max_possible > 0 else 0, 1.0)
    
    def _detect_leading_lines(self, gray_image: np.ndarray) -> float:
        """Detect leading lines in the image."""
        # Use Hough line detection
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0.0
        
        # Analyze line directions and convergence
        angles = []
        for rho, theta in lines[:10]:  # Analyze top 10 lines
            angles.append(theta)
        
        # Score based on number of lines and angle variety
        line_score = min(len(lines) / 20, 1.0)  # More lines generally better
        
        return line_score
    
    def _analyze_symmetry(self, gray_image: np.ndarray) -> float:
        """Analyze image symmetry."""
        height, width = gray_image.shape
        
        # Vertical symmetry
        left_half = gray_image[:, :width//2]
        right_half = np.fliplr(gray_image[:, width//2:])
        
        # Resize to match if odd width
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate correlation
        if left_half.size > 0 and right_half.size > 0:
            correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            symmetry_score = max(0, correlation) if not np.isnan(correlation) else 0
        else:
            symmetry_score = 0
        
        return symmetry_score
    
    async def _analyze_color_harmony(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze color harmony and palette."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract dominant colors using K-means
        pixels = np.array(image).reshape(-1, 3)
        
        # Sample pixels for performance
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # K-means clustering for dominant colors
        n_colors = min(5, len(pixels))
        if n_colors > 0:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            dominant_colors = kmeans.cluster_centers_.astype(int)
            color_percentages = np.bincount(kmeans.labels_) / len(pixels)
        else:
            dominant_colors = np.array([[128, 128, 128]])
            color_percentages = np.array([1.0])
        
        # Analyze color temperature
        avg_color = np.mean(pixels, axis=0)
        color_temp = self._calculate_color_temperature(avg_color)
        
        # Color harmony score based on color theory
        harmony_score = self._calculate_color_harmony_score(dominant_colors)
        
        # Saturation analysis
        hsv_pixels = cv2.cvtColor(pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)
        avg_saturation = np.mean(hsv_pixels[:, :, 1]) / 255.0
        
        return {
            'dominant_colors': [color.tolist() for color in dominant_colors],
            'color_percentages': color_percentages.tolist(),
            'color_temperature': color_temp,
            'harmony_score': harmony_score,
            'average_saturation': avg_saturation,
            'overall_color_score': (harmony_score + min(avg_saturation * 1.5, 1.0)) / 2
        }
    
    def _calculate_color_temperature(self, rgb_color: np.ndarray) -> str:
        """Calculate approximate color temperature."""
        r, g, b = rgb_color
        
        if r > g and r > b:
            return 'warm'
        elif b > r and b > g:
            return 'cool'
        else:
            return 'neutral'
    
    def _calculate_color_harmony_score(self, colors: np.ndarray) -> float:
        """Calculate color harmony based on color theory."""
        if len(colors) < 2:
            return 0.5
        
        # Convert to HSV for better color relationships
        hsv_colors = []
        for color in colors:
            hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
            hsv_colors.append(hsv)
        
        hsv_colors = np.array(hsv_colors)
        hues = hsv_colors[:, 0] / 180.0 * 360  # Convert to degrees
        
        # Analyze hue relationships
        harmony_score = 0.5  # Base score
        
        # Check for complementary colors (opposite on color wheel)
        for i, hue1 in enumerate(hues):
            for hue2 in hues[i+1:]:
                diff = abs(hue1 - hue2)
                if diff > 180:
                    diff = 360 - diff
                
                # Complementary (180°), triadic (120°), analogous (30°)
                if 170 <= diff <= 190:  # Complementary
                    harmony_score += 0.3
                elif 110 <= diff <= 130:  # Triadic
                    harmony_score += 0.2
                elif diff <= 30:  # Analogous
                    harmony_score += 0.1
        
        return min(harmony_score, 1.0)
    
    async def _assess_technical_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess technical image quality."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness assessment using Laplacian variance
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(sharpness / 1000, 1.0)  # Normalize
        
        # Contrast assessment
        contrast = gray.std()
        contrast_score = min(contrast / 64, 1.0)  # Normalize
        
        # Noise assessment
        noise_level = self._assess_noise(gray)
        noise_score = 1.0 - min(noise_level / 50, 1.0)  # Lower noise is better
        
        # Overall technical score
        technical_score = (sharpness_score * 0.4 + 
                         contrast_score * 0.4 + 
                         noise_score * 0.2)
        
        return {
            'sharpness': sharpness_score,
            'contrast': contrast_score,
            'noise_level': noise_level,
            'noise_score': noise_score,
            'technical_score': technical_score
        }
    
    def _assess_noise(self, gray_image: np.ndarray) -> float:
        """Assess image noise level."""
        # Use high-pass filter to detect noise
        blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
        noise = cv2.absdiff(gray_image, blur)
        return np.mean(noise)
    
    async def _analyze_with_google_vision(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using Google Vision API."""
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Request multiple features
            features = [
                vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=20),
                vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=10),
                vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
            ]
            
            request = vision.AnnotateImageRequest(image=image, features=features)
            response = self.vision_client.annotate_image(request)
            
            # Process response
            labels = []
            if response.label_annotations:
                labels = [
                    {
                        'description': label.description.lower(),
                        'score': label.score,
                        'topicality': label.topicality
                    }
                    for label in response.label_annotations
                ]
            
            objects = []
            if response.localized_object_annotations:
                objects = [
                    {
                        'name': obj.name.lower(),
                        'score': obj.score
                    }
                    for obj in response.localized_object_annotations
                ]
            
            return {
                'labels': [label['description'] for label in labels],
                'label_scores': labels,
                'objects': objects,
                'api_used': 'google_vision'
            }
            
        except Exception as e:
            logger.error(f"Google Vision API error: {e}")
            return {'labels': [], 'objects': [], 'api_used': 'none', 'error': str(e)}
    
    async def _analyze_content_locally(self, image: np.ndarray) -> Dict[str, Any]:
        """Local content analysis fallback."""
        # Basic local analysis - could be enhanced with local ML models
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple feature detection
        features = cv2.goodFeaturesToTrack(gray, maxCorners=100, 
                                         qualityLevel=0.01, minDistance=10)
        
        # Estimate content based on basic properties
        labels = ['photograph', 'image']
        
        # Add some basic scene classification based on color and structure
        avg_brightness = np.mean(gray)
        if avg_brightness < 80:
            labels.append('dark')
        elif avg_brightness > 180:
            labels.append('bright')
        
        return {
            'labels': labels,
            'feature_count': len(features) if features is not None else 0,
            'api_used': 'local'
        }
    
    def _calculate_category_scores(self, content_analysis: Dict[str, Any], 
                                 target_categories: List[str]) -> Dict[str, float]:
        """Calculate scores for each target category."""
        category_scores = {}
        detected_labels = [label.lower() for label in content_analysis.get('labels', [])]
        
        for category in target_categories:
            score = 0.0
            category_keywords = self.photography_categories.get(category.lower(), [category.lower()])
            
            for keyword in category_keywords:
                for label in detected_labels:
                    if keyword in label or label in keyword:
                        # Higher score for exact matches
                        if keyword == label:
                            score += 1.0
                        else:
                            score += 0.5
            
            # Normalize score
            category_scores[category] = min(score / len(category_keywords), 1.0)
        
        return category_scores
    
    def _assess_print_suitability(self, basic_props: Dict[str, Any],
                                technical_quality: Dict[str, Any],
                                composition: Dict[str, Any]) -> Dict[str, Any]:
        """Assess suitability for printing."""
        print_score = 0.0
        issues = []
        
        # Resolution check
        if basic_props.get('resolution_mp', 0) >= 2.0:
            print_score += 0.3
        else:
            issues.append('Low resolution for large prints')
        
        # Technical quality
        tech_score = technical_quality.get('technical_score', 0)
        print_score += tech_score * 0.4
        
        if tech_score < 0.6:
            issues.append('Technical quality concerns')
        
        # Composition quality
        comp_score = composition.get('composition_score', 0)
        print_score += comp_score * 0.3
        
        if comp_score < 0.5:
            issues.append('Composition could be improved')
        
        return {
            'print_suitability_score': print_score,
            'recommended_max_size': self._recommend_print_size(basic_props, print_score),
            'issues': issues,
            'suitable_for_printing': print_score >= 0.6
        }
    
    def _recommend_print_size(self, basic_props: Dict[str, Any], print_score: float) -> str:
        """Recommend maximum print size based on image properties."""
        resolution_mp = basic_props.get('resolution_mp', 0)
        
        if resolution_mp >= 6 and print_score >= 0.8:
            return '24x36'
        elif resolution_mp >= 4 and print_score >= 0.7:
            return '16x20'
        elif resolution_mp >= 2 and print_score >= 0.6:
            return '11x14'
        else:
            return '8x10'
    
    def _calculate_overall_score(self, composition: Dict[str, Any],
                               color_harmony: Dict[str, Any],
                               technical_quality: Dict[str, Any],
                               print_assessment: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        scores = {
            'composition': composition.get('composition_score', 0.5),
            'color_harmony': color_harmony.get('overall_color_score', 0.5),
            'technical': technical_quality.get('technical_score', 0.5),
            'print_suitability': print_assessment.get('print_suitability_score', 0.5)
        }
        
        # Weighted average
        overall_score = (
            scores['composition'] * self.quality_weights['composition'] +
            scores['color_harmony'] * self.quality_weights['color_harmony'] +
            scores['technical'] * (self.quality_weights['sharpness'] + self.quality_weights['contrast']) +
            scores['print_suitability'] * self.quality_weights['print_suitability']
        )
        
        return min(overall_score, 1.0)
    
    def _get_quality_breakdown(self, composition: Dict[str, Any],
                             color_harmony: Dict[str, Any],
                             technical_quality: Dict[str, Any],
                             print_assessment: Dict[str, Any]) -> Dict[str, float]:
        """Get detailed quality breakdown."""
        return {
            'composition_score': composition.get('composition_score', 0.5),
            'color_score': color_harmony.get('overall_color_score', 0.5),
            'sharpness_score': technical_quality.get('sharpness', 0.5),
            'contrast_score': technical_quality.get('contrast', 0.5),
            'print_score': print_assessment.get('print_suitability_score', 0.5)
        }