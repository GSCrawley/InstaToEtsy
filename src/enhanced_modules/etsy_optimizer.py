"""
Etsy Optimizer - Intelligent Listing Creation and SEO
Generates optimized Etsy listings with enhanced SEO and metadata.
"""

import os
import logging
import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random

# Setup logging
logger = logging.getLogger(__name__)

class EtsyOptimizer:
    """
    Intelligent Etsy listing optimizer for photography businesses.
    Creates SEO-optimized titles, descriptions, and tags based on photo analysis.
    """
    
    def __init__(self):
        """Initialize the Etsy optimizer with SEO knowledge."""
        
        # High-performing Etsy keywords for photography
        self.photography_keywords = {
            'landscape': [
                'landscape photography', 'nature print', 'scenic wall art', 'landscape poster',
                'mountain print', 'nature photography', 'outdoor wall decor', 'scenic art'
            ],
            'sunset': [
                'sunset print', 'golden hour art', 'sunset photography', 'warm wall art',
                'sunset poster', 'evening sky print', 'orange sunset art', 'dusk photography'
            ],
            'water': [
                'ocean print', 'water photography', 'lake art', 'seascape print',
                'beach wall art', 'waterfall print', 'coastal decor', 'water landscape'
            ],
            'mountains': [
                'mountain photography', 'alpine print', 'peak art', 'mountain landscape',
                'summit print', 'rocky mountain art', 'mountain range print', 'alpine decor'
            ],
            'forest': [
                'forest photography', 'woodland print', 'tree art', 'forest landscape',
                'nature print', 'green wall art', 'woods photography', 'tree line print'
            ],
            'urban': [
                'city photography', 'urban art', 'cityscape print', 'architecture print',
                'street photography', 'urban landscape', 'city wall art', 'building art'
            ],
            'bridge': [
                'bridge photography', 'bridge art', 'architectural print', 'bridge poster',
                'structural art', 'engineering print', 'bridge landscape', 'span photography'
            ]
        }
        
        # High-converting descriptive words
        self.power_words = [
            'stunning', 'breathtaking', 'magnificent', 'serene', 'dramatic',
            'vibrant', 'peaceful', 'majestic', 'captivating', 'inspiring',
            'tranquil', 'spectacular', 'beautiful', 'gorgeous', 'striking'
        ]
        
        # Room/space keywords for targeting
        self.room_keywords = [
            'living room', 'bedroom', 'office', 'home decor', 'wall decor',
            'apartment decor', 'modern home', 'minimalist decor', 'rustic decor',
            'contemporary art', 'gallery wall', 'statement piece'
        ]
        
        # Print-specific terms
        self.print_terms = {
            'canvas': [
                'canvas print', 'canvas wall art', 'gallery wrap', 'canvas poster',
                'stretched canvas', 'canvas artwork', 'ready to hang'
            ],
            'fine_art_paper': [
                'fine art print', 'photo print', 'art print', 'photography print',
                'paper print', 'archival print', 'museum quality'
            ],
            'metal': [
                'metal print', 'aluminum print', 'metal wall art', 'modern print',
                'contemporary art', 'sleek wall decor', 'metal photography'
            ],
            'acrylic': [
                'acrylic print', 'acrylic wall art', 'modern art', 'contemporary print',
                'clear acrylic', 'glossy print', 'premium wall art'
            ]
        }
        
        # Season-based keywords
        self.seasonal_keywords = {
            'warm': ['summer', 'golden', 'warm tones', 'sunny', 'bright'],
            'cool': ['winter', 'blue tones', 'cool colors', 'crisp', 'fresh'],
            'neutral': ['year round', 'timeless', 'classic', 'versatile', 'elegant']
        }
        
        logger.info("Etsy Optimizer initialized with SEO knowledge base")
    
    async def generate_listing_metadata(self,
                                      photo_analysis: Dict[str, Any],
                                      instagram_metadata: Dict[str, Any],
                                      variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive Etsy listing metadata.
        
        Args:
            photo_analysis: CV analysis results
            instagram_metadata: Original Instagram post data
            variants: List of print variants created
            
        Returns:
            Optimized listing metadata
        """
        logger.info("🏷️  Generating optimized Etsy listing metadata")
        
        # Extract key information
        category_matches = photo_analysis.get('category_matches', [])
        content_analysis = photo_analysis.get('content_analysis', {})
        color_analysis = photo_analysis.get('color_analysis', {})
        basic_props = photo_analysis.get('basic_properties', {})
        
        # Generate title
        title = await self._generate_seo_title(
            category_matches, color_analysis, content_analysis, variants
        )
        
        # Generate description
        description = await self._generate_compelling_description(
            photo_analysis, instagram_metadata, variants
        )
        
        # Generate tags
        tags = await self._generate_seo_tags(
            category_matches, color_analysis, content_analysis, variants
        )
        
        # Determine category
        etsy_category = self._determine_etsy_category(category_matches, content_analysis)
        
        # Generate materials list
        materials = self._extract_materials_list(variants)
        
        # Calculate suggested pricing
        pricing_suggestions = await self._calculate_pricing_suggestions(variants, photo_analysis)
        
        metadata = {
            'title': title[:140],  # Etsy title limit
            'description': description,
            'tags': tags[:13],  # Etsy tag limit
            'category': etsy_category,
            'materials': materials,
            'pricing_suggestions': pricing_suggestions,
            'target_keywords': self._extract_target_keywords(category_matches),
            'seo_score': self._calculate_seo_score(title, description, tags),
            'generated_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Generated metadata with SEO score: {metadata['seo_score']:.2f}")
        return metadata
    
    async def _generate_seo_title(self,
                                category_matches: List[str],
                                color_analysis: Dict[str, Any],
                                content_analysis: Dict[str, Any],
                                variants: List[Dict[str, Any]]) -> str:
        """Generate SEO-optimized title."""
        
        # Start with primary category
        primary_category = category_matches[0] if category_matches else 'landscape'
        
        # Get descriptive power word
        power_word = random.choice(self.power_words)
        
        # Get primary keyword
        primary_keywords = self.photography_keywords.get(primary_category, ['photography'])
        primary_keyword = primary_keywords[0]
        
        # Get color temperature for additional descriptors
        color_temp = color_analysis.get('color_temperature', 'neutral')
        color_descriptors = self.seasonal_keywords.get(color_temp, ['beautiful'])
        color_descriptor = random.choice(color_descriptors)
        
        # Get material info
        materials = list(set([v['material'] for v in variants]))
        material_terms = []
        for material in materials[:2]:  # Max 2 materials in title
            terms = self.print_terms.get(material, [material])
            material_terms.append(terms[0])
        
        # Construct title variations and pick best
        title_patterns = [
            f"{power_word.title()} {primary_keyword.title()} - {color_descriptor.title()} Wall Art Print",
            f"{color_descriptor.title()} {primary_keyword.title()} - {power_word.title()} Photography Print",
            f"{primary_keyword.title()} Wall Art - {power_word.title()} {color_descriptor.title()} Print",
            f"{power_word.title()} {color_descriptor.title()} {primary_category.title()} Photography Print"
        ]
        
        # Add material if there's space
        if material_terms and len(title_patterns[0]) < 120:
            title_patterns.append(
                f"{power_word.title()} {primary_keyword.title()} {material_terms[0].title()}"
            )
        
        # Choose title with best keyword density
        best_title = max(title_patterns, key=lambda t: self._calculate_title_score(t, category_matches))
        
        return best_title
    
    def _calculate_title_score(self, title: str, categories: List[str]) -> float:
        """Calculate title SEO score based on keyword inclusion."""
        score = 0.0
        title_lower = title.lower()
        
        # Points for category keywords
        for category in categories:
            if category.lower() in title_lower:
                score += 2.0
            
            # Points for related keywords
            related_keywords = self.photography_keywords.get(category, [])
            for keyword in related_keywords:
                if keyword.lower() in title_lower:
                    score += 1.0
        
        # Points for power words
        for word in self.power_words:
            if word.lower() in title_lower:
                score += 0.5
        
        # Points for print terms
        for print_type in self.print_terms.values():
            for term in print_type:
                if term.lower() in title_lower:
                    score += 1.0
        
        return score
    
    async def _generate_compelling_description(self,
                                             photo_analysis: Dict[str, Any],
                                             instagram_metadata: Dict[str, Any],
                                             variants: List[Dict[str, Any]]) -> str:
        """Generate compelling, SEO-optimized description."""
        
        # Extract key information
        category_matches = photo_analysis.get('category_matches', [])
        color_analysis = photo_analysis.get('color_analysis', {})
        content_analysis = photo_analysis.get('content_analysis', {})
        print_assessment = photo_analysis.get('print_assessment', {})
        
        # Start with compelling opening
        primary_category = category_matches[0] if category_matches else 'landscape'
        power_word = random.choice(self.power_words)
        
        description_parts = []
        
        # Opening hook
        description_parts.append(
            f"Transform your space with this {power_word} {primary_category} photography print. "
            f"This high-quality wall art captures the essence of nature's beauty and brings "
            f"tranquility to any room."
        )
        
        # Description based on photo content
        if category_matches:
            category_desc = self._generate_category_description(category_matches, color_analysis)
            description_parts.append(category_desc)
        
        # Technical quality points
        if print_assessment.get('print_suitability_score', 0) > 0.7:
            description_parts.append(
                "Shot with professional photography techniques and optimized for printing, "
                "this artwork ensures exceptional clarity and vibrant colors in your home."
            )
        
        # Print options
        materials_desc = self._generate_materials_description(variants)
        description_parts.append(materials_desc)
        
        # Room placement suggestions
        room_suggestions = self._generate_room_suggestions(color_analysis, category_matches)
        description_parts.append(room_suggestions)
        
        # Features and benefits
        features = [
            "• Professional photography print optimized for home decor",
            "• Multiple size options to fit your space perfectly",
            "• High-resolution printing for exceptional detail",
            "• Ready to frame or hang (depending on material)",
            "• Perfect for gift giving or personal collection"
        ]
        
        if variants:
            max_size = max(variants, key=lambda v: v.get('dimensions_pixels', (0, 0))[0])
            features.append(f"• Available up to {max_size.get('size', '16x20')} inches")
        
        description_parts.append("FEATURES:\n" + "\n".join(features))
        
        # Care instructions
        care_instructions = self._generate_care_instructions(variants)
        if care_instructions:
            description_parts.append("CARE INSTRUCTIONS:\n" + care_instructions)
        
        # SEO keyword integration
        seo_footer = self._generate_seo_footer(category_matches, color_analysis)
        description_parts.append(seo_footer)
        
        # Join all parts
        full_description = "\n\n".join(description_parts)
        
        return full_description
    
    def _generate_category_description(self, 
                                     categories: List[str], 
                                     color_analysis: Dict[str, Any]) -> str:
        """Generate category-specific description."""
        
        primary_category = categories[0]
        color_temp = color_analysis.get('color_temperature', 'neutral')
        
        category_descriptions = {
            'landscape': [
                "This breathtaking landscape captures the raw beauty of nature's grandest vistas.",
                "Experience the serenity of wide-open spaces and natural horizons.",
                "Bring the outdoors inside with this stunning landscape photography."
            ],
            'sunset': [
                "Golden hour magic preserved in this stunning sunset photography.",
                "Warm, glowing tones create a peaceful ambiance in any space.",
                "The perfect blend of vibrant colors and natural beauty."
            ],
            'mountains': [
                "Majestic peaks that inspire adventure and contemplation.",
                "The timeless strength and beauty of mountain landscapes.",
                "Rugged natural beauty that adds character to your walls."
            ],
            'water': [
                "Flowing water brings a sense of calm and tranquility.",
                "The mesmerizing beauty of water in its natural habitat.",
                "Reflective surfaces that add depth and serenity to your space."
            ],
            'forest': [
                "Deep woodland scenes that connect you with nature's quiet strength.",
                "The peaceful energy of ancient trees and natural growth.",
                "Green spaces that refresh and revitalize your environment."
            ],
            'urban': [
                "Modern architecture and city life captured in striking detail.",
                "Urban energy and contemporary design for modern spaces.",
                "The dynamic intersection of human creativity and city life."
            ]
        }
        
        base_descriptions = category_descriptions.get(primary_category, 
                                                    ["Beautiful photography that enhances any space."])
        
        selected_description = random.choice(base_descriptions)
        
        # Add color temperature context
        if color_temp == 'warm':
            selected_description += " The warm tones create a cozy, inviting atmosphere."
        elif color_temp == 'cool':
            selected_description += " Cool tones bring a fresh, calming energy to your decor."
        
        return selected_description
    
    def _generate_materials_description(self, variants: List[Dict[str, Any]]) -> str:
        """Generate description of available materials."""
        
        materials = list(set([v['material'] for v in variants]))
        sizes = list(set([v['size'] for v in variants]))
        
        desc_parts = ["PRINT OPTIONS:"]
        
        material_descriptions = {
            'canvas': "Gallery-wrapped canvas prints ready to hang, no framing needed",
            'fine_art_paper': "Museum-quality fine art paper perfect for framing",
            'metal': "Modern aluminum prints with vibrant colors and sleek finish",
            'acrylic': "Premium acrylic prints with stunning clarity and depth"
        }
        
        for material in materials:
            if material in material_descriptions:
                desc_parts.append(f"• {material_descriptions[material]}")
        
        if sizes:
            size_list = ", ".join(sorted(sizes))
            desc_parts.append(f"• Available sizes: {size_list} inches")
        
        return "\n".join(desc_parts)
    
    def _generate_room_suggestions(self, 
                                 color_analysis: Dict[str, Any], 
                                 categories: List[str]) -> str:
        """Generate room placement suggestions."""
        
        color_temp = color_analysis.get('color_temperature', 'neutral')
        primary_category = categories[0] if categories else 'landscape'
        
        # Room suggestions based on content and color
        suggestions = []
        
        if color_temp == 'warm':
            suggestions.extend([
                "Perfect for living rooms and bedrooms where you want to create a cozy atmosphere",
                "Ideal for spaces that need warm, inviting energy"
            ])
        elif color_temp == 'cool':
            suggestions.extend([
                "Excellent for modern offices and contemporary spaces",
                "Perfect for bathrooms and kitchens where fresh energy is desired"
            ])
        else:
            suggestions.extend([
                "Versatile piece that complements any room's existing decor",
                "Ideal for any space where you want to add natural beauty"
            ])
        
        # Category-specific suggestions
        if primary_category in ['landscape', 'mountains', 'forest']:
            suggestions.append("Great for meditation spaces, offices, or anywhere you want to feel connected to nature")
        elif primary_category == 'urban':
            suggestions.append("Perfect for modern homes, offices, or urban loft spaces")
        
        return "PERFECT FOR:\n" + "\n".join([f"• {s}" for s in suggestions[:3]])
    
    def _generate_care_instructions(self, variants: List[Dict[str, Any]]) -> str:
        """Generate care instructions based on materials."""
        
        materials = list(set([v['material'] for v in variants]))
        instructions = []
        
        if 'canvas' in materials:
            instructions.append("Canvas: Dust gently with soft cloth, avoid direct sunlight")
        
        if 'fine_art_paper' in materials:
            instructions.append("Paper prints: Frame with UV-protective glass, handle with clean hands")
        
        if 'metal' in materials:
            instructions.append("Metal prints: Clean with soft microfiber cloth, avoid abrasive cleaners")
        
        if 'acrylic' in materials:
            instructions.append("Acrylic: Clean with soft cloth and mild soap solution")
        
        return "\n".join([f"• {instruction}" for instruction in instructions])
    
    def _generate_seo_footer(self, 
                           categories: List[str], 
                           color_analysis: Dict[str, Any]) -> str:
        """Generate SEO-rich footer with relevant keywords."""
        
        keywords = []
        
        # Add category keywords
        for category in categories[:2]:  # Max 2 categories
            keywords.extend(self.photography_keywords.get(category, [])[:3])
        
        # Add room keywords
        keywords.extend(random.sample(self.room_keywords, 3))
        
        # Add color-based keywords
        color_temp = color_analysis.get('color_temperature', 'neutral')
        keywords.extend(self.seasonal_keywords.get(color_temp, ['timeless'])[:2])
        
        # Remove duplicates and limit
        unique_keywords = list(set(keywords))[:8]
        
        footer = "Tags: " + " | ".join(unique_keywords)
        
        return footer
    
    async def _generate_seo_tags(self,
                               category_matches: List[str],
                               color_analysis: Dict[str, Any],
                               content_analysis: Dict[str, Any],
                               variants: List[Dict[str, Any]]) -> List[str]:
        """Generate SEO-optimized tags for Etsy."""
        
        tags = []
        
        # Category-based tags
        for category in category_matches[:2]:
            category_keywords = self.photography_keywords.get(category, [])
            tags.extend(category_keywords[:3])
        
        # Material-based tags
        materials = list(set([v['material'] for v in variants]))
        for material in materials[:2]:
            material_keywords = self.print_terms.get(material, [])
            tags.extend(material_keywords[:2])
        
        # Color-based tags
        color_temp = color_analysis.get('color_temperature', 'neutral')
        color_keywords = self.seasonal_keywords.get(color_temp, [])
        tags.extend(color_keywords[:2])
        
        # Room/decor tags
        tags.extend(random.sample(self.room_keywords, 3))
        
        # Generic high-performing tags
        generic_tags = [
            'wall art', 'home decor', 'photography print', 'wall decor',
            'art print', 'fine art', 'nature art', 'modern art'
        ]
        tags.extend(generic_tags)
        
        # Clean and limit tags
        cleaned_tags = []
        for tag in tags:
            # Clean tag
            clean_tag = re.sub(r'[^\w\s-]', '', tag.strip().lower())
            if len(clean_tag) <= 20 and clean_tag not in cleaned_tags:  # Etsy tag limits
                cleaned_tags.append(clean_tag)
        
        return cleaned_tags[:13]  # Etsy allows max 13 tags
    
    def _determine_etsy_category(self, 
                               categories: List[str], 
                               content_analysis: Dict[str, Any]) -> str:
        """Determine appropriate Etsy category."""
        
        # Etsy photography categories mapping
        etsy_categories = {
            'landscape': 'Art & Collectibles > Prints > Digital Prints',
            'sunset': 'Art & Collectibles > Prints > Photography',
            'mountains': 'Art & Collectibles > Prints > Digital Prints',
            'water': 'Art & Collectibles > Prints > Photography',
            'forest': 'Art & Collectibles > Prints > Nature & Landscape',
            'urban': 'Art & Collectibles > Prints > Architecture & Cities',
            'bridge': 'Art & Collectibles > Prints > Architecture & Cities'
        }
        
        if categories:
            return etsy_categories.get(categories[0], 'Art & Collectibles > Prints > Photography')
        
        return 'Art & Collectibles > Prints > Photography'
    
    def _extract_materials_list(self, variants: List[Dict[str, Any]]) -> List[str]:
        """Extract materials list for Etsy."""
        
        material_mapping = {
            'canvas': 'Canvas',
            'fine_art_paper': 'Paper',
            'metal': 'Metal',
            'acrylic': 'Acrylic'
        }
        
        materials = list(set([v['material'] for v in variants]))
        return [material_mapping.get(mat, mat.title()) for mat in materials]
    
    async def _calculate_pricing_suggestions(self, 
                                           variants: List[Dict[str, Any]], 
                                           photo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate pricing suggestions based on size, material, and quality."""
        
        # Base pricing matrix (in USD)
        base_prices = {
            '8x10': {'canvas': 25, 'fine_art_paper': 15, 'metal': 35, 'acrylic': 40},
            '11x14': {'canvas': 35, 'fine_art_paper': 20, 'metal': 50, 'acrylic': 55},
            '16x20': {'canvas': 55, 'fine_art_paper': 30, 'metal': 75, 'acrylic': 85},
            '24x36': {'canvas': 95, 'fine_art_paper': 50, 'metal': 135, 'acrylic': 155}
        }
        
        pricing = {}
        
        # Quality multiplier
        overall_score = photo_analysis.get('overall_score', 0.5)
        quality_multiplier = 1.0 + (overall_score - 0.5) * 0.4  # 0.8 to 1.2 range
        
        for variant in variants:
            size = variant['size']
            material = variant['material']
            
            base_price = base_prices.get(size, {}).get(material, 25)
            suggested_price = int(base_price * quality_multiplier)
            
            pricing[f"{size}_{material}"] = {
                'suggested_price': suggested_price,
                'base_price': base_price,
                'quality_multiplier': quality_multiplier,
                'competitive_range': (suggested_price - 5, suggested_price + 10)
            }
        
        return pricing
    
    def _extract_target_keywords(self, categories: List[str]) -> List[str]:
        """Extract target keywords for SEO tracking."""
        
        keywords = []
        for category in categories:
            category_keywords = self.photography_keywords.get(category, [])
            keywords.extend(category_keywords[:2])
        
        # Add universal keywords
        keywords.extend([
            'wall art', 'photography print', 'home decor', 'art print'
        ])
        
        return list(set(keywords))[:10]
    
    def _calculate_seo_score(self, title: str, description: str, tags: List[str]) -> float:
        """Calculate overall SEO score for the listing."""
        
        score = 0.0
        
        # Title score (40% weight)
        title_lower = title.lower()
        title_score = 0.0
        
        # Check for power words in title
        power_word_count = sum(1 for word in self.power_words if word in title_lower)
        title_score += min(power_word_count * 0.2, 0.4)
        
        # Check for category keywords in title
        for keywords in self.photography_keywords.values():
            for keyword in keywords:
                if keyword in title_lower:
                    title_score += 0.1
                    break
        
        score += min(title_score, 1.0) * 0.4
        
        # Description score (30% weight)
        desc_lower = description.lower()
        desc_score = 0.0
        
        # Length score
        if 500 <= len(description) <= 2000:
            desc_score += 0.3
        
        # Keyword density
        total_keywords = 0
        for keywords in self.photography_keywords.values():
            for keyword in keywords:
                if keyword in desc_lower:
                    total_keywords += 1
        
        desc_score += min(total_keywords * 0.05, 0.7)
        score += min(desc_score, 1.0) * 0.3
        
        # Tags score (30% weight)
        tag_score = min(len(tags) / 13, 1.0)  # Max 13 tags
        score += tag_score * 0.3
        
        return min(score, 1.0)
    
    async def optimize_existing_listing(self, 
                                      current_metadata: Dict[str, Any],
                                      performance_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize existing listing based on performance data.
        
        Args:
            current_metadata: Current listing metadata
            performance_data: Views, favorites, sales data
            
        Returns:
            Optimized metadata suggestions
        """
        logger.info("🔄 Optimizing existing listing based on performance")
        
        suggestions = {
            'title_suggestions': [],
            'tag_suggestions': [],
            'description_improvements': [],
            'price_adjustments': {},
            'priority_changes': []
        }
        
        current_title = current_metadata.get('title', '')
        current_tags = current_metadata.get('tags', [])
        current_seo_score = self._calculate_seo_score(
            current_title, 
            current_metadata.get('description', ''), 
            current_tags
        )
        
        # Performance-based optimization
        if performance_data:
            views = performance_data.get('views', 0)
            favorites = performance_data.get('favorites', 0)
            sales = performance_data.get('sales', 0)
            
            # Low views suggest title/tag issues
            if views < 100:
                suggestions['title_suggestions'].append(
                    "Consider adding more trending keywords to your title"
                )
                suggestions['tag_suggestions'].append(
                    "Try broader, more searched keywords"
                )
                suggestions['priority_changes'].append("Improve discoverability")
            
            # High views but low sales suggest pricing issues
            if views > 500 and sales < 5:
                suggestions['price_adjustments']['reason'] = "High views but low conversion"
                suggestions['price_adjustments']['suggestion'] = "Consider reducing prices by 10-15%"
                suggestions['priority_changes'].append("Optimize pricing")
            
            # High favorites but low sales suggest description issues
            if favorites > views * 0.1 and sales < favorites * 0.1:
                suggestions['description_improvements'].append(
                    "Add stronger call-to-action and benefits"
                )
                suggestions['priority_changes'].append("Improve description")
        
        # SEO improvements
        if current_seo_score < 0.7:
            suggestions['title_suggestions'].append(
                f"Current SEO score: {current_seo_score:.2f}. Add more descriptive keywords."
            )
        
        return suggestions