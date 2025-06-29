#!/usr/bin/env python3
"""
Enhanced Instagram to Etsy Automation - Personal Photography Focus

Optimized for personal photography business with improved computer vision,
streamlined workflow, and better error handling.
"""

import os
import sys
import logging
import argparse
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
from datetime import datetime

# Setup enhanced logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'photography_automation_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import enhanced modules
from src.enhanced_modules.smart_cv_analyzer import SmartCVAnalyzer
from src.enhanced_modules.photography_processor import PhotographyProcessor  
from src.enhanced_modules.etsy_optimizer import EtsyOptimizer
from src.phase1_acquisition.instagram_scraper import process_instagram_posts
from src.phase3_pod_integration.printify_api import PrintifyAPI

class PersonalPhotographyWorkflow:
    """
    Enhanced workflow optimized for personal photography business.
    Focuses on quality over quantity with sophisticated CV analysis.
    """
    
    def __init__(self, base_dir: str = 'data'):
        """Initialize the enhanced workflow."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Initialize enhanced components
        self.cv_analyzer = SmartCVAnalyzer()
        self.photo_processor = PhotographyProcessor(use_gcs=False)  # Local processing for personal use
        self.etsy_optimizer = EtsyOptimizer()
        self.printify = PrintifyAPI()
        
        # Create enhanced directory structure
        self.setup_directories()
        
        logger.info("Enhanced Personal Photography Workflow initialized")
    
    def setup_directories(self):
        """Create enhanced directory structure for organized processing."""
        directories = [
            'raw_photos',
            'analyzed_photos', 
            'processed_prints',
            'etsy_ready',
            'metadata',
            'logs',
            'categories'  # For organizing by photo type
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(exist_ok=True)
    
    async def discover_and_analyze_photos(self, 
                                        profile_url: str,
                                        photo_categories: List[str],
                                        max_photos: int = 50,
                                        quality_threshold: float = 0.75) -> List[Dict[str, Any]]:
        """
        Enhanced photo discovery with sophisticated CV analysis.
        
        Args:
            profile_url: Your Instagram profile URL
            photo_categories: Categories to search for (e.g., ['sunset', 'landscape'])
            max_photos: Maximum photos to analyze
            quality_threshold: Minimum quality score (0.0-1.0)
            
        Returns:
            List of analyzed photos meeting criteria
        """
        logger.info(f"🔍 Discovering photos in categories: {photo_categories}")
        
        # Step 1: Scrape your Instagram posts
        logger.info("📱 Scraping Instagram posts...")
        raw_posts = process_instagram_posts(
            profile_urls=[profile_url],
            max_posts=max_photos * 2,  # Get extra to account for filtering
            landscape_only=True,
            base_dir=str(self.base_dir / 'raw_photos'),
            use_gcs=False,
            content_filter_terms=photo_categories,
            use_content_filter=True
        )
        
        if not raw_posts:
            logger.warning("No posts found from Instagram scraping")
            return []
        
        logger.info(f"📥 Found {len(raw_posts)} raw posts")
        
        # Step 2: Enhanced CV analysis
        logger.info("🤖 Running enhanced computer vision analysis...")
        analyzed_photos = []
        
        for post in raw_posts:
            try:
                image_path = post.get('local_path')
                if not image_path or not os.path.exists(image_path):
                    continue
                
                # Enhanced CV analysis
                analysis = await self.cv_analyzer.analyze_photography(
                    image_path=image_path,
                    target_categories=photo_categories,
                    assess_print_quality=True
                )
                
                # Quality filtering
                if analysis['overall_score'] >= quality_threshold:
                    # Add original post metadata
                    analysis['instagram_metadata'] = post
                    analysis['category_matches'] = self._find_category_matches(
                        analysis, photo_categories
                    )
                    
                    analyzed_photos.append(analysis)
                    
                    # Save detailed analysis
                    self._save_analysis(analysis, image_path)
                    
                    logger.info(f"✅ {os.path.basename(image_path)}: "
                              f"Score {analysis['overall_score']:.2f}, "
                              f"Categories: {analysis['category_matches']}")
                else:
                    logger.info(f"❌ {os.path.basename(image_path)}: "
                              f"Score {analysis['overall_score']:.2f} below threshold")
                    
            except Exception as e:
                logger.error(f"Error analyzing {image_path}: {e}")
                continue
        
        # Sort by quality score
        analyzed_photos.sort(key=lambda x: x['overall_score'], reverse=True)
        
        logger.info(f"🎯 Found {len(analyzed_photos)} high-quality photos matching criteria")
        return analyzed_photos[:max_photos]  # Return top photos
    
    def _find_category_matches(self, analysis: Dict[str, Any], 
                             target_categories: List[str]) -> List[str]:
        """Find which target categories this photo matches."""
        matches = []
        detected_labels = [label.lower() for label in analysis.get('labels', [])]
        
        for category in target_categories:
            category_lower = category.lower()
            if any(category_lower in label or label in category_lower 
                   for label in detected_labels):
                matches.append(category)
        
        return matches
    
    def _save_analysis(self, analysis: Dict[str, Any], image_path: str):
        """Save detailed CV analysis for future reference."""
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        analysis_path = self.base_dir / 'metadata' / f"{image_name}_analysis.json"
        
        # Make analysis JSON serializable
        serializable_analysis = json.loads(json.dumps(analysis, default=str))
        
        with open(analysis_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
    
    async def process_for_printing(self, analyzed_photos: List[Dict[str, Any]],
                                 print_sizes: List[str] = None,
                                 materials: List[str] = None) -> Dict[str, Any]:
        """
        Process photos for high-quality printing with multiple variants.
        
        Args:
            analyzed_photos: Photos from CV analysis
            print_sizes: Sizes to create (e.g., ['8x10', '11x14', '16x20'])
            materials: Materials to prepare (e.g., ['canvas', 'fine_art_paper'])
            
        Returns:
            Processing results with variant information
        """
        if not print_sizes:
            print_sizes = ['8x10', '11x14', '16x20', '24x36']
        if not materials:
            materials = ['canvas', 'fine_art_paper', 'metal']
        
        logger.info(f"🖼️  Processing {len(analyzed_photos)} photos for printing")
        logger.info(f"📏 Sizes: {print_sizes}")
        logger.info(f"🎨 Materials: {materials}")
        
        processing_results = {
            'successful': 0,
            'failed': 0,
            'variants_created': 0,
            'processed_photos': []
        }
        
        for photo_data in analyzed_photos:
            try:
                image_path = photo_data['instagram_metadata']['local_path']
                
                # Enhanced processing with photography-specific optimizations
                variants = await self.photo_processor.create_print_variants(
                    image_path=image_path,
                    photo_analysis=photo_data,
                    print_sizes=print_sizes,
                    materials=materials,
                    output_dir=str(self.base_dir / 'processed_prints')
                )
                
                if variants:
                    processing_results['successful'] += 1
                    processing_results['variants_created'] += len(variants)
                    processing_results['processed_photos'].append({
                        'original_path': image_path,
                        'analysis': photo_data,
                        'variants': variants
                    })
                    
                    logger.info(f"✅ Processed {os.path.basename(image_path)}: "
                              f"{len(variants)} variants created")
                else:
                    processing_results['failed'] += 1
                    logger.warning(f"❌ Failed to process {os.path.basename(image_path)}")
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                processing_results['failed'] += 1
        
        logger.info(f"🎯 Processing complete: {processing_results['successful']} successful, "
                   f"{processing_results['variants_created']} variants created")
        
        return processing_results
    
    async def create_etsy_listings(self, processing_results: Dict[str, Any],
                                 price_multiplier: float = 2.5,
                                 auto_publish: bool = True) -> Dict[str, Any]:
        """
        Create optimized Etsy listings with enhanced SEO and metadata.
        
        Args:
            processing_results: Results from photo processing
            price_multiplier: Multiplier for pricing (e.g., 2.5x cost)
            auto_publish: Whether to automatically publish to Etsy
            
        Returns:
            Listing creation results
        """
        logger.info(f"🛒 Creating Etsy listings for {len(processing_results['processed_photos'])} photos")
        
        listing_results = {
            'created': 0,
            'published': 0,
            'failed': 0,
            'listings': []
        }
        
        for photo_data in processing_results['processed_photos']:
            try:
                # Generate enhanced Etsy metadata
                etsy_metadata = await self.etsy_optimizer.generate_listing_metadata(
                    photo_analysis=photo_data['analysis'],
                    instagram_metadata=photo_data['analysis']['instagram_metadata'],
                    variants=photo_data['variants']
                )
                
                # Create Printify products
                for variant in photo_data['variants']:
                    try:
                        product_result = await self._create_printify_product(
                            variant=variant,
                            metadata=etsy_metadata,
                            price_multiplier=price_multiplier,
                            auto_publish=auto_publish
                        )
                        
                        if product_result['success']:
                            listing_results['created'] += 1
                            if product_result.get('published'):
                                listing_results['published'] += 1
                            
                            listing_results['listings'].append(product_result)
                            
                            logger.info(f"✅ Created listing: {etsy_metadata['title'][:50]}...")
                        else:
                            listing_results['failed'] += 1
                            logger.warning(f"❌ Failed to create listing for variant")
                            
                    except Exception as e:
                        logger.error(f"Error creating product for variant: {e}")
                        listing_results['failed'] += 1
                        
            except Exception as e:
                logger.error(f"Error creating listings for photo: {e}")
                listing_results['failed'] += 1
        
        logger.info(f"🎯 Listing creation complete: {listing_results['created']} created, "
                   f"{listing_results['published']} published")
        
        return listing_results
    
    async def _create_printify_product(self, variant: Dict[str, Any],
                                     metadata: Dict[str, Any],
                                     price_multiplier: float,
                                     auto_publish: bool) -> Dict[str, Any]:
        """Create a single Printify product with enhanced error handling."""
        try:
            # Find appropriate blueprint and provider
            wall_art_blueprints = self.printify.find_wall_art_blueprints()
            if not wall_art_blueprints:
                return {'success': False, 'error': 'No wall art blueprints found'}
            
            blueprint = wall_art_blueprints[0]  # Use best available
            providers = self.printify.get_print_providers(blueprint['id'])
            if not providers:
                return {'success': False, 'error': 'No print providers found'}
            
            provider = providers[0]  # Use best available
            
            # Create product
            result = self.printify.create_and_publish_product(
                image_path=variant['file_path'],
                title=metadata['title'],
                description=metadata['description'],
                blueprint_id=blueprint['id'],
                print_provider_id=provider['id'],
                tags=metadata['tags'],
                price_multiplier=price_multiplier,
                publish=auto_publish
            )
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def run_complete_workflow(self, 
                                  profile_url: str,
                                  photo_categories: List[str],
                                  max_photos: int = 20,
                                  quality_threshold: float = 0.75,
                                  print_sizes: List[str] = None,
                                  materials: List[str] = None,
                                  price_multiplier: float = 2.5,
                                  auto_publish: bool = True) -> Dict[str, Any]:
        """
        Run the complete enhanced workflow from Instagram to Etsy.
        
        Returns comprehensive results with metrics and file locations.
        """
        workflow_start = time.time()
        logger.info("🚀 Starting Enhanced Personal Photography Workflow")
        
        try:
            # Step 1: Discover and analyze photos
            analyzed_photos = await self.discover_and_analyze_photos(
                profile_url=profile_url,
                photo_categories=photo_categories,
                max_photos=max_photos,
                quality_threshold=quality_threshold
            )
            
            if not analyzed_photos:
                return {
                    'success': False,
                    'error': 'No suitable photos found',
                    'execution_time': time.time() - workflow_start
                }
            
            # Step 2: Process for printing
            processing_results = await self.process_for_printing(
                analyzed_photos=analyzed_photos,
                print_sizes=print_sizes,
                materials=materials
            )
            
            # Step 3: Create Etsy listings
            listing_results = await self.create_etsy_listings(
                processing_results=processing_results,
                price_multiplier=price_multiplier,
                auto_publish=auto_publish
            )
            
            # Generate comprehensive results
            execution_time = time.time() - workflow_start
            
            results = {
                'success': True,
                'execution_time': execution_time,
                'metrics': {
                    'photos_analyzed': len(analyzed_photos),
                    'photos_processed': processing_results['successful'],
                    'variants_created': processing_results['variants_created'],
                    'listings_created': listing_results['created'],
                    'listings_published': listing_results['published']
                },
                'categories_found': list(set(sum([p['category_matches'] for p in analyzed_photos], []))),
                'output_directories': {
                    'raw_photos': str(self.base_dir / 'raw_photos'),
                    'processed_prints': str(self.base_dir / 'processed_prints'),
                    'metadata': str(self.base_dir / 'metadata')
                },
                'detailed_results': {
                    'analyzed_photos': analyzed_photos,
                    'processing_results': processing_results,
                    'listing_results': listing_results
                }
            }
            
            # Save comprehensive results
            results_file = self.base_dir / 'metadata' / f'workflow_results_{int(time.time())}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"🎉 Workflow completed successfully in {execution_time:.1f}s")
            logger.info(f"📊 Results: {results['metrics']['photos_analyzed']} analyzed → "
                       f"{results['metrics']['listings_published']} published")
            
            return results
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - workflow_start
            }

async def main():
    """Main entry point for the enhanced workflow."""
    parser = argparse.ArgumentParser(description='Enhanced Instagram to Etsy Photography Automation')
    
    parser.add_argument('--profile', '-p', type=str, required=True,
                        help='Your Instagram profile URL')
    parser.add_argument('--categories', '-c', type=str, nargs='+', 
                        default=['sunset', 'landscape', 'mountains'],
                        help='Photo categories to search for')
    parser.add_argument('--max-photos', '-m', type=int, default=20,
                        help='Maximum photos to process')
    parser.add_argument('--quality-threshold', '-q', type=float, default=0.75,
                        help='Minimum quality score (0.0-1.0)')
    parser.add_argument('--print-sizes', '-s', type=str, nargs='+',
                        default=['8x10', '11x14', '16x20'],
                        help='Print sizes to create')
    parser.add_argument('--materials', '-mat', type=str, nargs='+',
                        default=['canvas', 'fine_art_paper'],
                        help='Materials to prepare')
    parser.add_argument('--price-multiplier', '-pm', type=float, default=2.5,
                        help='Price multiplier (e.g., 2.5x cost)')
    parser.add_argument('--no-publish', '-np', action='store_true',
                        help='Create products but don\'t publish to Etsy')
    parser.add_argument('--output-dir', '-o', type=str, default='enhanced_photography_data',
                        help='Output directory for processed files')
    
    args = parser.parse_args()
    
    # Initialize enhanced workflow
    workflow = PersonalPhotographyWorkflow(base_dir=args.output_dir)
    
    # Run complete workflow
    results = await workflow.run_complete_workflow(
        profile_url=args.profile,
        photo_categories=args.categories,
        max_photos=args.max_photos,
        quality_threshold=args.quality_threshold,
        print_sizes=args.print_sizes,
        materials=args.materials,
        price_multiplier=args.price_multiplier,
        auto_publish=not args.no_publish
    )
    
    # Print summary
    if results['success']:
        print(f"\n🎉 Workflow completed successfully!")
        print(f"📊 Processed {results['metrics']['photos_analyzed']} photos")
        print(f"🖼️  Created {results['metrics']['variants_created']} print variants")
        print(f"🛒 Published {results['metrics']['listings_published']} Etsy listings")
        print(f"⏱️  Execution time: {results['execution_time']:.1f} seconds")
        print(f"📂 Results saved to: {args.output_dir}")
    else:
        print(f"\n❌ Workflow failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())