#!/usr/bin/env python3
"""
Photography Automation CLI - User-Friendly Interface
Clean command-line interface for Instagram to Etsy photography automation.
"""

import os
import sys
import asyncio
import argparse
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

# Colorful terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text: str, color: str = Colors.ENDC):
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.ENDC}")

def print_header(text: str):
    """Print a formatted header."""
    print_colored("=" * 60, Colors.CYAN)
    print_colored(f" {text}", Colors.HEADER + Colors.BOLD)
    print_colored("=" * 60, Colors.CYAN)

def print_success(text: str):
    """Print success message."""
    print_colored(f"✅ {text}", Colors.GREEN)

def print_error(text: str):
    """Print error message."""
    print_colored(f"❌ {text}", Colors.RED)

def print_warning(text: str):
    """Print warning message."""
    print_colored(f"⚠️  {text}", Colors.YELLOW)

def print_info(text: str):
    """Print info message."""
    print_colored(f"ℹ️  {text}", Colors.BLUE)

class PhotographyCLI:
    """
    User-friendly CLI for photography automation.
    Provides guided interaction and clear feedback.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.project_root = Path(__file__).parent
        
        # Set up logging to be less verbose for CLI
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s'
        )
    
    async def run_interactive_mode(self):
        """Run interactive mode with guided setup."""
        print_header("📸 Instagram to Etsy Photography Automation")
        
        print_info("Welcome! This tool will help you turn your Instagram photos into Etsy listings.")
        print_info("Let's get started with a few questions...")
        print()
        
        # Gather configuration interactively
        config = {}
        
        # Instagram profile
        config['profile'] = self._get_instagram_profile()
        
        # Photo categories
        config['categories'] = self._get_photo_categories()
        
        # Processing options
        config['max_photos'] = self._get_max_photos()
        config['quality_threshold'] = self._get_quality_threshold()
        
        # Print options
        config['print_sizes'] = self._get_print_sizes()
        config['materials'] = self._get_materials()
        config['price_multiplier'] = self._get_price_multiplier()
        
        # Publishing options
        config['auto_publish'] = self._get_publishing_preference()
        
        # Confirm and run
        print_header("🔍 Configuration Summary")
        self._display_config_summary(config)
        
        if self._confirm_execution():
            await self._run_automation(config)
        else:
            print_info("Automation cancelled by user.")
    
    def _get_instagram_profile(self) -> str:
        """Get Instagram profile URL from user."""
        print_colored("📱 Instagram Profile Setup", Colors.BOLD)
        print("Enter your Instagram profile URL (the one with your photography):")
        print("Example: https://www.instagram.com/yourphotographyaccount/")
        print()
        
        while True:
            profile = input("Instagram URL: ").strip()
            if self._validate_instagram_url(profile):
                return profile
            else:
                print_error("Please enter a valid Instagram URL")
    
    def _validate_instagram_url(self, url: str) -> bool:
        """Validate Instagram URL format."""
        return (url.startswith('https://www.instagram.com/') or 
                url.startswith('https://instagram.com/')) and len(url) > 25
    
    def _get_photo_categories(self) -> List[str]:
        """Get photo categories from user."""
        print()
        print_colored("🏷️  Photo Categories", Colors.BOLD)
        print("What types of photos should we look for? Choose from these popular categories:")
        
        available_categories = [
            "sunset", "landscape", "mountains", "water", "forest", 
            "urban", "bridges", "cityscapes", "golden hour", "nature"
        ]
        
        print("Available categories:")
        for i, category in enumerate(available_categories, 1):
            print(f"  {i}. {category}")
        
        print("\nEnter category numbers (e.g., 1,2,5) or type custom categories:")
        
        while True:
            user_input = input("Categories: ").strip()
            
            # Try to parse as numbers
            if all(c.isdigit() or c in ',. ' for c in user_input):
                try:
                    indices = [int(x.strip()) - 1 for x in user_input.split(',')]
                    categories = [available_categories[i] for i in indices if 0 <= i < len(available_categories)]
                    if categories:
                        return categories
                except (ValueError, IndexError):
                    pass
            
            # Try to parse as custom categories
            if user_input:
                categories = [cat.strip() for cat in user_input.split(',') if cat.strip()]
                if categories:
                    return categories
            
            print_error("Please enter valid category numbers or custom category names")
    
    def _get_max_photos(self) -> int:
        """Get maximum number of photos to process."""
        print()
        print_colored("📊 Processing Limits", Colors.BOLD)
        print("How many photos should we analyze? (Recommended: start with 10-20 for testing)")
        
        while True:
            try:
                max_photos = int(input("Max photos (default 10): ") or "10")
                if 1 <= max_photos <= 100:
                    return max_photos
                else:
                    print_error("Please enter a number between 1 and 100")
            except ValueError:
                print_error("Please enter a valid number")
    
    def _get_quality_threshold(self) -> float:
        """Get quality threshold."""
        print()
        print("Quality threshold (0.5-1.0, higher = more selective, default 0.75):")
        
        while True:
            try:
                threshold = float(input("Quality threshold: ") or "0.75")
                if 0.5 <= threshold <= 1.0:
                    return threshold
                else:
                    print_error("Please enter a number between 0.5 and 1.0")
            except ValueError:
                print_error("Please enter a valid number")
    
    def _get_print_sizes(self) -> List[str]:
        """Get print sizes."""
        print()
        print_colored("📏 Print Sizes", Colors.BOLD)
        print("Which print sizes should we create? (Enter numbers separated by commas)")
        
        available_sizes = ["8x10", "11x14", "16x20", "24x36"]
        for i, size in enumerate(available_sizes, 1):
            print(f"  {i}. {size} inches")
        
        while True:
            user_input = input("Print sizes (default: 1,2,3): ") or "1,2,3"
            try:
                indices = [int(x.strip()) - 1 for x in user_input.split(',')]
                sizes = [available_sizes[i] for i in indices if 0 <= i < len(available_sizes)]
                if sizes:
                    return sizes
            except (ValueError, IndexError):
                pass
            
            print_error("Please enter valid size numbers")
    
    def _get_materials(self) -> List[str]:
        """Get materials."""
        print()
        print_colored("🎨 Print Materials", Colors.BOLD)
        print("Which materials should we prepare? (Enter numbers separated by commas)")
        
        available_materials = ["canvas", "fine_art_paper", "metal", "acrylic"]
        material_descriptions = {
            "canvas": "Gallery-wrapped canvas (ready to hang)",
            "fine_art_paper": "Museum-quality paper (needs framing)",
            "metal": "Modern aluminum prints",
            "acrylic": "Premium acrylic with depth"
        }
        
        for i, material in enumerate(available_materials, 1):
            print(f"  {i}. {material} - {material_descriptions[material]}")
        
        while True:
            user_input = input("Materials (default: 1,2): ") or "1,2"
            try:
                indices = [int(x.strip()) - 1 for x in user_input.split(',')]
                materials = [available_materials[i] for i in indices if 0 <= i < len(available_materials)]
                if materials:
                    return materials
            except (ValueError, IndexError):
                pass
            
            print_error("Please enter valid material numbers")
    
    def _get_price_multiplier(self) -> float:
        """Get price multiplier."""
        print()
        print_colored("💰 Pricing", Colors.BOLD)
        print("Price multiplier (how much above cost to price items):")
        print("  2.0 = 2x cost (100% profit)")
        print("  2.5 = 2.5x cost (150% profit) - Recommended")
        print("  3.0 = 3x cost (200% profit)")
        
        while True:
            try:
                multiplier = float(input("Price multiplier (default 2.5): ") or "2.5")
                if 1.5 <= multiplier <= 5.0:
                    return multiplier
                else:
                    print_error("Please enter a number between 1.5 and 5.0")
            except ValueError:
                print_error("Please enter a valid number")
    
    def _get_publishing_preference(self) -> bool:
        """Get publishing preference."""
        print()
        print_colored("🚀 Publishing", Colors.BOLD)
        print("Should we automatically publish to Etsy, or just prepare the products?")
        print("  1. Test run - prepare only (recommended for first time)")
        print("  2. Full run - prepare and publish to Etsy")
        
        while True:
            choice = input("Choice (default 1): ").strip() or "1"
            if choice == "1":
                return False
            elif choice == "2":
                return True
            else:
                print_error("Please enter 1 or 2")
    
    def _display_config_summary(self, config: Dict[str, Any]):
        """Display configuration summary."""
        print(f"📱 Instagram Profile: {config['profile']}")
        print(f"🏷️  Categories: {', '.join(config['categories'])}")
        print(f"📊 Max Photos: {config['max_photos']}")
        print(f"🎯 Quality Threshold: {config['quality_threshold']}")
        print(f"📏 Print Sizes: {', '.join(config['print_sizes'])}")
        print(f"🎨 Materials: {', '.join(config['materials'])}")
        print(f"💰 Price Multiplier: {config['price_multiplier']}x")
        print(f"🚀 Auto Publish: {'Yes' if config['auto_publish'] else 'No (test run)'}")
        print()
    
    def _confirm_execution(self) -> bool:
        """Confirm execution with user."""
        while True:
            confirm = input("Start automation? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return True
            elif confirm in ['n', 'no']:
                return False
            else:
                print_error("Please enter 'y' for yes or 'n' for no")
    
    async def _run_automation(self, config: Dict[str, Any]):
        """Run the automation with the given configuration."""
        print_header("🚀 Starting Photography Automation")
        
        start_time = time.time()
        
        try:
            # Import the enhanced workflow (only when needed)
            sys.path.insert(0, str(self.project_root))
            from src.enhanced_main import PersonalPhotographyWorkflow
            
            # Initialize workflow
            print_info("Initializing automation workflow...")
            workflow = PersonalPhotographyWorkflow(base_dir='photography_automation_data')
            
            # Run the complete workflow
            results = await workflow.run_complete_workflow(
                profile_url=config['profile'],
                photo_categories=config['categories'],
                max_photos=config['max_photos'],
                quality_threshold=config['quality_threshold'],
                print_sizes=config['print_sizes'],
                materials=config['materials'],
                price_multiplier=config['price_multiplier'],
                auto_publish=config['auto_publish']
            )
            
            # Display results
            self._display_results(results, time.time() - start_time)
            
        except ImportError as e:
            print_error(f"Module import failed: {e}")
            print_info("Make sure you've run the setup process first!")
        except Exception as e:
            print_error(f"Automation failed: {e}")
            print_info("Check the logs for more details.")
    
    def _display_results(self, results: Dict[str, Any], execution_time: float):
        """Display execution results."""
        if results.get('success'):
            print_header("🎉 Automation Completed Successfully!")
            
            metrics = results.get('metrics', {})
            
            print_success(f"Processed {metrics.get('photos_analyzed', 0)} photos")
            print_success(f"Created {metrics.get('variants_created', 0)} print variants")
            
            if metrics.get('listings_published', 0) > 0:
                print_success(f"Published {metrics.get('listings_published', 0)} Etsy listings")
            else:
                print_info(f"Prepared {metrics.get('listings_created', 0)} products (test run)")
            
            print_info(f"Execution time: {execution_time:.1f} seconds")
            print()
            
            # Display output directories
            output_dirs = results.get('output_directories', {})
            print_colored("📂 Results Location:", Colors.BOLD)
            for name, path in output_dirs.items():
                print(f"  {name}: {path}")
            
            print()
            print_colored("📈 Categories Found:", Colors.BOLD)
            categories_found = results.get('categories_found', [])
            if categories_found:
                print(f"  {', '.join(categories_found)}")
            else:
                print_warning("No matching categories found - try different search terms")
            
            print()
            print_info("Check the output directories for your processed photos and metadata!")
            
        else:
            print_header("❌ Automation Failed")
            error = results.get('error', 'Unknown error')
            print_error(f"Error: {error}")
            print_info("Check your configuration and try again.")

def create_cli_parser():
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description='Instagram to Etsy Photography Automation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s --profile "instagram.com/user" --categories sunset landscape
  %(prog)s --quick-test                       # Quick test with 5 photos
  %(prog)s --batch --max-photos 50            # Large batch processing
        '''
    )
    
    # Basic options
    parser.add_argument('--profile', '-p', type=str,
                        help='Instagram profile URL')
    parser.add_argument('--categories', '-c', type=str, nargs='+',
                        default=['sunset', 'landscape'],
                        help='Photo categories to search for')
    parser.add_argument('--max-photos', '-m', type=int, default=10,
                        help='Maximum photos to process')
    
    # Quality options
    parser.add_argument('--quality-threshold', '-q', type=float, default=0.75,
                        help='Minimum quality score (0.5-1.0)')
    
    # Print options
    parser.add_argument('--print-sizes', '-s', type=str, nargs='+',
                        default=['8x10', '11x14', '16x20'],
                        help='Print sizes to create')
    parser.add_argument('--materials', '-mat', type=str, nargs='+',
                        default=['canvas', 'fine_art_paper'],
                        help='Materials to prepare')
    parser.add_argument('--price-multiplier', '-pm', type=float, default=2.5,
                        help='Price multiplier (e.g., 2.5x cost)')
    
    # Publishing options
    parser.add_argument('--publish', action='store_true',
                        help='Publish to Etsy (default is test run)')
    
    # Special modes
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with 5 photos, no publishing')
    parser.add_argument('--batch', action='store_true',
                        help='Batch processing mode')
    
    # Output options
    parser.add_argument('--output-dir', '-o', type=str, default='photography_automation_data',
                        help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    return parser

async def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Set up logging based on verbosity
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    cli = PhotographyCLI()
    
    # Special modes
    if args.quick_test:
        print_header("🧪 Quick Test Mode")
        config = {
            'profile': args.profile or input("Instagram URL: "),
            'categories': ['sunset', 'landscape'],
            'max_photos': 5,
            'quality_threshold': 0.7,
            'print_sizes': ['8x10', '11x14'],
            'materials': ['canvas'],
            'price_multiplier': 2.5,
            'auto_publish': False
        }
        await cli._run_automation(config)
        return
    
    if args.interactive or not args.profile:
        # Run interactive mode if no profile provided or explicitly requested
        await cli.run_interactive_mode()
    else:
        # Run with command line arguments
        config = {
            'profile': args.profile,
            'categories': args.categories,
            'max_photos': args.max_photos,
            'quality_threshold': args.quality_threshold,
            'print_sizes': args.print_sizes,
            'materials': args.materials,
            'price_multiplier': args.price_multiplier,
            'auto_publish': args.publish
        }
        
        print_header("📸 Instagram to Etsy Photography Automation")
        cli._display_config_summary(config)
        
        if cli._confirm_execution():
            await cli._run_automation(config)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_info("\nAutomation cancelled by user.")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)