#!/usr/bin/env python3
"""
Enhanced Instagram to Etsy Photography Automation - Deployment System
Easy setup and deployment for personal photography business automation.
"""

import os
import sys
import subprocess
import logging
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import platform

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PhotographyAutomationDeployer:
    """
    Deployment system for the Instagram to Etsy photography automation.
    Handles setup, configuration, and deployment.
    """
    
    def __init__(self):
        """Initialize the deployer."""
        self.project_root = Path(__file__).parent
        self.system_os = platform.system().lower()
        
        # Required Python packages
        self.required_packages = [
            'requests>=2.31.0',
            'Pillow>=10.0.0',
            'opencv-python>=4.8.0',
            'numpy>=1.24.0',
            'scikit-learn>=1.3.0',
            'google-cloud-vision>=3.4.0',
            'google-cloud-storage>=2.10.0',
            'python-dotenv>=1.0.0',
            'apify-client>=1.11.0',
            'asyncio',
            'aiofiles>=23.0.0'
        ]
        
        # Optional packages for enhanced functionality
        self.optional_packages = [
            'tensorflow>=2.13.0',  # For advanced CV models
            'torch>=2.0.0',        # Alternative ML framework
            'ultralytics>=8.0.0',  # YOLO models
            'transformers>=4.30.0' # For text analysis
        ]
        
        logger.info("Photography Automation Deployer initialized")
    
    def setup_environment(self, enhanced_cv: bool = False) -> bool:
        """
        Set up the complete environment for the photography automation.
        
        Args:
            enhanced_cv: Whether to install enhanced CV packages
            
        Returns:
            Success status
        """
        logger.info("🚀 Setting up Photography Automation environment")
        
        try:
            # Step 1: Check Python version
            if not self._check_python_version():
                return False
            
            # Step 2: Create virtual environment
            if not self._create_virtual_environment():
                return False
            
            # Step 3: Install packages
            if not self._install_packages(enhanced_cv):
                return False
            
            # Step 4: Setup directory structure
            self._setup_directory_structure()
            
            # Step 5: Create configuration files
            self._create_configuration_files()
            
            # Step 6: Setup environment variables
            self._setup_environment_variables()
            
            # Step 7: Validate setup
            if self._validate_setup():
                logger.info("✅ Environment setup completed successfully!")
                self._print_next_steps()
                return True
            else:
                logger.error("❌ Environment setup validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    def _check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= min_version:
            logger.info(f"✅ Python {'.'.join(map(str, current_version))} is compatible")
            return True
        else:
            logger.error(f"❌ Python {'.'.join(map(str, min_version))} or higher required. "
                        f"Current: {'.'.join(map(str, current_version))}")
            return False
    
    def _create_virtual_environment(self) -> bool:
        """Create a virtual environment for the project."""
        venv_path = self.project_root / 'photography_env'
        
        if venv_path.exists():
            logger.info("📁 Virtual environment already exists")
            return True
        
        try:
            logger.info("📦 Creating virtual environment...")
            subprocess.run([
                sys.executable, '-m', 'venv', str(venv_path)
            ], check=True, capture_output=True)
            
            logger.info("✅ Virtual environment created successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def _install_packages(self, enhanced_cv: bool = False) -> bool:
        """Install required packages."""
        venv_path = self.project_root / 'photography_env'
        
        # Determine pip executable path
        if self.system_os == 'windows':
            pip_exe = venv_path / 'Scripts' / 'pip.exe'
        else:
            pip_exe = venv_path / 'bin' / 'pip'
        
        if not pip_exe.exists():
            logger.error(f"❌ Pip not found at {pip_exe}")
            return False
        
        try:
            # Upgrade pip first
            logger.info("📦 Upgrading pip...")
            subprocess.run([
                str(pip_exe), 'install', '--upgrade', 'pip'
            ], check=True, capture_output=True)
            
            # Install required packages
            logger.info("📦 Installing required packages...")
            for package in self.required_packages:
                logger.info(f"Installing {package}...")
                subprocess.run([
                    str(pip_exe), 'install', package
                ], check=True, capture_output=True)
            
            # Install enhanced CV packages if requested
            if enhanced_cv:
                logger.info("🤖 Installing enhanced computer vision packages...")
                for package in self.optional_packages:
                    try:
                        logger.info(f"Installing {package}...")
                        subprocess.run([
                            str(pip_exe), 'install', package
                        ], check=True, capture_output=True)
                    except subprocess.CalledProcessError:
                        logger.warning(f"⚠️  Failed to install optional package: {package}")
            
            logger.info("✅ Package installation completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Package installation failed: {e}")
            return False
    
    def _setup_directory_structure(self):
        """Create the enhanced directory structure."""
        directories = [
            'src/enhanced_modules',
            'data/raw_photos',
            'data/analyzed_photos', 
            'data/processed_prints',
            'data/etsy_ready',
            'data/metadata',
            'data/logs',
            'data/categories',
            'config',
            'logs',
            'tests',
            'scripts',
            'docs'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")
    
    def _create_configuration_files(self):
        """Create configuration files."""
        
        # Enhanced config.py
        config_content = '''"""
Enhanced Configuration for Instagram to Etsy Photography Automation
Personal photography business optimized settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Credentials
APIFY_API_TOKEN = os.getenv('APIFY_API_TOKEN')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
PRINTIFY_API_TOKEN = os.getenv('PRINTIFY_API_TOKEN')
PRINTIFY_SHOP_ID = os.getenv('PRINTIFY_SHOP_ID')

# Google Cloud Storage (Optional)
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
GCS_PROJECT_ID = os.getenv('GCS_PROJECT_ID')

# Instagram Configuration
INSTAGRAM_TARGET_PROFILES = os.getenv('INSTAGRAM_TARGET_PROFILES', '').split(',')

# Computer Vision Settings
CV_CONTENT_DESCRIPTIONS_FILTER = os.getenv('CV_CONTENT_DESCRIPTIONS_FILTER', '').split(',')
USE_GOOGLE_VISION = os.getenv('USE_GOOGLE_VISION', 'True').lower() == 'true'

# Processing Settings
MAX_CONCURRENT_PROCESSING = int(os.getenv('MAX_CONCURRENT_PROCESSING', '3'))
DEFAULT_QUALITY_THRESHOLD = float(os.getenv('DEFAULT_QUALITY_THRESHOLD', '0.75'))

# Print Settings
DEFAULT_PRINT_SIZES = os.getenv('DEFAULT_PRINT_SIZES', '8x10,11x14,16x20').split(',')
DEFAULT_MATERIALS = os.getenv('DEFAULT_MATERIALS', 'canvas,fine_art_paper').split(',')
DEFAULT_PRICE_MULTIPLIER = float(os.getenv('DEFAULT_PRICE_MULTIPLIER', '2.5'))

# Etsy Optimization
AUTO_PUBLISH_TO_ETSY = os.getenv('AUTO_PUBLISH_TO_ETSY', 'True').lower() == 'true'
USE_ENHANCED_SEO = os.getenv('USE_ENHANCED_SEO', 'True').lower() == 'true'

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_TO_FILE = os.getenv('LOG_TO_FILE', 'True').lower() == 'true'

# Safety Settings
MAX_IMAGES_PER_RUN = int(os.getenv('MAX_IMAGES_PER_RUN', '50'))
RATE_LIMIT_DELAY = float(os.getenv('RATE_LIMIT_DELAY', '1.0'))
'''
        
        config_path = self.project_root / 'src' / 'enhanced_config.py'
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Create __init__.py files
        init_files = [
            'src/__init__.py',
            'src/enhanced_modules/__init__.py'
        ]
        
        for init_file in init_files:
            init_path = self.project_root / init_file
            init_path.touch()
    
    def _setup_environment_variables(self):
        """Create example environment variables file."""
        env_content = '''# Instagram to Etsy Photography Automation - Environment Variables
# Copy this file to .env and fill in your actual credentials

# REQUIRED: Apify API Token (for Instagram scraping)
APIFY_API_TOKEN=your_apify_api_token_here

# REQUIRED: Printify API Credentials
PRINTIFY_API_TOKEN=your_printify_api_token_here
PRINTIFY_SHOP_ID=your_printify_shop_id_here

# REQUIRED: Your Instagram Profile URL
INSTAGRAM_TARGET_PROFILES=https://www.instagram.com/your_photography_account/

# OPTIONAL: Google Cloud Vision API (for enhanced image analysis)
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/google-credentials.json
GCS_PROJECT_ID=your-google-cloud-project-id
GCS_BUCKET_NAME=your-gcs-bucket-name

# Photography Categories to Search For
CV_CONTENT_DESCRIPTIONS_FILTER=sunset,landscape,mountains,bridges,cityscapes

# Processing Settings
MAX_CONCURRENT_PROCESSING=3
DEFAULT_QUALITY_THRESHOLD=0.75

# Print Settings
DEFAULT_PRINT_SIZES=8x10,11x14,16x20,24x36
DEFAULT_MATERIALS=canvas,fine_art_paper,metal
DEFAULT_PRICE_MULTIPLIER=2.5

# Etsy Settings
AUTO_PUBLISH_TO_ETSY=True
USE_ENHANCED_SEO=True

# Safety Settings
MAX_IMAGES_PER_RUN=50
RATE_LIMIT_DELAY=1.0

# Advanced Settings
USE_GOOGLE_VISION=True
LOG_LEVEL=INFO
LOG_TO_FILE=True
'''
        
        env_example_path = self.project_root / '.env.example'
        with open(env_example_path, 'w') as f:
            f.write(env_content)
        
        logger.info("📝 Created .env.example file - copy to .env and add your credentials")
    
    def _validate_setup(self) -> bool:
        """Validate the setup is working correctly."""
        venv_path = self.project_root / 'photography_env'
        
        # Determine python executable path
        if self.system_os == 'windows':
            python_exe = venv_path / 'Scripts' / 'python.exe'
        else:
            python_exe = venv_path / 'bin' / 'python'
        
        if not python_exe.exists():
            logger.error(f"❌ Python executable not found at {python_exe}")
            return False
        
        try:
            # Test imports
            test_script = '''
import sys
try:
    import requests
    import PIL
    import cv2
    import numpy
    import sklearn
    from google.cloud import vision, storage
    from apify_client import ApifyClient
    print("✅ All required packages imported successfully")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
'''
            
            result = subprocess.run([
                str(python_exe), '-c', test_script
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Package validation successful")
                return True
            else:
                logger.error(f"❌ Package validation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return False
    
    def _print_next_steps(self):
        """Print next steps for the user."""
        next_steps = f"""
🎉 SETUP COMPLETED SUCCESSFULLY!

📋 NEXT STEPS:

1. 🔑 Configure your API credentials:
   - Copy .env.example to .env
   - Add your Apify API token
   - Add your Printify API credentials
   - Add your Instagram profile URL
   - (Optional) Add Google Cloud credentials for enhanced CV

2. 🧪 Test the system:
   cd {self.project_root}
   source photography_env/bin/activate  # (Linux/Mac)
   # OR
   photography_env\\Scripts\\activate   # (Windows)
   
   python -m src.enhanced_main --help

3. 🚀 Run your first automation:
   python -m src.enhanced_main \\
     --profile "https://instagram.com/your_account" \\
     --categories sunset landscape \\
     --max-photos 5 \\
     --no-publish  # Test run without publishing

4. 📖 Documentation:
   - Check docs/ folder for detailed guides
   - See logs/ folder for execution logs
   - View data/ folder for processed results

🆘 NEED HELP?
   - Check the logs in logs/ folder
   - Ensure all API credentials are correctly set
   - Start with small batches (5-10 images) for testing

🎯 TIPS FOR SUCCESS:
   - Use high-quality source images (2MP+)
   - Test with --no-publish flag first
   - Monitor API rate limits
   - Start with popular photo categories (sunset, landscape)
"""
        
        print(next_steps)
    
    def create_launcher_script(self):
        """Create easy launcher scripts for different platforms."""
        
        # Linux/Mac launcher
        bash_script = f'''#!/bin/bash
# Photography Automation Launcher Script

cd "{self.project_root}"
source photography_env/bin/activate

echo "🎨 Instagram to Etsy Photography Automation"
echo "=========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please copy .env.example to .env and configure your credentials."
    exit 1
fi

# Parse command line arguments
PROFILE=""
CATEGORIES="sunset landscape mountains"
MAX_PHOTOS=10
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile|-p)
            PROFILE="$2"
            shift 2
            ;;
        --categories|-c)
            CATEGORIES="$2"
            shift 2
            ;;
        --max|-m)
            MAX_PHOTOS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Prompt for profile if not provided
if [ -z "$PROFILE" ]; then
    echo "📱 Enter your Instagram profile URL:"
    read PROFILE
fi

echo "🚀 Starting automation with:"
echo "   Profile: $PROFILE"
echo "   Categories: $CATEGORIES"
echo "   Max photos: $MAX_PHOTOS"
echo ""

# Run the automation
python -m src.enhanced_main \\
    --profile "$PROFILE" \\
    --categories $CATEGORIES \\
    --max-photos $MAX_PHOTOS \\
    $EXTRA_ARGS

echo "✅ Automation completed! Check data/ folder for results."
'''
        
        bash_path = self.project_root / 'run_automation.sh'
        with open(bash_path, 'w') as f:
            f.write(bash_script)
        
        # Make executable
        if self.system_os != 'windows':
            os.chmod(bash_path, 0o755)
        
        # Windows launcher
        batch_script = f'''@echo off
REM Photography Automation Launcher Script

cd /d "{self.project_root}"
call photography_env\\Scripts\\activate.bat

echo 🎨 Instagram to Etsy Photography Automation
echo ==========================================

REM Check if .env file exists
if not exist .env (
    echo ❌ .env file not found. Please copy .env.example to .env and configure your credentials.
    pause
    exit /b 1
)

REM Get profile URL from user if not provided
set PROFILE=%1
if "%PROFILE%"=="" (
    echo 📱 Enter your Instagram profile URL:
    set /p PROFILE=
)

REM Set defaults
if "%2"=="" (set CATEGORIES=sunset landscape mountains) else (set CATEGORIES=%2)
if "%3"=="" (set MAX_PHOTOS=10) else (set MAX_PHOTOS=%3)

echo 🚀 Starting automation with:
echo    Profile: %PROFILE%
echo    Categories: %CATEGORIES%
echo    Max photos: %MAX_PHOTOS%
echo.

REM Run the automation
python -m src.enhanced_main ^
    --profile "%PROFILE%" ^
    --categories %CATEGORIES% ^
    --max-photos %MAX_PHOTOS%

echo ✅ Automation completed! Check data\\ folder for results.
pause
'''
        
        batch_path = self.project_root / 'run_automation.bat'
        with open(batch_path, 'w') as f:
            f.write(batch_script)
        
        logger.info("🚀 Created launcher scripts:")
        logger.info(f"   - Linux/Mac: {bash_path}")
        logger.info(f"   - Windows: {batch_path}")
    
    def create_quick_start_guide(self):
        """Create a comprehensive quick start guide."""
        
        guide_content = '''# Instagram to Etsy Photography Automation - Quick Start Guide

## 🎯 Overview
This system automates the process of finding your best Instagram photos, optimizing them for print, and creating Etsy listings. Perfect for photographers looking to monetize their Instagram content.

## 🚀 Quick Start (5 Minutes)

### 1. Get Your API Credentials

#### Apify (Required - for Instagram scraping)
1. Go to [apify.com](https://apify.com)
2. Sign up for free account
3. Go to Settings > Integrations > API tokens
4. Copy your API token

#### Printify (Required - for print-on-demand)
1. Go to [printify.com](https://printify.com)
2. Create account and connect to Etsy
3. Go to My account > API
4. Generate API token
5. Note your Shop ID

#### Google Cloud Vision (Optional - for enhanced AI analysis)
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create new project
3. Enable Vision API
4. Create service account key
5. Download JSON credentials file

### 2. Configure Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use any text editor
```

### 3. Run Your First Test
```bash
# Activate environment
source photography_env/bin/activate  # Linux/Mac
# OR
photography_env\\Scripts\\activate   # Windows

# Test run (won't publish to Etsy)
python -m src.enhanced_main \\
  --profile "https://instagram.com/youraccount" \\
  --categories sunset landscape \\
  --max-photos 5 \\
  --no-publish
```

## 📊 Understanding Results

After running, check these directories:
- `data/analyzed_photos/` - AI analysis results
- `data/processed_prints/` - Print-ready images
- `data/metadata/` - Detailed analysis and SEO data
- `logs/` - Execution logs

## 🎨 Photo Categories

The system can identify and optimize for:
- **Landscapes**: Mountains, valleys, scenic vistas
- **Sunsets**: Golden hour, colorful skies
- **Water**: Oceans, lakes, waterfalls
- **Forests**: Trees, woodland scenes
- **Urban**: City, architecture, bridges
- **Weather**: Storms, fog, dramatic skies

## 💰 Pricing Strategy

The system suggests prices based on:
- Image quality score (AI-assessed)
- Print size and material costs
- Market research data
- Your specified multiplier (default 2.5x cost)

## 🏷️ SEO Optimization

Automatic Etsy SEO includes:
- High-converting keywords
- Room placement suggestions
- Material-specific descriptions
- Category-optimized tags

## ⚙️ Advanced Usage

### Custom Categories
```bash
python -m src.enhanced_main \\
  --categories "golden hour" "mountain peaks" "urban bridges" \\
  --quality-threshold 0.8
```

### Specific Print Options
```bash
python -m src.enhanced_main \\
  --print-sizes 16x20 24x36 \\
  --materials canvas metal \\
  --price-multiplier 3.0
```

### Batch Processing
```bash
python -m src.enhanced_main \\
  --max-photos 50 \\
  --quality-threshold 0.7 \\
  --output-dir "large_batch_$(date +%Y%m%d)"
```

## 🔧 Troubleshooting

### Common Issues

**"No images found"**
- Check Instagram profile URL format
- Ensure profile is public
- Try broader categories

**"API rate limit exceeded"**
- Wait 1 hour and retry
- Reduce max-photos parameter
- Check API quotas

**"Low quality scores"**
- Use higher resolution source images
- Try different photo categories
- Lower quality-threshold

**"Google Vision errors"**
- Check credentials file path
- Verify project billing enabled
- Use --no-enhanced-cv flag

### Performance Tips

1. **Start Small**: Begin with 5-10 images
2. **High Quality Sources**: Use 2MP+ images
3. **Popular Categories**: sunset, landscape perform best
4. **Test Mode**: Always use --no-publish for testing
5. **Monitor Logs**: Check logs/ for detailed execution info

## 📈 Optimizing Results

### Photo Selection
- Use images with clear subjects
- Prefer landscape orientation
- Ensure good lighting/contrast
- Avoid heavily filtered images

### Category Matching
- Be specific with categories
- Use multiple related categories
- Match categories to your style

### Pricing Optimization
- Start with default 2.5x multiplier
- Monitor Etsy analytics
- Adjust based on market response
- Consider seasonal pricing

## 🔄 Automation Workflow

1. **Discovery**: AI finds photos matching your categories
2. **Analysis**: Computer vision assesses quality and content
3. **Processing**: Images optimized for different print sizes/materials
4. **SEO Generation**: Titles, descriptions, tags created
5. **Product Creation**: Printify products generated
6. **Etsy Publishing**: Listings created with optimized metadata

## 📞 Support

- Check logs in `logs/` directory
- Review example configurations in `config/`
- Test with small batches first
- Ensure all API credentials are valid

## 🎯 Success Metrics

Track these metrics for optimization:
- Quality scores (aim for >0.75)
- Category match rates
- SEO scores (aim for >0.7)
- Processing success rates
- Etsy listing performance

---

Ready to start? Run the quick test above and check your results!
'''
        
        guide_path = self.project_root / 'docs' / 'QUICK_START.md'
        self.project_root.joinpath('docs').mkdir(exist_ok=True)
        
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        logger.info(f"📖 Created quick start guide: {guide_path}")

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Instagram to Etsy Photography Automation Deployer')
    
    parser.add_argument('--setup', action='store_true',
                        help='Set up the complete environment')
    parser.add_argument('--enhanced-cv', action='store_true',
                        help='Install enhanced computer vision packages')
    parser.add_argument('--create-launchers', action='store_true',
                        help='Create launcher scripts')
    parser.add_argument('--create-docs', action='store_true',
                        help='Create documentation')
    parser.add_argument('--full-setup', action='store_true',
                        help='Complete setup with all options')
    
    args = parser.parse_args()
    
    deployer = PhotographyAutomationDeployer()
    
    if args.full_setup:
        # Full setup with everything
        if deployer.setup_environment(enhanced_cv=True):
            deployer.create_launcher_script()
            deployer.create_quick_start_guide()
            logger.info("🎉 Full setup completed successfully!")
        else:
            logger.error("❌ Setup failed")
            sys.exit(1)
    
    elif args.setup:
        if not deployer.setup_environment(enhanced_cv=args.enhanced_cv):
            logger.error("❌ Setup failed")
            sys.exit(1)
    
    elif args.create_launchers:
        deployer.create_launcher_script()
    
    elif args.create_docs:
        deployer.create_quick_start_guide()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()