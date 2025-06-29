"""
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
