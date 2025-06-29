#!/bin/bash
# Photography Automation Launcher Script

cd "/Users/gideoncrawley/Business Projects/InstaToEtsy"
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
python -m src.enhanced_main \
    --profile "$PROFILE" \
    --categories $CATEGORIES \
    --max-photos $MAX_PHOTOS \
    $EXTRA_ARGS

echo "✅ Automation completed! Check data/ folder for results."
