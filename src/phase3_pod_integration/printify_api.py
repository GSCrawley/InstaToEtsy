import os
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from urllib.parse import urljoin

from .. import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Printify API Base URL
PRINTIFY_API_BASE = "https://api.printify.com/v1/"

class PrintifyAPI:
    """
    Class for interacting with the Printify API to create and publish products
    to print-on-demand services and Etsy.
    """
    
    def __init__(self, api_token: str = None, shop_id: str = None):
        """
        Initialize the Printify API client.
        
        Args:
            api_token: Printify API token. If None, loaded from config/environment.
            shop_id: Printify shop ID. If None, loaded from config/environment.
        """
        self.api_token = api_token or config.PRINTIFY_API_TOKEN
        self.shop_id = shop_id or config.PRINTIFY_SHOP_ID
        
        if not self.api_token:
            logger.error("Printify API token not provided. Cannot connect to Printify API.")
        else:
            logger.info("Printify API client initialized.")
            
        if not self.shop_id:
            logger.warning("Printify shop ID not provided. You'll need to specify shop_id for operations.")
            
        # Setup session for API requests
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Cache for blueprints and print providers
        self._blueprints_cache = None
        self._print_providers_cache = {}
        
    def _make_request(self, method: str, endpoint: str, 
                     params: Dict[str, Any] = None, 
                     data: Dict[str, Any] = None, 
                     files: Dict[str, Any] = None,
                     retry_count: int = 3, 
                     retry_delay: float = 1.0) -> Dict[str, Any]:
        """
        Make a request to the Printify API with retry logic.
        
        Args:
            method: HTTP method ('GET', 'POST', 'PUT', etc.)
            endpoint: API endpoint (without the base URL)
            params: URL parameters
            data: Request body data
            files: Files to upload
            retry_count: Number of retries on failure
            retry_delay: Delay between retries (exponential backoff applied)
            
        Returns:
            Response data as dictionary
        """
        url = urljoin(PRINTIFY_API_BASE, endpoint)
        current_retry = 0
        
        while current_retry <= retry_count:
            try:
                if files:
                    # For file uploads, don't send JSON
                    headers = self.session.headers.copy()
                    headers.pop('Content-Type', None)
                    response = self.session.request(
                        method=method,
                        url=url,
                        params=params,
                        data=data,
                        files=files,
                        headers=headers,
                        timeout=30
                    )
                else:
                    # Standard JSON request
                    json_data = json.dumps(data) if data else None
                    response = self.session.request(
                        method=method,
                        url=url,
                        params=params,
                        data=json_data,
                        timeout=30
                    )
                
                # Handle rate limits
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited by Printify API. Waiting {retry_after} seconds.")
                    time.sleep(retry_after)
                    current_retry += 1
                    continue
                    
                # Check for success
                response.raise_for_status()
                
                # Return JSON response if available
                if response.content:
                    return response.json()
                return {}
                
            except requests.exceptions.RequestException as e:
                # Handle retryable errors
                if current_retry < retry_count:
                    # Exponential backoff
                    sleep_time = retry_delay * (2 ** current_retry)
                    logger.warning(f"Request to {url} failed: {e}. Retrying in {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
                    current_retry += 1
                else:
                    logger.error(f"Request to {url} failed after {retry_count} retries: {e}")
                    raise
                    
        # This should not be reached, but just in case
        raise RuntimeError(f"Failed to make request to {url} after {retry_count} retries")
        
    def get_shops(self) -> List[Dict[str, Any]]:
        """
        Get list of shops connected to the Printify account.
        
        Returns:
            List of shop dictionaries
        """
        logger.info("Getting list of shops from Printify")
        response = self._make_request('GET', 'shops.json')
        shops = response.get('data', [])
        logger.info(f"Found {len(shops)} shops")
        return shops
        
    def get_shop_info(self, shop_id: str = None) -> Dict[str, Any]:
        """
        Get information about a specific shop.
        
        Args:
            shop_id: Shop ID. If None, uses the default shop_id.
            
        Returns:
            Shop information dictionary
        """
        shop_id = shop_id or self.shop_id
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        logger.info(f"Getting information for shop {shop_id}")
        return self._make_request('GET', f'shops/{shop_id}.json')
        
    def get_blueprints(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of available product blueprints (product types).
        
        Args:
            force_refresh: Whether to force a refresh of cached blueprints
            
        Returns:
            List of blueprint dictionaries
        """
        if self._blueprints_cache is None or force_refresh:
            logger.info("Getting product blueprints from Printify")
            response = self._make_request('GET', 'catalog/blueprints.json')
            self._blueprints_cache = response.get('data', [])
            logger.info(f"Found {len(self._blueprints_cache)} product blueprints")
            
        return self._blueprints_cache
        
    def get_blueprint_details(self, blueprint_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific product blueprint.
        
        Args:
            blueprint_id: Blueprint ID
            
        Returns:
            Blueprint details dictionary
        """
        logger.info(f"Getting details for blueprint {blueprint_id}")
        return self._make_request('GET', f'catalog/blueprints/{blueprint_id}.json')
        
    def get_print_providers(self, blueprint_id: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of print providers for a specific blueprint.
        
        Args:
            blueprint_id: Blueprint ID
            force_refresh: Whether to force a refresh of cached print providers
            
        Returns:
            List of print provider dictionaries
        """
        cache_key = str(blueprint_id)
        
        if cache_key not in self._print_providers_cache or force_refresh:
            logger.info(f"Getting print providers for blueprint {blueprint_id}")
            response = self._make_request('GET', f'catalog/blueprints/{blueprint_id}/print_providers.json')
            self._print_providers_cache[cache_key] = response.get('data', [])
            logger.info(f"Found {len(self._print_providers_cache[cache_key])} print providers for blueprint {blueprint_id}")
            
        return self._print_providers_cache[cache_key]
        
    def get_variants(self, blueprint_id: int, print_provider_id: int) -> List[Dict[str, Any]]:
        """
        Get list of variants for a specific blueprint and print provider.
        
        Args:
            blueprint_id: Blueprint ID
            print_provider_id: Print provider ID
            
        Returns:
            List of variant dictionaries
        """
        logger.info(f"Getting variants for blueprint {blueprint_id} and provider {print_provider_id}")
        endpoint = f'catalog/blueprints/{blueprint_id}/print_providers/{print_provider_id}/variants.json'
        response = self._make_request('GET', endpoint)
        variants = response.get('data', [])
        logger.info(f"Found {len(variants)} variants")
        return variants
        
    def get_shipping_info(self, blueprint_id: int, print_provider_id: int) -> Dict[str, Any]:
        """
        Get shipping information for a specific blueprint and print provider.
        
        Args:
            blueprint_id: Blueprint ID
            print_provider_id: Print provider ID
            
        Returns:
            Shipping information dictionary
        """
        logger.info(f"Getting shipping info for blueprint {blueprint_id} and provider {print_provider_id}")
        endpoint = f'catalog/blueprints/{blueprint_id}/print_providers/{print_provider_id}/shipping.json'
        return self._make_request('GET', endpoint)
        
    def upload_image(self, image_path: str, file_name: str = None) -> Dict[str, Any]:
        """
        Upload an image to Printify.
        
        Args:
            image_path: Path to the image file
            file_name: Name to use for the uploaded file. If None, uses the basename of image_path.
            
        Returns:
            Response containing the image information including ID
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        if file_name is None:
            file_name = os.path.basename(image_path)
            
        logger.info(f"Uploading image {image_path} to Printify")
        
        with open(image_path, 'rb') as f:
            files = {
                'file': (file_name, f, 'image/jpeg')
            }
            
            shop_id = self.shop_id
            if not shop_id:
                raise ValueError("Shop ID is required for uploading images")
                
            endpoint = f'shops/{shop_id}/images.json'
            response = self._make_request('POST', endpoint, files=files)
            
            if 'id' in response:
                logger.info(f"Image uploaded successfully. Image ID: {response['id']}")
            else:
                logger.error(f"Failed to upload image. Response: {response}")
                
            return response
            
    def create_product(self, shop_id: str = None, product_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new product on Printify.
        
        Args:
            shop_id: Shop ID. If None, uses the default shop_id.
            product_data: Product data dictionary including:
                - title: Product title
                - description: Product description
                - blueprint_id: Blueprint ID
                - print_provider_id: Print provider ID
                - variants: List of variant dictionaries
                - print_areas: Dictionary of print areas with image IDs
                
        Returns:
            Response containing the created product information
        """
        shop_id = shop_id or self.shop_id
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        if not product_data:
            raise ValueError("Product data is required")
            
        required_fields = ['title', 'description', 'blueprint_id', 'print_provider_id', 'variants', 'print_areas']
        for field in required_fields:
            if field not in product_data:
                raise ValueError(f"Missing required field in product data: {field}")
                
        logger.info(f"Creating product '{product_data['title']}' in shop {shop_id}")
        endpoint = f'shops/{shop_id}/products.json'
        response = self._make_request('POST', endpoint, data=product_data)
        
        if 'id' in response:
            logger.info(f"Product created successfully. Product ID: {response['id']}")
        else:
            logger.error(f"Failed to create product. Response: {response}")
            
        return response
        
    def update_product(self, shop_id: str, product_id: str, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing product on Printify.
        
        Args:
            shop_id: Shop ID
            product_id: Product ID
            product_data: Updated product data dictionary
            
        Returns:
            Response containing the updated product information
        """
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        if not product_id:
            raise ValueError("Product ID is required")
            
        logger.info(f"Updating product {product_id} in shop {shop_id}")
        endpoint = f'shops/{shop_id}/products/{product_id}.json'
        response = self._make_request('PUT', endpoint, data=product_data)
        
        if 'id' in response:
            logger.info(f"Product updated successfully. Product ID: {response['id']}")
        else:
            logger.error(f"Failed to update product. Response: {response}")
            
        return response
        
    def publish_product(self, shop_id: str, product_id: str, publish: bool = True) -> Dict[str, Any]:
        """
        Publish or unpublish a product to external marketplaces (e.g., Etsy).
        
        Args:
            shop_id: Shop ID
            product_id: Product ID
            publish: Whether to publish (True) or unpublish (False) the product
            
        Returns:
            Response containing the publish operation result
        """
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        if not product_id:
            raise ValueError("Product ID is required")
            
        action = "Publishing" if publish else "Unpublishing"
        logger.info(f"{action} product {product_id} in shop {shop_id}")
        
        endpoint = f'shops/{shop_id}/products/{product_id}/publish.json'
        data = {"publish": publish}
        response = self._make_request('POST', endpoint, data=data)
        
        status = "published" if publish else "unpublished"
        if response.get('status') == status:
            logger.info(f"Product {status} successfully")
        else:
            logger.error(f"Failed to {action.lower()} product. Response: {response}")
            
        return response
        
    def get_product(self, shop_id: str, product_id: str) -> Dict[str, Any]:
        """
        Get information about a specific product.
        
        Args:
            shop_id: Shop ID
            product_id: Product ID
            
        Returns:
            Product information dictionary
        """
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        if not product_id:
            raise ValueError("Product ID is required")
            
        logger.info(f"Getting information for product {product_id} in shop {shop_id}")
        endpoint = f'shops/{shop_id}/products/{product_id}.json'
        return self._make_request('GET', endpoint)
        
    def get_products(self, shop_id: str = None, page: int = 1, limit: int = 10) -> Dict[str, Any]:
        """
        Get list of products in a shop.
        
        Args:
            shop_id: Shop ID. If None, uses the default shop_id.
            page: Page number for pagination
            limit: Number of products per page
            
        Returns:
            Response containing the list of products
        """
        shop_id = shop_id or self.shop_id
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        logger.info(f"Getting products for shop {shop_id} (page {page}, limit {limit})")
        endpoint = f'shops/{shop_id}/products.json'
        params = {
            'page': page,
            'limit': limit
        }
        return self._make_request('GET', endpoint, params=params)
        
    def delete_product(self, shop_id: str, product_id: str) -> Dict[str, Any]:
        """
        Delete a product from Printify.
        
        Args:
            shop_id: Shop ID
            product_id: Product ID
            
        Returns:
            Response indicating success or failure
        """
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        if not product_id:
            raise ValueError("Product ID is required")
            
        logger.info(f"Deleting product {product_id} from shop {shop_id}")
        endpoint = f'shops/{shop_id}/products/{product_id}.json'
        return self._make_request('DELETE', endpoint)
        
    def create_order(self, shop_id: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new order on Printify.
        
        Args:
            shop_id: Shop ID
            order_data: Order data dictionary
            
        Returns:
            Response containing the created order information
        """
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        if not order_data:
            raise ValueError("Order data is required")
            
        logger.info(f"Creating order in shop {shop_id}")
        endpoint = f'shops/{shop_id}/orders.json'
        return self._make_request('POST', endpoint, data=order_data)
        
    def get_order(self, shop_id: str, order_id: str) -> Dict[str, Any]:
        """
        Get information about a specific order.
        
        Args:
            shop_id: Shop ID
            order_id: Order ID
            
        Returns:
            Order information dictionary
        """
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        if not order_id:
            raise ValueError("Order ID is required")
            
        logger.info(f"Getting information for order {order_id} in shop {shop_id}")
        endpoint = f'shops/{shop_id}/orders/{order_id}.json'
        return self._make_request('GET', endpoint)
        
    def get_orders(self, shop_id: str = None, page: int = 1, limit: int = 10) -> Dict[str, Any]:
        """
        Get list of orders in a shop.
        
        Args:
            shop_id: Shop ID. If None, uses the default shop_id.
            page: Page number for pagination
            limit: Number of orders per page
            
        Returns:
            Response containing the list of orders
        """
        shop_id = shop_id or self.shop_id
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        logger.info(f"Getting orders for shop {shop_id} (page {page}, limit {limit})")
        endpoint = f'shops/{shop_id}/orders.json'
        params = {
            'page': page,
            'limit': limit
        }
        return self._make_request('GET', endpoint, params=params)
        
    def cancel_order(self, shop_id: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order on Printify.
        
        Args:
            shop_id: Shop ID
            order_id: Order ID
            
        Returns:
            Response indicating success or failure
        """
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        if not order_id:
            raise ValueError("Order ID is required")
            
        logger.info(f"Cancelling order {order_id} in shop {shop_id}")
        endpoint = f'shops/{shop_id}/orders/{order_id}/cancel.json'
        return self._make_request('POST', endpoint)
        
    def calculate_shipping(self, shop_id: str, shipping_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate shipping costs for an order.
        
        Args:
            shop_id: Shop ID
            shipping_data: Shipping calculation data including address and items
            
        Returns:
            Response containing shipping cost information
        """
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        if not shipping_data:
            raise ValueError("Shipping data is required")
            
        logger.info(f"Calculating shipping costs for shop {shop_id}")
        endpoint = f'shops/{shop_id}/orders/shipping.json'
        return self._make_request('POST', endpoint, data=shipping_data)

    def find_wall_art_blueprints(self) -> List[Dict[str, Any]]:
        """
        Find all blueprints related to wall art (posters, canvas, framed prints, etc.).
        
        Returns:
            List of wall art blueprint dictionaries
        """
        all_blueprints = self.get_blueprints()
        wall_art_blueprints = []
        
        # Keywords that indicate wall art products
        wall_art_keywords = [
            'poster', 'canvas', 'print', 'frame', 'wall', 'art', 'photo', 'picture',
            'artwork', 'painting', 'metal print', 'acrylic print'
        ]
        
        for blueprint in all_blueprints:
            title = blueprint.get('title', '').lower()
            if any(keyword in title for keyword in wall_art_keywords):
                wall_art_blueprints.append(blueprint)
                
        logger.info(f"Found {len(wall_art_blueprints)} wall art blueprints out of {len(all_blueprints)} total")
        return wall_art_blueprints
        
    def prepare_product_from_image(self, 
                                 image_path: str, 
                                 title: str, 
                                 description: str,
                                 blueprint_id: int,
                                 print_provider_id: int,
                                 variant_ids: List[int] = None,
                                 tags: List[str] = None,
                                 price_multiplier: float = 2.0) -> Dict[str, Any]:
        """
        Prepare product data for creating a product from an image.
        
        Args:
            image_path: Path to the image file
            title: Product title
            description: Product description
            blueprint_id: Blueprint ID
            print_provider_id: Print provider ID
            variant_ids: List of variant IDs to include. If None, all available variants are used.
            tags: List of tags for the product
            price_multiplier: Multiplier for setting the retail price based on the print cost
            
        Returns:
            Product data dictionary ready for create_product()
        """
        # Upload the image
        image_response = self.upload_image(image_path)
        if 'id' not in image_response:
            raise ValueError(f"Failed to upload image: {image_response}")
            
        image_id = image_response['id']
        
        # Get available variants
        all_variants = self.get_variants(blueprint_id, print_provider_id)
        if not all_variants:
            raise ValueError(f"No variants available for blueprint {blueprint_id} and provider {print_provider_id}")
            
        # Filter variants if specific IDs provided
        variants_to_use = all_variants
        if variant_ids:
            variants_to_use = [v for v in all_variants if v['id'] in variant_ids]
            if not variants_to_use:
                raise ValueError(f"None of the specified variant IDs were found")
                
        # Prepare variant data
        variants = []
        for variant in variants_to_use:
            # Calculate price (cost * multiplier)
            cost = float(variant['cost']) / 100  # Convert from cents to dollars
            price = round(cost * price_multiplier, 2)
            price_cents = int(price * 100)  # Convert back to cents
            
            variant_data = {
                'id': variant['id'],
                'price': price_cents,
                'is_enabled': True
            }
            variants.append(variant_data)
            
        # Get blueprint details to determine print areas
        blueprint_details = self.get_blueprint_details(blueprint_id)
        print_areas = {}
        
        # For simplicity, use the same image for all print areas
        for print_area in blueprint_details.get('print_areas', []):
            print_areas[print_area['id']] = {
                'placement': 'center',
                'images': [
                    {
                        'id': image_id,
                        'x': 0.5,
                        'y': 0.5,
                        'scale': 1.0,
                        'angle': 0
                    }
                ]
            }
            
        # Prepare product data
        product_data = {
            'title': title,
            'description': description,
            'blueprint_id': blueprint_id,
            'print_provider_id': print_provider_id,
            'variants': variants,
            'print_areas': print_areas
        }
        
        # Add tags if provided
        if tags:
            product_data['tags'] = tags
            
        return product_data
        
    def create_and_publish_product(self,
                                 image_path: str,
                                 title: str,
                                 description: str,
                                 blueprint_id: int,
                                 print_provider_id: int,
                                 variant_ids: List[int] = None,
                                 tags: List[str] = None,
                                 price_multiplier: float = 2.0,
                                 shop_id: str = None,
                                 publish: bool = True) -> Dict[str, Any]:
        """
        Create and optionally publish a product from an image in one operation.
        
        Args:
            image_path: Path to the image file
            title: Product title
            description: Product description
            blueprint_id: Blueprint ID
            print_provider_id: Print provider ID
            variant_ids: List of variant IDs to include
            tags: List of tags for the product
            price_multiplier: Multiplier for setting the retail price
            shop_id: Shop ID. If None, uses the default shop_id.
            publish: Whether to publish the product after creation
            
        Returns:
            Dictionary with created product information and publish status
        """
        shop_id = shop_id or self.shop_id
        if not shop_id:
            raise ValueError("Shop ID is required")
            
        # Prepare product data
        product_data = self.prepare_product_from_image(
            image_path=image_path,
            title=title,
            description=description,
            blueprint_id=blueprint_id,
            print_provider_id=print_provider_id,
            variant_ids=variant_ids,
            tags=tags,
            price_multiplier=price_multiplier
        )
        
        # Create the product
        create_response = self.create_product(shop_id=shop_id, product_data=product_data)
        if 'id' not in create_response:
            return {
                'success': False,
                'error': 'Failed to create product',
                'response': create_response
            }
            
        product_id = create_response['id']
        result = {
            'success': True,
            'product': create_response,
            'published': False
        }
        
        # Publish if requested
        if publish:
            publish_response = self.publish_product(shop_id=shop_id, product_id=product_id)
            result['publish_response'] = publish_response
            result['published'] = publish_response.get('status') == 'published'
            
        return result

# Example usage
if __name__ == "__main__":
    import sys
    import os
    from pprint import pprint
    
    # Determine project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Add project root to sys.path
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        
    # Load .env file
    dotenv_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(dotenv_path):
        from dotenv import load_dotenv
        print(f"Loading .env file from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
        
    # Create Printify API client
    printify = PrintifyAPI()
    
    # Get connected shops
    shops = printify.get_shops()
    print("\nConnected shops:")
    for shop in shops:
        print(f"  - {shop['title']} (ID: {shop['id']}, Platform: {shop['shop_type']})")
        
    # Find wall art blueprints
    wall_art_blueprints = printify.find_wall_art_blueprints()
    print("\nWall art blueprints:")
    for blueprint in wall_art_blueprints[:5]:  # Show first 5
        print(f"  - {blueprint['title']} (ID: {blueprint['id']})")
        
    if wall_art_blueprints:
        # Get details for the first wall art blueprint
        blueprint_id = wall_art_blueprints[0]['id']
        print(f"\nGetting details for blueprint ID {blueprint_id}...")
        blueprint_details = printify.get_blueprint_details(blueprint_id)
        print(f"Title: {blueprint_details['title']}")
        print(f"Description: {blueprint_details['description']}")
        
        # Get print providers for this blueprint
        print(f"\nGetting print providers for blueprint ID {blueprint_id}...")
        providers = printify.get_print_providers(blueprint_id)
        for provider in providers:
            print(f"  - {provider['title']} (ID: {provider['id']})")
