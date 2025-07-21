from bson.objectid import ObjectId
from datetime import datetime
import math
from database.db import serialize_doc, serialize_cursor

# MongoDB instance
mongo = None

def init_product_service(app, mongo_instance):
    """
    Initialize the product service
    
    Args:
        app: Flask application
        mongo_instance: PyMongo instance
    """
    global mongo
    mongo = mongo_instance


def get_products(page=1, per_page=12, category=None, search=None, sort_by='name', sort_dir='asc'):
    """
    Get paginated products with optional filtering and sorting
    
    Args:
        page: Page number
        per_page: Items per page
        category: Filter by category
        search: Search query
        sort_by: Sort field
        sort_dir: Sort direction ('asc' or 'desc')
        
    Returns:
        (products, total_pages, current_page) tuple
    """
    # Build query
    query = {}
    
    if category:
        query['category'] = category
    
    if search:
        query['$or'] = [
            {'name': {'$regex': search, '$options': 'i'}},
            {'description': {'$regex': search, '$options': 'i'}}
        ]
    
    # Get total count for pagination
    total_products = mongo.db.products.count_documents(query)
    total_pages = math.ceil(total_products / per_page)
    
    # Ensure valid page number
    page = max(1, min(page, total_pages)) if total_pages > 0 else 1
    
    # Apply sorting
    sort_direction = 1 if sort_dir == 'asc' else -1
    sort_criteria = [(sort_by, sort_direction)]
    
    # Get paginated products
    skip = (page - 1) * per_page
    products_cursor = mongo.db.products.find(query).sort(sort_criteria).skip(skip).limit(per_page)
    products = serialize_cursor(products_cursor)
    
    return products, total_pages, page


def get_product(product_id):
    """
    Get product by ID
    
    Args:
        product_id: Product ID
        
    Returns:
        Product document or None
    """
    try:
        product = mongo.db.products.find_one({'_id': ObjectId(product_id)})
        return serialize_doc(product)
    except:
        return None


def get_product_categories():
    """
    Get all product categories
    
    Returns:
        List of categories
    """
    categories = mongo.db.products.distinct('category')
    return sorted(categories)


def create_product(data):
    """
    Create a new product
    
    Args:
        data: Product data
        
    Returns:
        (success, message, product_id) tuple
    """
    # Validate required fields
    required_fields = ['name', 'description', 'price', 'category']
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"Field '{field}' is required", None
    
    # Validate price
    try:
        price = float(data['price'])
        if price <= 0:
            return False, "Price must be greater than 0", None
    except:
        return False, "Price must be a valid number", None
    
    # Prepare product data
    product = {
        'name': data['name'],
        'description': data['description'],
        'price': price,
        'category': data['category'],
        'subcategory': data.get('subcategory', ''),
        'stock_quantity': int(data.get('stock_quantity', 0)),
        'features': data.get('features', []),
        'ratings_average': 0,
        'images': data.get('images', []),
        'created_at': datetime.now(),
        'updated_at': datetime.now()
    }
    
    # Insert product into database
    product_id = mongo.db.products.insert_one(product).inserted_id
    
    return True, "Product created successfully", str(product_id)


def update_product(product_id, data):
    """
    Update a product
    
    Args:
        product_id: Product ID
        data: Updated product data
        
    Returns:
        (success, message) tuple
    """
    try:
        # Check if product exists
        product = mongo.db.products.find_one({'_id': ObjectId(product_id)})
        if not product:
            return False, "Product not found"
        
        # Validate price if provided
        if 'price' in data:
            try:
                price = float(data['price'])
                if price <= 0:
                    return False, "Price must be greater than 0"
                data['price'] = price
            except:
                return False, "Price must be a valid number"
        
        # Validate stock_quantity if provided
        if 'stock_quantity' in data:
            try:
                data['stock_quantity'] = int(data['stock_quantity'])
            except:
                return False, "Stock quantity must be a valid integer"
        
        # Update timestamp
        data['updated_at'] = datetime.now()
        
        # Update product in database
        mongo.db.products.update_one(
            {'_id': ObjectId(product_id)},
            {'$set': data}
        )
        
        return True, "Product updated successfully"
    except Exception as e:
        return False, f"Error updating product: {str(e)}"


def delete_product(product_id):
    """
    Delete a product
    
    Args:
        product_id: Product ID
        
    Returns:
        (success, message) tuple
    """
    try:
        # Check if product exists
        product = mongo.db.products.find_one({'_id': ObjectId(product_id)})
        if not product:
            return False, "Product not found"
        
        # Check if product is in any active orders
        order_count = mongo.db.orders.count_documents({
            'products.product_id': ObjectId(product_id),
            'status': {'$nin': ['cancelled', 'returned']}
        })
        
        if order_count > 0:
            return False, "Cannot delete product with active orders"
        
        # Delete product from database
        mongo.db.products.delete_one({'_id': ObjectId(product_id)})
        
        # Delete related interactions
        mongo.db.interactions.delete_many({'product_id': ObjectId(product_id)})
        
        return True, "Product deleted successfully"
    except Exception as e:
        return False, f"Error deleting product: {str(e)}"


def record_product_view(user_id, product_id):
    """
    Record a product view interaction
    
    Args:
        user_id: User ID
        product_id: Product ID
    """
    if not user_id:
        return
    
    try:
        # Create interaction
        interaction = {
            'user_id': ObjectId(user_id),
            'product_id': ObjectId(product_id),
            'type': 'view',
            'timestamp': datetime.now()
        }
        
        # Insert interaction into database
        mongo.db.interactions.insert_one(interaction)
    except:
        # Silently fail to not disrupt user experience
        pass


def get_related_products(product_id, limit=4):
    """
    Get related products based on category
    
    Args:
        product_id: Product ID
        limit: Number of related products to return
        
    Returns:
        List of related products
    """
    try:
        # Get product
        product = mongo.db.products.find_one({'_id': ObjectId(product_id)})
        if not product:
            return []
        
        # Get products from the same category
        related_cursor = mongo.db.products.find({
            '_id': {'$ne': ObjectId(product_id)},
            'category': product['category']
        }).limit(limit)
        
        return serialize_cursor(related_cursor)
    except:
        return []
# Add this debugging function to services/product_service.py

def debug_product_list():
    """
    Get all products for debugging
    
    Returns:
        List of all products with id fields for verification
    """
    try:
        products_cursor = mongo.db.products.find({})
        products = []
        for product in products_cursor:
            if product:
                product_dict = {
                    '_id': str(product['_id']),
                    'name': product.get('name', 'No name'),
                    'price': product.get('price', 0),
                    'category': product.get('category', 'No category')
                }
                products.append(product_dict)
        
        return products
    except Exception as e:
        print(f"Debug error: {str(e)}")
        return []

# Add this at the end of your get_products function to debug any issues
def get_products(page=1, per_page=12, category=None, search=None, sort_by='name', sort_dir='asc'):
    """
    Get paginated products with optional filtering and sorting
    
    Args:
        page: Page number
        per_page: Items per page
        category: Filter by category
        search: Search query
        sort_by: Sort field
        sort_dir: Sort direction ('asc' or 'desc')
        
    Returns:
        (products, total_pages, current_page) tuple
    """
    # Build query
    query = {}
    
    if category:
        query['category'] = category
    
    if search:
        query['$or'] = [
            {'name': {'$regex': search, '$options': 'i'}},
            {'description': {'$regex': search, '$options': 'i'}}
        ]
    
    # Get total count for pagination
    total_products = mongo.db.products.count_documents(query)
    total_pages = math.ceil(total_products / per_page)
    
    # Ensure valid page number
    page = max(1, min(page, total_pages)) if total_pages > 0 else 1
    
    # Apply sorting
    sort_direction = 1 if sort_dir == 'asc' else -1
    sort_criteria = [(sort_by, sort_direction)]
    
    # Get paginated products
    skip = (page - 1) * per_page
    products_cursor = mongo.db.products.find(query).sort(sort_criteria).skip(skip).limit(per_page)
    products = serialize_cursor(products_cursor)
    
    # Add this debug code AFTER products is defined
    if not products:
        print("No products found with query:", query)
        print("Total products in database:", mongo.db.products.count_documents({}))
    
    return products, total_pages, page