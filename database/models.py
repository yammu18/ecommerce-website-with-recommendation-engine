"""
Database models (MongoDB schemas) for the e-commerce application
"""
from datetime import datetime
from bson.objectid import ObjectId

# User model schema
user_schema = {
    "_id": ObjectId,
    "username": str,
    "email": str,
    "password": str,  # Hashed password
    "roles": list,  # ['user', 'admin', etc.]
    "full_name": str,
    "phone": str,
    "address": {
        "street": str,
        "city": str,
        "state": str,
        "postal_code": str,
        "country": str
    },
    "profile_picture": str,  # Path to image
    "created_at": datetime,
    "updated_at": datetime,
    "last_login": datetime
}

# Product model schema
product_schema = {
    "_id": ObjectId,
    "name": str,
    "description": str,
    "price": float,
    "category": str,
    "subcategory": str,
    "stock_quantity": int,
    "features": list,  # List of feature strings
    "ratings_average": float,
    "images": list,  # List of image paths
    "created_at": datetime,
    "updated_at": datetime
}

# Interaction model schema
interaction_schema = {
    "_id": ObjectId,
    "user_id": ObjectId,
    "product_id": ObjectId,
    "type": str,  # 'view', 'purchase', 'add_to_cart', 'rating'
    "rating": float,  # Only for 'rating' type
    "timestamp": datetime
}

# Order model schema
order_schema = {
    "_id": ObjectId,
    "user_id": ObjectId,
    "products": [
        {
            "product_id": ObjectId,
            "name": str,
            "price": float,
            "quantity": int,
            "subtotal": float
        }
    ],
    "total_price": float,
    "order_date": datetime,
    "status": str,  # 'pending', 'processing', 'shipped', 'delivered', 'cancelled'
    "shipping_address": {
        "full_name": str,
        "address_line1": str,
        "address_line2": str,
        "city": str,
        "state": str,
        "postal_code": str,
        "country": str,
        "phone": str
    },
    "payment_method": str,  # 'credit_card', 'paypal', 'cash_on_delivery'
    "notes": str
}

# Recommendation model schema
recommendation_schema = {
    "_id": ObjectId,
    "user_id": ObjectId,
    "recommended_products": [
        {
            "product_id": ObjectId,
            "score": float
        }
    ],
    "algorithm_used": str,  # 'ncf', 'deepfm', 'vae', 'hybrid'
    "timestamp": datetime
}

# Activity log model schema
activity_log_schema = {
    "_id": ObjectId,
    "user_id": ObjectId,
    "type": str,  # Type of activity
    "route": str,  # URL route
    "method": str,  # HTTP method
    "timestamp": float,  # Unix timestamp
    "ip_address": str,
    "user_agent": str
}

# Create MongoDB indexes function
def create_indexes(db):
    """Create MongoDB indexes for better performance"""
    # Users collection
    db.users.create_index([('email', 1)], unique=True)
    db.users.create_index([('username', 1)], unique=True)
    
    # Products collection
    db.products.create_index([('name', 1)])
    db.products.create_index([('category', 1)])
    db.products.create_index([('price', 1)])
    
    # Interactions collection
    db.interactions.create_index([('user_id', 1), ('product_id', 1)])
    db.interactions.create_index([('timestamp', -1)])
    
    # Orders collection
    db.orders.create_index([('user_id', 1)])
    db.orders.create_index([('order_date', -1)])
    
    # Recommendations collection
    db.recommendations.create_index([('user_id', 1)])
    db.recommendations.create_index([('timestamp', -1)])