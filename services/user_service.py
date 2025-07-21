from bson.objectid import ObjectId
from datetime import datetime
import re
from database.db import serialize_doc

# MongoDB and Bcrypt instances
mongo = None
bcrypt = None

def init_user_service(app, mongo_instance):
    """
    Initialize the user service
    
    Args:
        app: Flask application
        mongo_instance: PyMongo instance
    """
    global mongo
    mongo = mongo_instance
    
    # Get Bcrypt instance from app context
    from services.auth_service import bcrypt as auth_bcrypt
    global bcrypt
    bcrypt = auth_bcrypt


def get_user(user_id):
    """
    Get user by ID
    
    Args:
        user_id: User ID
        
    Returns:
        User document or None
    """
    try:
        user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        return serialize_doc(user)
    except:
        return None


def update_user_profile(user_id, data):
    """
    Update user profile
    
    Args:
        user_id: User ID
        data: Updated user data
        
    Returns:
        (success, message) tuple
    """
    try:
        # Get user
        user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if not user:
            return False, "User not found"
        
        # Validate email if provided
        if 'email' in data and data['email'] != user['email']:
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', data['email']):
                return False, "Invalid email address"
            
            # Check if email already exists
            if mongo.db.users.find_one({'email': data['email'], '_id': {'$ne': ObjectId(user_id)}}):
                return False, "Email already taken"
            
            data['email'] = data['email'].lower()
        
        # Validate username if provided
        if 'username' in data and data['username'] != user['username']:
            if not re.match(r'^[a-zA-Z0-9_]{3,20}$', data['username']):
                return False, "Username must be 3-20 characters and can only contain letters, numbers, and underscores"
            
            # Check if username already exists
            if mongo.db.users.find_one({'username': data['username'], '_id': {'$ne': ObjectId(user_id)}}):
                return False, "Username already taken"
        
        # Update timestamp
        data['updated_at'] = datetime.now()
        
        # Update user in database
        mongo.db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': data}
        )
        
        return True, "Profile updated successfully"
    except Exception as e:
        return False, f"Error updating profile: {str(e)}"


def get_user_activity(user_id, limit=10):
    """
    Get recent user activity
    
    Args:
        user_id: User ID
        limit: Maximum number of activities to return
        
    Returns:
        List of recent activities
    """
    try:
        # Get recent interactions
        interactions_cursor = mongo.db.interactions.find(
            {'user_id': ObjectId(user_id)}
        ).sort('timestamp', -1).limit(limit)
        
        # Format activities
        activities = []
        for interaction in interactions_cursor:
            # Get product details
            product = mongo.db.products.find_one({'_id': interaction['product_id']})
            if not product:
                continue
                
            # Format activity
            activity = {
                'type': interaction['type'],
                'timestamp': interaction['timestamp'],
                'product': {
                    'id': str(product['_id']),
                    'name': product['name'],
                    'price': product['price'],
                    'image': product.get('images', [])[0] if product.get('images') else None
                }
            }
            
            activities.append(activity)
        
        return activities
    except Exception as e:
        print(f"Error getting user activity: {e}")
        return []


def get_user_stats(user_id):
    """
    Get user statistics
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary of user statistics
    """
    try:
        # Count total orders
        total_orders = mongo.db.orders.count_documents({'user_id': ObjectId(user_id)})
        
        # Count total products viewed
        total_views = mongo.db.interactions.count_documents({
            'user_id': ObjectId(user_id),
            'type': 'view'
        })
        
        # Count unique products viewed
        unique_views = len(mongo.db.interactions.distinct('product_id', {
            'user_id': ObjectId(user_id),
            'type': 'view'
        }))
        
        # Calculate total spent
        pipeline = [
            {'$match': {'user_id': ObjectId(user_id)}},
            {'$group': {'_id': None, 'total': {'$sum': '$total_price'}}}
        ]
        result = list(mongo.db.orders.aggregate(pipeline))
        total_spent = result[0]['total'] if result else 0
        
        return {
            'total_orders': total_orders,
            'total_views': total_views,
            'unique_views': unique_views,
            'total_spent': total_spent
        }
    except Exception as e:
        print(f"Error getting user stats: {e}")
        return {
            'total_orders': 0,
            'total_views': 0,
            'unique_views': 0,
            'total_spent': 0
        }