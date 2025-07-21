from bson.objectid import ObjectId
from datetime import datetime
from database.db import serialize_doc, serialize_cursor

# MongoDB instance
mongo = None

def init_order_service(app, mongo_instance):
    """
    Initialize the order service
    
    Args:
        app: Flask application
        mongo_instance: PyMongo instance
    """
    global mongo
    mongo = mongo_instance


def get_user_orders(user_id, page=1, per_page=10):
    """
    Get orders for a specific user with pagination
    
    Args:
        user_id: User ID
        page: Page number
        per_page: Items per page
        
    Returns:
        (orders, total_pages, current_page) tuple
    """
    try:
        # Build query
        query = {'user_id': ObjectId(user_id)}
        
        # Get total count for pagination
        total_orders = mongo.db.orders.count_documents(query)
        total_pages = (total_orders + per_page - 1) // per_page  # Ceiling division
        
        # Ensure valid page number
        page = max(1, min(page, total_pages)) if total_pages > 0 else 1
        
        # Get paginated orders
        skip = (page - 1) * per_page
        orders_cursor = mongo.db.orders.find(query).sort('order_date', -1).skip(skip).limit(per_page)
        orders = serialize_cursor(orders_cursor)
        
        return orders, total_pages, page
    except Exception as e:
        print(f"Error fetching user orders: {e}")
        return [], 0, 1


def get_order(order_id, user_id=None):
    """
    Get order by ID
    
    Args:
        order_id: Order ID
        user_id: Optional user ID for security check
        
    Returns:
        Order document or None
    """
    try:
        # Build query
        query = {'_id': ObjectId(order_id)}
        if user_id:
            query['user_id'] = ObjectId(user_id)
        
        # Get order
        order = mongo.db.orders.find_one(query)
        return serialize_doc(order)
    except:
        return None


def get_order_status_counts(user_id):
    """
    Get counts of orders by status for a user
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary of status counts
    """
    try:
        pipeline = [
            {'$match': {'user_id': ObjectId(user_id)}},
            {'$group': {'_id': '$status', 'count': {'$sum': 1}}},
            {'$sort': {'_id': 1}}
        ]
        
        result = mongo.db.orders.aggregate(pipeline)
        status_counts = {item['_id']: item['count'] for item in result}
        
        # Ensure all statuses have a count
        all_statuses = ['pending', 'processing', 'shipped', 'delivered', 'cancelled']
        for status in all_statuses:
            if status not in status_counts:
                status_counts[status] = 0
        
        return status_counts
    except Exception as e:
        print(f"Error getting order status counts: {e}")
        return {}


def update_order_status(order_id, status, user_id=None):
    """
    Update order status
    
    Args:
        order_id: Order ID
        status: New status
        user_id: Optional user ID for security check
        
    Returns:
        (success, message) tuple
    """
    try:
        # Validate status
        valid_statuses = ['pending', 'processing', 'shipped', 'delivered', 'cancelled']
        if status not in valid_statuses:
            return False, f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        
        # Build query
        query = {'_id': ObjectId(order_id)}
        if user_id:
            query['user_id'] = ObjectId(user_id)
        
        # Get order
        order = mongo.db.orders.find_one(query)
        if not order:
            return False, "Order not found"
        
        # Update status
        mongo.db.orders.update_one(
            {'_id': ObjectId(order_id)},
            {'$set': {'status': status, 'updated_at': datetime.now()}}
        )
        
        return True, f"Order status updated to {status}"
    except Exception as e:
        return False, f"Error updating order status: {str(e)}"


def get_order_history(user_id, limit=5):
    """
    Get recent order history for a user
    
    Args:
        user_id: User ID
        limit: Maximum number of orders to return
        
    Returns:
        List of recent orders
    """
    try:
        # Get recent orders
        orders_cursor = mongo.db.orders.find(
            {'user_id': ObjectId(user_id)}
        ).sort('order_date', -1).limit(limit)
        
        return serialize_cursor(orders_cursor)
    except:
        return []


def get_order_statistics(user_id):
    """
    Get order statistics for a user
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary of order statistics
    """
    try:
        # Count total orders
        total_orders = mongo.db.orders.count_documents({'user_id': ObjectId(user_id)})
        
        # Calculate total spent
        pipeline = [
            {'$match': {'user_id': ObjectId(user_id)}},
            {'$group': {'_id': None, 'total': {'$sum': '$total_price'}}}
        ]
        result = list(mongo.db.orders.aggregate(pipeline))
        total_spent = result[0]['total'] if result else 0
        
        # Calculate average order value
        average_order_value = total_spent / total_orders if total_orders > 0 else 0
        
        # Get most recent order
        recent_order = mongo.db.orders.find_one(
            {'user_id': ObjectId(user_id)},
            sort=[('order_date', -1)]
        )
        
        # Get most purchased products
        pipeline = [
            {'$match': {'user_id': ObjectId(user_id)}},
            {'$unwind': '$products'},
            {'$group': {
                '_id': '$products.product_id',
                'name': {'$first': '$products.name'},
                'count': {'$sum': '$products.quantity'}
            }},
            {'$sort': {'count': -1}},
            {'$limit': 3}
        ]
        most_purchased = list(mongo.db.orders.aggregate(pipeline))
        
        return {
            'total_orders': total_orders,
            'total_spent': total_spent,
            'average_order_value': average_order_value,
            'recent_order': serialize_doc(recent_order) if recent_order else None,
            'most_purchased': [
                {'id': str(item['_id']), 'name': item['name'], 'count': item['count']}
                for item in most_purchased
            ]
        }
    except Exception as e:
        print(f"Error getting order statistics: {e}")
        return {
            'total_orders': 0,
            'total_spent': 0,
            'average_order_value': 0,
            'recent_order': None,
            'most_purchased': []
        }