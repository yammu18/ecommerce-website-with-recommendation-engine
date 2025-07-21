from flask_pymongo import PyMongo
from bson import ObjectId
import json
import datetime

# MongoDB instance
mongo = None
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
def init_db(app, mongo_instance):
    """
    Initialize the database connection
    
    Args:
        app: Flask application
        mongo_instance: PyMongo instance
    """
    global mongo
    mongo = mongo_instance
    
    # Instead of before_first_request which is removed in newer Flask versions
    # Create indexes during app initialization
    with app.app_context():
        create_indexes(mongo.db)
    
    # Or you can use this alternative to run once on startup
    @app.before_request
    def create_indexes_on_first_request():
        # Use a flag to ensure it only runs once
        if not hasattr(app, '_db_initialized'):
            with app.app_context():
                create_indexes(mongo.db)
            app._db_initialized = True
# MongoDB JSON Encoder to handle ObjectId and datetime
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super(MongoJSONEncoder, self).default(obj)


def get_db():
    """
    Get the MongoDB database instance
    
    Returns:
        MongoDB database instance
    """
    return mongo.db


def object_id(id_str):
    """
    Convert string ID to MongoDB ObjectId
    
    Args:
        id_str: ID as string
        
    Returns:
        ObjectId instance
    """
    try:
        return ObjectId(id_str)
    except:
        return None


# Updated serialize_doc function in database/db.py

def serialize_doc(doc):
    """
    Serialize MongoDB document to JSON
    
    Args:
        doc: MongoDB document
        
    Returns:
        JSON serializable dictionary
    """
    if doc is None:
        return None
    
    result = {}
    for key, value in doc.items():
        if key == '_id':
            # Keep the original _id AND add an id field
            result['_id'] = str(value)  # Keep _id for template compatibility
            result['id'] = str(value)   # Also add id for other parts of the system
        elif isinstance(value, ObjectId):
            result[key] = str(value)
        elif isinstance(value, datetime.datetime):
            result[key] = value.isoformat()
        elif isinstance(value, list) and all(isinstance(x, ObjectId) for x in value):
            result[key] = [str(x) for x in value]
        else:
            result[key] = value
    return result
# Add this function to your database/db.py file

def serialize_cursor(cursor):
    """
    Serialize MongoDB cursor to JSON
    
    Args:
        cursor: MongoDB cursor
        
    Returns:
        List of JSON serializable dictionaries
    """
    return [serialize_doc(doc) for doc in cursor]