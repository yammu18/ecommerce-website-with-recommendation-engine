"""
Database package for MongoDB operations
"""
# Import database functions
from database.db import init_db, get_db, mongo, serialize_doc, serialize_cursor, object_id

# Initialize database with the app
def init_app(app, mongo_instance):
    """
    Initialize the database with the Flask application
    
    Args:
        app: Flask application
        mongo_instance: PyMongo instance
    """
    init_db(app, mongo_instance)