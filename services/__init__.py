"""
Services package for business logic
"""
# Import all service initialization functions
from services.auth_service import init_auth_service
from services.product_service import init_product_service
from services.user_service import init_user_service
from services.cart_service import init_cart_service
from services.order_service import init_order_service

# Initialize all services with the app
def init_app(app, mongo, bcrypt):
    """
    Initialize all services with the Flask application
    
    Args:
        app: Flask application
        mongo: PyMongo instance
        bcrypt: Bcrypt instance
    """
    init_auth_service(app, mongo, bcrypt)
    init_product_service(app, mongo)
    init_user_service(app, mongo)
    init_cart_service(app, mongo)
    init_order_service(app, mongo)