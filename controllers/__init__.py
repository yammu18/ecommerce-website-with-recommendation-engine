"""
Controllers package for route handling
"""
# Import all blueprint objects
from controllers.auth_controller import auth_bp
from controllers.product_controller import product_bp
from controllers.user_controller import user_bp
from controllers.cart_controller import cart_bp
from controllers.order_controller import order_bp

# Register all blueprints with the app
def init_app(app):
    """
    Register all blueprints with the Flask application
    
    Args:
        app: Flask application
    """
    app.register_blueprint(auth_bp)
    app.register_blueprint(product_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(cart_bp)
    app.register_blueprint(order_bp)