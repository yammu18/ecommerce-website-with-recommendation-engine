"""
Utils package for helper functions
"""
# Import helper functions
from utils.helpers import (
    generate_unique_filename, allowed_file, save_uploaded_file,
    format_date, format_currency, get_product_categories,
    parse_object_id, validate_email, truncate_text, paginate
)

# Import validation functions
from utils.validators import (
    validate_username, validate_password, validate_email,
    validate_phone, validate_address, validate_product_data,
    validate_cart_quantity, validate_checkout_data
)

# Import decorators
from utils.decorators import (
    login_required, admin_required, guest_only,
    rate_limit, cache_control, log_activity
)

# Initialize utility functions with the app
def init_app(app):
    """
    Register utility functions with the Flask application
    
    Args:
        app: Flask application
    """
    # Register template filters
    @app.template_filter('format_date')
    def _format_date(value, format='%Y-%m-%d'):
        return format_date(value, format)
    
    @app.template_filter('currency')
    def _format_currency(value):
        return format_currency(value)
    
    @app.template_filter('truncate')
    def _truncate_text(text, length=100):
        return truncate_text(text, length)
    
    # Make helpers available in templates
    @app.context_processor
    def utility_processor():
        return {
            'format_date': format_date,
            'format_currency': format_currency,
            'get_product_categories': get_product_categories,
            'truncate_text': truncate_text
        }