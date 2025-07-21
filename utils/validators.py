"""
Input validation functions for the e-commerce application
"""
import re

def validate_username(username):
    """
    Validate username format
    
    Args:
        username: Username to validate
        
    Returns:
        (is_valid, message) tuple
    """
    if not username:
        return False, "Username is required"
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    
    if len(username) > 20:
        return False, "Username must be at most 20 characters long"
    
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores"
    
    return True, "Username is valid"

def validate_password(password):
    """
    Validate password strength
    
    Args:
        password: Password to validate
        
    Returns:
        (is_valid, message) tuple
    """
    if not password:
        return False, "Password is required"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    # Check for more complex requirements if needed
    has_uppercase = any(c.isupper() for c in password)
    has_lowercase = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(not c.isalnum() for c in password)
    
    if not (has_uppercase and has_lowercase and has_digit):
        return False, "Password must contain at least one uppercase letter, one lowercase letter, and one digit"
    
    return True, "Password is valid"

def validate_email(email):
    """
    Validate email format
    
    Args:
        email: Email to validate
        
    Returns:
        (is_valid, message) tuple
    """
    if not email:
        return False, "Email is required"
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, "Invalid email address"
    
    return True, "Email is valid"

def validate_phone(phone):
    """
    Validate phone number format
    
    Args:
        phone: Phone number to validate
        
    Returns:
        (is_valid, message) tuple
    """
    if not phone:
        return True, "Phone number is optional"
    
    # Strip all non-digit characters
    phone_digits = re.sub(r'\D', '', phone)
    
    if len(phone_digits) < 10:
        return False, "Phone number must have at least 10 digits"
    
    if len(phone_digits) > 15:
        return False, "Phone number is too long"
    
    return True, "Phone number is valid"

def validate_address(address):
    """
    Validate address fields
    
    Args:
        address: Dictionary with address fields
        
    Returns:
        (is_valid, message) tuple
    """
    required_fields = ['street', 'city', 'state', 'postal_code', 'country']
    
    # Check required fields
    for field in required_fields:
        if field not in address or not address[field]:
            return False, f"{field.replace('_', ' ').title()} is required"
    
    # Validate postal code
    postal_code = address['postal_code']
    if not postal_code.isalnum():
        return False, "Postal code should only contain letters and numbers"
    
    return True, "Address is valid"

def validate_product_data(data):
    """
    Validate product data
    
    Args:
        data: Product data dictionary
        
    Returns:
        (is_valid, message) tuple
    """
    # Check required fields
    required_fields = ['name', 'description', 'price', 'category']
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"{field.replace('_', ' ').title()} is required"
    
    # Validate price
    try:
        price = float(data['price'])
        if price <= 0:
            return False, "Price must be greater than 0"
    except (ValueError, TypeError):
        return False, "Price must be a valid number"
    
    # Validate stock quantity if provided
    if 'stock_quantity' in data:
        try:
            stock_quantity = int(data['stock_quantity'])
            if stock_quantity < 0:
                return False, "Stock quantity cannot be negative"
        except (ValueError, TypeError):
            return False, "Stock quantity must be a valid number"
    
    return True, "Product data is valid"

def validate_cart_quantity(quantity, stock_quantity):
    """
    Validate cart item quantity
    
    Args:
        quantity: Quantity to validate
        stock_quantity: Available stock quantity
        
    Returns:
        (is_valid, message) tuple
    """
    try:
        quantity = int(quantity)
    except (ValueError, TypeError):
        return False, "Quantity must be a valid number"
    
    if quantity <= 0:
        return False, "Quantity must be greater than 0"
    
    if quantity > stock_quantity:
        return False, f"Only {stock_quantity} items available in stock"
    
    return True, "Quantity is valid"

def validate_checkout_data(data):
    """
    Validate checkout form data
    
    Args:
        data: Checkout form data
        
    Returns:
        (is_valid, message) tuple
    """
    # Check personal information
    if not data.get('full_name'):
        return False, "Full name is required"
    
    if not data.get('phone'):
        return False, "Phone number is required"
    
    # Check shipping address
    address_fields = {
        'address_line1': 'Address line 1',
        'city': 'City',
        'state': 'State/Province',
        'postal_code': 'Postal code',
        'country': 'Country'
    }
    
    for field, label in address_fields.items():
        if not data.get(field):
            return False, f"{label} is required"
    
    # Check payment method
    if not data.get('payment_method'):
        return False, "Payment method is required"
    
    return True, "Checkout data is valid"