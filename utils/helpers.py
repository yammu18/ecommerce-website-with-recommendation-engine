"""
Helper utility functions for the e-commerce application
"""
import os
import uuid
import re
from datetime import datetime
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId

def generate_unique_filename(filename):
    """
    Generate a unique filename by adding a UUID
    
    Args:
        filename: Original filename
        
    Returns:
        Unique secure filename
    """
    ext = os.path.splitext(filename)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    return secure_filename(unique_name)

def allowed_file(filename, allowed_extensions=None):
    """
    Check if a file has an allowed extension
    
    Args:
        filename: Filename to check
        allowed_extensions: Set of allowed extensions (e.g., {'jpg', 'png'})
        
    Returns:
        Boolean indicating if file is allowed
    """
    if allowed_extensions is None:
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, upload_folder, allowed_extensions=None):
    """
    Save an uploaded file with a unique name
    
    Args:
        file: File object from request.files
        upload_folder: Folder to save the file
        allowed_extensions: Set of allowed extensions
        
    Returns:
        Path to the saved file or None if error
    """
    if file and allowed_file(file.filename, allowed_extensions):
        filename = generate_unique_filename(file.filename)
        
        # Create upload folder if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)
        
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        # Return the relative path from static folder
        return os.path.join(os.path.basename(upload_folder), filename)
    
    return None

def format_date(date_obj, format_string='%Y-%m-%d'):
    """
    Format a date object to string
    
    Args:
        date_obj: Date object
        format_string: Format string for strftime
        
    Returns:
        Formatted date string
    """
    if not date_obj:
        return ""
    
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.fromisoformat(date_obj)
        except ValueError:
            return date_obj
    
    return date_obj.strftime(format_string)

def format_currency(amount):
    """
    Format a number as currency
    
    Args:
        amount: Amount to format
        
    Returns:
        Formatted currency string
    """
    try:
        return f"${float(amount):.2f}"
    except (ValueError, TypeError):
        return "$0.00"

def get_product_categories():
    """
    Get list of product categories (to be called from templates)
    
    Returns:
        List of category names
    """
    from database.db import mongo
    
    try:
        return mongo.db.products.distinct('category')
    except Exception as e:
        print(f"Error getting product categories: {e}")
        return []

def parse_object_id(id_string):
    """
    Safely parse a string to ObjectId
    
    Args:
        id_string: String to parse
        
    Returns:
        ObjectId or None if invalid
    """
    try:
        return ObjectId(id_string)
    except:
        return None

def validate_email(email):
    """
    Validate email format
    
    Args:
        email: Email to validate
        
    Returns:
        Boolean indicating if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def truncate_text(text, max_length=100):
    """
    Truncate text to a maximum length with ellipsis
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + '...'

def paginate(items, page=1, per_page=10):
    """
    Paginate a list of items
    
    Args:
        items: List of items
        page: Current page number
        per_page: Items per page
        
    Returns:
        (paginated_items, total_pages, current_page) tuple
    """
    total_items = len(items)
    total_pages = (total_items + per_page - 1) // per_page
    
    # Ensure valid page number
    page = max(1, min(page, total_pages)) if total_pages > 0 else 1
    
    # Get items for current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_items = items[start_idx:end_idx]
    
    return paginated_items, total_pages, page