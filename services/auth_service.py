from datetime import datetime
from bson.objectid import ObjectId
from flask import session
import re

# MongoDB and Bcrypt instances
mongo = None
bcrypt = None

def init_auth_service(app, mongo_instance, bcrypt_instance):
    """
    Initialize the authentication service
    
    Args:
        app: Flask application
        mongo_instance: PyMongo instance
        bcrypt_instance: Bcrypt instance
    """
    global mongo, bcrypt
    mongo = mongo_instance
    bcrypt = bcrypt_instance


def register_user(username, email, password, confirm_password):
    """
    Register a new user
    
    Args:
        username: Username
        email: Email address
        password: Password
        confirm_password: Password confirmation
        
    Returns:
        (success, message, user_id) tuple
    """
    # Validate input
    if not username or not email or not password or not confirm_password:
        return False, "All fields are required", None
    
    # Validate username
    if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
        return False, "Username must be 3-20 characters and can only contain letters, numbers, and underscores", None
    
    # Validate email
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False, "Invalid email address", None
    
    # Validate password
    if len(password) < 8:
        return False, "Password must be at least 8 characters long", None
    
    # Validate password confirmation
    if password != confirm_password:
        return False, "Passwords do not match", None
    
    # Check if username already exists
    if mongo.db.users.find_one({'username': username}):
        return False, "Username already taken", None
    
    # Check if email already exists
    if mongo.db.users.find_one({'email': email}):
        return False, "Email already registered", None
    
    # Hash password
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    
    # Insert user into database
    new_user = {
        'username': username,
        'email': email.lower(),
        'password': hashed_password,
        'created_at': datetime.now(),
        'updated_at': datetime.now()
    }
    
    user_id = mongo.db.users.insert_one(new_user).inserted_id
    
    return True, "Registration successful. You can now log in.", str(user_id)


def login_user(email, password, remember=False):
    """
    Log in a user
    
    Args:
        email: Email address
        password: Password
        remember: Remember login
        
    Returns:
        (success, message, user_id) tuple
    """
    # Validate input
    if not email or not password:
        return False, "Email and password are required", None
    
    # Find user by email
    user = mongo.db.users.find_one({'email': email.lower()})
    
    # Check if user exists
    if not user:
        return False, "Invalid email or password", None
    
    # Check password
    if not bcrypt.check_password_hash(user['password'], password):
        return False, "Invalid email or password", None
    
    # Update last login timestamp
    mongo.db.users.update_one(
        {'_id': user['_id']},
        {'$set': {'last_login': datetime.now()}}
    )
    
    # Set session
    session.clear()
    session['user_id'] = str(user['_id'])
    session['username'] = user['username']
    session.permanent = remember
    
    return True, "Login successful", str(user['_id'])


def logout_user():
    """
    Log out a user
    
    Returns:
        (success, message) tuple
    """
    session.clear()
    return True, "Logout successful"


def get_current_user():
    """
    Get the current logged-in user
    
    Returns:
        User document or None
    """
    user_id = session.get('user_id')
    if not user_id:
        return None
    
    return mongo.db.users.find_one({'_id': ObjectId(user_id)})


def is_authenticated():
    """
    Check if user is authenticated
    
    Returns:
        Boolean
    """
    return 'user_id' in session


def change_password(current_password, new_password, confirm_password):
    """
    Change user password
    
    Args:
        current_password: Current password
        new_password: New password
        confirm_password: New password confirmation
        
    Returns:
        (success, message) tuple
    """
    user = get_current_user()
    if not user:
        return False, "You must be logged in to change your password"
    
    # Validate input
    if not current_password or not new_password or not confirm_password:
        return False, "All fields are required"
    
    # Check current password
    if not bcrypt.check_password_hash(user['password'], current_password):
        return False, "Current password is incorrect"
    
    # Validate new password
    if len(new_password) < 8:
        return False, "New password must be at least 8 characters long"
    
    # Validate password confirmation
    if new_password != confirm_password:
        return False, "New passwords do not match"
    
    # Hash new password
    hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
    
    # Update password in database
    mongo.db.users.update_one(
        {'_id': user['_id']},
        {'$set': {
            'password': hashed_password,
            'updated_at': datetime.now()
        }}
    )
    
    return True, "Password changed successfully"