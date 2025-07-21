"""
Custom decorators for the e-commerce application
"""
from functools import wraps
from flask import request, redirect, url_for, flash, g, abort, current_app
import time

def login_required(f):
    """
    Decorator to require login for a route
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if g.user is None:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """
    Decorator to require admin role for a route
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if g.user is None:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('auth.login', next=request.url))
        
        if 'admin' not in g.user.get('roles', []):
            flash('You do not have permission to access this page', 'danger')
            return abort(403)
        
        return f(*args, **kwargs)
    return decorated_function

def guest_only(f):
    """
    Decorator to restrict a route to guests only (not logged in users)
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if g.user is not None:
            flash('You are already logged in', 'info')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

def rate_limit(requests_per_minute=60):
    """
    Decorator to rate limit a route
    """
    def decorator(f):
        # Store last request timestamps per IP
        last_requests = {}
        
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get client IP
            ip = request.remote_addr
            
            # Current time
            now = time.time()
            
            # Initialize rate limit data for this IP if needed
            if ip not in last_requests:
                last_requests[ip] = []
            
            # Clean up old requests
            last_requests[ip] = [t for t in last_requests[ip] if now - t < 60]
            
            # Check rate limit
            if len(last_requests[ip]) >= requests_per_minute:
                return abort(429)  # Too Many Requests
            
            # Add current request
            last_requests[ip].append(now)
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator

def cache_control(max_age=0, private=True, no_cache=False, no_store=False):
    """
    Decorator to set cache control headers
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = f(*args, **kwargs)
            
            cache_control_directives = []
            
            if max_age > 0:
                cache_control_directives.append(f"max-age={max_age}")
            
            if private:
                cache_control_directives.append("private")
            else:
                cache_control_directives.append("public")
            
            if no_cache:
                cache_control_directives.append("no-cache")
            
            if no_store:
                cache_control_directives.append("no-store")
            
            cache_control_header = ", ".join(cache_control_directives)
            response.headers["Cache-Control"] = cache_control_header
            
            return response
        
        return decorated_function
    
    return decorator

def log_activity(activity_type):
    """
    Decorator to log user activity
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Process the request
            response = f(*args, **kwargs)
            
            # If there's a logged-in user
            if g.user is not None:
                from database.db import mongo
                
                user_id = g.user['_id']
                route = request.path
                method = request.method
                timestamp = time.time()
                ip_address = request.remote_addr
                user_agent = request.user_agent.string
                
                # Log activity
                activity_log = {
                    'user_id': user_id,
                    'type': activity_type,
                    'route': route,
                    'method': method,
                    'timestamp': timestamp,
                    'ip_address': ip_address,
                    'user_agent': user_agent
                }
                
                try:
                    mongo.db.activity_logs.insert_one(activity_log)
                except Exception as e:
                    current_app.logger.error(f"Error logging activity: {e}")
            
            return response
        
        return decorated_function
    
    return decorator