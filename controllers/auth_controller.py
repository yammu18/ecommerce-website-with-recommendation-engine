from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from functools import wraps
from services.auth_service import (
    register_user, login_user, logout_user, is_authenticated,
    change_password, get_current_user
)

# Create blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

def login_required(f):
    """
    Decorator to require login for a route
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def guest_only(f):
    """
    Decorator to restrict a route to guests only
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if is_authenticated():
            flash('You are already logged in', 'info')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/register', methods=['GET', 'POST'])
@guest_only
def register():
    """Handle user registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        success, message, user_id = register_user(username, email, password, confirm_password)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('auth.login'))
        else:
            flash(message, 'danger')
    
    return render_template('auth/register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
@guest_only
def login():
    """Handle user login"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'
        
        success, message, user_id = login_user(email, password, remember)
        
        if success:
            flash(message, 'success')
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('home'))
        else:
            flash(message, 'danger')
    
    return render_template('auth/login.html')

@auth_bp.route('/logout')
def logout():
    """Handle user logout"""
    success, message = logout_user()
    flash(message, 'info')
    return redirect(url_for('home'))

@auth_bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password_route():
    """Handle password change"""
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        success, message = change_password(current_password, new_password, confirm_password)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('user.profile'))
        else:
            flash(message, 'danger')
    
    return render_template('auth/change_password.html')

@auth_bp.route('/profile')
@login_required
def profile():
    """Display user profile"""
    user = get_current_user()
    return render_template('auth/profile.html', user=user)