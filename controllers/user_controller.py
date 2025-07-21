from flask import Blueprint, render_template, request, redirect, url_for, flash, g
from services.user_service import (
    get_user, update_user_profile, get_user_activity, get_user_stats
)
from services.order_service import get_order_history
from controllers.auth_controller import login_required
from recommender.engine import get_recommendations
import os
from werkzeug.utils import secure_filename

# Create blueprint
user_bp = Blueprint('user', __name__, url_prefix='/user')

@user_bp.route('/profile')
@login_required
def profile():
    """Display user profile"""
    # Get user ID
    user_id = str(g.user['_id'])
    
    # Get user stats and activity
    stats = get_user_stats(user_id)
    activity = get_user_activity(user_id)
    
    # Get recent orders
    recent_orders = get_order_history(user_id, limit=3)
    
    # Get personalized recommendations
    recommendations = get_recommendations(user_id, limit=4)
    
    return render_template(
        'user/profile.html',
        user=g.user,
        stats=stats,
        activity=activity,
        recent_orders=recent_orders,
        recommendations=recommendations
    )

@user_bp.route('/edit-profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Edit user profile"""
    if request.method == 'POST':
        # Get form data
        data = {
            'username': request.form.get('username'),
            'email': request.form.get('email'),
            'full_name': request.form.get('full_name'),
            'phone': request.form.get('phone'),
            'address': {
                'street': request.form.get('street'),
                'city': request.form.get('city'),
                'state': request.form.get('state'),
                'postal_code': request.form.get('postal_code'),
                'country': request.form.get('country')
            }
        }
        
        # Handle profile picture upload
        if 'profile_picture' in request.files:
            profile_picture = request.files['profile_picture']
            if profile_picture.filename:
                # Save profile picture
                filename = secure_filename(f"{g.user['_id']}_{profile_picture.filename}")
                upload_dir = os.path.join('static', 'uploads', 'profile_pictures')
                os.makedirs(upload_dir, exist_ok=True)
                filepath = os.path.join(upload_dir, filename)
                profile_picture.save(filepath)
                data['profile_picture'] = filepath
        
        # Update profile
        user_id = str(g.user['_id'])
        success, message = update_user_profile(user_id, data)
        
        # Flash message
        if success:
            flash(message, 'success')
            return redirect(url_for('user.profile'))
        else:
            flash(message, 'danger')
    
    return render_template('user/edit_profile.html', user=g.user)

# Update the history function in user_controller.py

# Complete fix for the history function in user_controller.py

@user_bp.route('/purchase-history')
@login_required
def history():
    """Display user's purchase history"""
    user_id = str(g.user['_id'])
    
    try:
        # Get the page parameter from the request
        page = request.args.get('page', 1, type=int)
        
        # Get user's orders using the correct service method that includes pagination
        from services.order_service import get_user_orders, get_order_statistics
        
        # Get orders with pagination
        orders, total_pages, current_page = get_user_orders(user_id, page=page)
        
        # Get order statistics
        statistics = get_order_statistics(user_id)
        
        # Debug output
        print(f"User {user_id} has {len(orders)} orders on page {current_page} of {total_pages}")
        print(f"Statistics: {statistics}")
        
        return render_template(
            'user/purchase_history.html', 
            orders=orders, 
            statistics=statistics,
            current_page=current_page,
            total_pages=total_pages
        )
    
    except Exception as e:
        # Log the error and show an error message
        print(f"Error getting purchase history: {str(e)}")
        flash("There was an error retrieving your purchase history. Please try again later.", "danger")
        return redirect(url_for('user.profile'))
@user_bp.route('/activity')
@login_required
def activity():
    """Display user activity"""
    # Get user ID
    user_id = str(g.user['_id'])
    
    # Get user activity
    activity = get_user_activity(user_id, limit=20)
    
    return render_template('user/activity.html', activity=activity)