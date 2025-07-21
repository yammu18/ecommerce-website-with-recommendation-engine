from flask import Blueprint, render_template, request, redirect, url_for, flash, session, g
from bson.objectid import ObjectId
from services.order_service import (
    get_user_orders, get_order, get_order_status_counts,
    update_order_status, get_order_history, get_order_statistics
)
from services.product_service import get_product  # Import the get_product function
from controllers.auth_controller import login_required
from recommender.engine import get_recommendations

# Create blueprint
order_bp = Blueprint('order', __name__, url_prefix='/orders')

@order_bp.route('/')
@login_required
def index():
    """Display user orders"""
    # Get query parameters for pagination
    page = int(request.args.get('page', 1))
    
    # Get user orders
    user_id = str(g.user['_id'])
    orders, total_pages, current_page = get_user_orders(user_id, page=page)
    
    # Get order status counts
    status_counts = get_order_status_counts(user_id)
    
    return render_template(
        'orders/list.html',
        orders=orders,
        status_counts=status_counts,
        total_pages=total_pages,
        current_page=current_page,
        get_product=get_product  # Pass the get_product function to the template
    )

@order_bp.route('/<order_id>')
@login_required
def detail(order_id):
    """Display order detail"""
    # Get user ID
    user_id = str(g.user['_id'])
    
    # Get order
    order = get_order(order_id, user_id)
    if not order:
        flash('Order not found', 'danger')
        return redirect(url_for('order.index'))
    
    # Get recommendations based on this order
    product_ids = [product['product_id'] for product in order.get('products', [])]
    recommendations = get_recommendations(user_id, seed_product_ids=product_ids, limit=4)
    
    return render_template(
        'orders/detail.html',
        order=order,
        recommendations=recommendations,
        get_product=get_product  # Also pass it to detail template if needed
    )

@order_bp.route('/<order_id>/cancel', methods=['POST'])
@login_required
def cancel(order_id):
    """Cancel an order"""
    # Get user ID
    user_id = str(g.user['_id'])
    
    # Update order status
    success, message = update_order_status(order_id, 'cancelled', user_id)
    
    # Flash message
    if success:
        flash(message, 'success')
    else:
        flash(message, 'danger')
    
    return redirect(url_for('order.detail', order_id=order_id))

@order_bp.route('/history')
@login_required
def history():
    """Display purchase history and statistics"""
    # Get user ID
    user_id = str(g.user['_id'])
    
    # Get order history and statistics
    orders = get_order_history(user_id, limit=10)
    statistics = get_order_statistics(user_id)
    
    # Get recommendations based on purchase history
    recommendations = get_recommendations(user_id, limit=4)
    
    return render_template(
        'orders/history.html',
        orders=orders,
        statistics=statistics,
        recommendations=recommendations,
        get_product=get_product  # Also pass it to history template if needed
    )

@order_bp.route('/track/<order_id>')
@login_required
def track(order_id):
    """Track order status"""
    # Get user ID
    user_id = str(g.user['_id'])
    
    # Get order
    order = get_order(order_id, user_id)
    if not order:
        flash('Order not found', 'danger')
        return redirect(url_for('order.index'))
    
    return render_template('orders/track.html', order=order, get_product=get_product)