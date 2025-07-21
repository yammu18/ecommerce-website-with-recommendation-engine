from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, g
from services.cart_service import get_cart, add_to_cart, update_cart_item, remove_from_cart, clear_cart
from controllers.auth_controller import login_required
from recommender.engine import get_recommendations
import werkzeug.routing
# Create blueprint
cart_bp = Blueprint('cart', __name__, url_prefix='/cart')

@cart_bp.route('/')
def index():
    """Display shopping cart"""
    # Get cart
    cart = get_cart()
    
    # Cart items are now in cart_items key instead of items
    cart_items = cart['cart_items']  # Updated key name
    
    # Get personalized recommendations for cart items
    recommendations = []
    if g.user and cart_items:
        user_id = str(g.user['_id'])
        # Get product IDs from cart
        product_ids = [item.get('id') for item in cart_items if item.get('id')]
        # Get recommendations based on cart items
        if product_ids:
            recommendations = get_recommendations(user_id, seed_product_ids=product_ids, limit=4)
    
    return render_template('cart/view.html', cart=cart, cart_items=cart_items, recommendations=recommendations)
@cart_bp.route('/add/<product_id>', methods=['POST'])
def add(product_id):
    """Add a product to the cart"""
    # Get quantity from form
    quantity = request.form.get('quantity', 1)
    
    # Add to cart
    success, message = add_to_cart(product_id, quantity)
    
    # Flash message
    flash(message, 'success' if success else 'danger')
    
    # Redirect back
    return redirect(request.referrer or url_for('product.index'))

@cart_bp.route('/update/<product_id>', methods=['POST'])
def update(product_id):
    """Update a cart item"""
    # Get quantity from form
    quantity = request.form.get('quantity', 1)
    
    # Update cart
    success, message = update_cart_item(product_id, quantity)
    
    # Handle AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': success, 'message': message})
    
    # Flash message for non-AJAX request
    flash(message, 'success' if success else 'danger')
    
    # Redirect back
    return redirect(request.referrer or url_for('cart.index'))

@cart_bp.route('/remove/<product_id>', methods=['POST'])
def remove(product_id):
    """Remove a product from the cart"""
    # Remove from cart
    success, message = remove_from_cart(product_id)
    
    # Flash message
    flash(message, 'success' if success else 'danger')
    
    # Redirect back
    return redirect(request.referrer or url_for('cart.index'))

@cart_bp.route('/clear', methods=['POST'])
def clear():
    """Clear the cart"""
    # Clear cart
    success, message = clear_cart()
    
    # Flash message
    flash(message, 'success' if success else 'danger')
    
    # Redirect back
    return redirect(url_for('cart.index'))

# Fix for cart_controller.py's checkout function

@cart_bp.route('/checkout', methods=['GET', 'POST'])
@login_required
def checkout():
    """Checkout page"""
    # Get cart
    cart_data = get_cart()
    
    # Use cart_items key consistently 
    cart_items = cart_data['cart_items']
    
    # Check if cart is empty
    if not cart_items:
        flash('Your cart is empty', 'warning')
        return redirect(url_for('product.index'))
    
    # Handle checkout form submission
    if request.method == 'POST':
        # Get form data
        shipping_address = {
            'full_name': request.form.get('full_name'),
            'address_line1': request.form.get('address_line1'),
            'address_line2': request.form.get('address_line2', ''),
            'city': request.form.get('city'),
            'state': request.form.get('state'),
            'postal_code': request.form.get('postal_code'),
            'country': request.form.get('country'),
            'phone': request.form.get('phone')
        }
        
        payment_method = request.form.get('payment_method', 'cash_on_delivery')
        
        # Process checkout
        from services.cart_service import checkout_cart
        success, message, order_id = checkout_cart(str(g.user['_id']), shipping_address, payment_method)
        
        if success:
            # Display success message and redirect to order list
            flash(f'Order placed successfully! Order ID: {order_id}', 'success')
            return redirect(url_for('order.index'))
        else:
            # Flash error message
            flash(message, 'danger')
    
    return render_template('cart/checkout.html', cart=cart_data, cart_items=cart_items)