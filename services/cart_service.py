from bson.objectid import ObjectId
from datetime import datetime
from flask import session
from database.db import serialize_doc
import json

# MongoDB instance
mongo = None

def init_cart_service(app, mongo_instance):
    """
    Initialize the cart service
    
    Args:
        app: Flask application
        mongo_instance: PyMongo instance
    """
    global mongo
    mongo = mongo_instance


# Modification to get_cart() function in cart_service.py

# Updated get_cart() function in cart_service.py

def get_cart():
    """
    Get the current cart
    
    Returns:
        Cart dict with items and total
    """
    # Initialize cart if not exists
    if 'cart' not in session:
        session['cart'] = []
    
    cart_items = session['cart']
    products = []
    total = 0
    
    # Get product details for each cart item
    for item in cart_items:
        product_id = item.get('product_id')
        quantity = item.get('quantity', 1)
        
        try:
            product = mongo.db.products.find_one({'_id': ObjectId(product_id)})
            if product:
                product_dict = serialize_doc(product)
                product_dict['quantity'] = quantity
                product_dict['subtotal'] = product['price'] * quantity
                
                products.append(product_dict)
                total += product_dict['subtotal']
        except Exception as e:
            print(f"Error getting product {product_id}: {str(e)}")
    
    # Ensure products is a list
    products_list = list(products) if products else []
    
    # For debugging
    print(f"Cart has {len(products_list)} items with total {total}")
    
    # Return dict with BOTH 'items' and 'cart_items' keys for backwards compatibility
    # This way, both code using cart['items'] and cart['cart_items'] will work
    return {
        'items': products_list,      # Keep this key for backwards compatibility
        'cart_items': products_list,  # Use this key to avoid collision with dict.items()
        'total': total,
        'count': len(products_list)
    }
def add_to_cart(product_id, quantity=1):
    """
    Add a product to the cart
    
    Args:
        product_id: Product ID
        quantity: Quantity to add
        
    Returns:
        (success, message) tuple
    """
    try:
        # Validate product
        product = mongo.db.products.find_one({'_id': ObjectId(product_id)})
        if not product:
            return False, "Product not found"
        
        # Validate quantity
        try:
            quantity = int(quantity)
            if quantity <= 0:
                return False, "Quantity must be greater than 0"
        except:
            return False, "Invalid quantity"
        
        # Check stock
        if product.get('stock_quantity', 0) < quantity:
            return False, "Not enough stock available"
        
        # Initialize cart if not exists
        if 'cart' not in session:
            session['cart'] = []
        
        # Check if product is already in cart
        found = False
        for item in session['cart']:
            if item.get('product_id') == product_id:
                # Update quantity
                item['quantity'] = item.get('quantity', 1) + quantity
                found = True
                break
        
        # Add new item if not found
        if not found:
            session['cart'].append({
                'product_id': product_id,
                'quantity': quantity
            })
        
        # Save cart to session
        session.modified = True
        
        # Record interaction if user is logged in
        if 'user_id' in session:
            interaction = {
                'user_id': ObjectId(session['user_id']),
                'product_id': ObjectId(product_id),
                'type': 'add_to_cart',
                'timestamp': datetime.now()
            }
            mongo.db.interactions.insert_one(interaction)
        
        return True, f"Added {quantity} item(s) to cart"
    except Exception as e:
        return False, f"Error adding to cart: {str(e)}"


def update_cart_item(product_id, quantity):
    """
    Update a cart item quantity
    
    Args:
        product_id: Product ID
        quantity: New quantity
        
    Returns:
        (success, message) tuple
    """
    try:
        # Validate quantity
        try:
            quantity = int(quantity)
            if quantity <= 0:
                return remove_from_cart(product_id)
        except:
            return False, "Invalid quantity"
        
        # Validate product
        product = mongo.db.products.find_one({'_id': ObjectId(product_id)})
        if not product:
            return False, "Product not found"
        
        # Check stock
        if product.get('stock_quantity', 0) < quantity:
            return False, "Not enough stock available"
        
        # Initialize cart if not exists
        if 'cart' not in session:
            session['cart'] = []
        
        # Update quantity if product is in cart
        found = False
        for item in session['cart']:
            if item.get('product_id') == product_id:
                item['quantity'] = quantity
                found = True
                break
        
        # Error if product not in cart
        if not found:
            return False, "Product not in cart"
        
        # Save cart to session
        session.modified = True
        
        return True, "Cart updated"
    except Exception as e:
        return False, f"Error updating cart: {str(e)}"


def remove_from_cart(product_id):
    """
    Remove a product from the cart
    
    Args:
        product_id: Product ID
        
    Returns:
        (success, message) tuple
    """
    try:
        # Initialize cart if not exists
        if 'cart' not in session:
            session['cart'] = []
        
        # Remove product from cart
        session['cart'] = [item for item in session['cart'] if item.get('product_id') != product_id]
        
        # Save cart to session
        session.modified = True
        
        return True, "Item removed from cart"
    except Exception as e:
        return False, f"Error removing from cart: {str(e)}"


def clear_cart():
    """
    Clear the cart
    
    Returns:
        (success, message) tuple
    """
    try:
        # Clear cart
        session['cart'] = []
        session.modified = True
        
        return True, "Cart cleared"
    except Exception as e:
        return False, f"Error clearing cart: {str(e)}"


def checkout_cart(user_id, shipping_address, payment_method='cash_on_delivery'):
    """
    Checkout the cart and create an order
    
    Args:
        user_id: User ID
        shipping_address: Shipping address
        payment_method: Payment method
        
    Returns:
        (success, message, order_id) tuple
    """
    try:
        # Get cart
        cart = get_cart()
        
        # Validate cart
        if not cart['items']:
            return False, "Cart is empty", None
        
        # Validate user
        user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if not user:
            return False, "User not found", None
        
        # Create order
        order = {
            'user_id': ObjectId(user_id),
            'products': [],
            'total_price': cart['total'],
            'order_date': datetime.now(),
            'status': 'pending',
            'shipping_address': shipping_address,
            'payment_method': payment_method
        }
        
        # Add products to order
        for item in cart['items']:
            order['products'].append({
                'product_id': ObjectId(item['id']),
                'name': item['name'],
                'price': item['price'],
                'quantity': item['quantity'],
                'subtotal': item['subtotal']
            })
            
            # Update stock quantity
            mongo.db.products.update_one(
                {'_id': ObjectId(item['id'])},
                {'$inc': {'stock_quantity': -item['quantity']}}
            )
            
            # Record purchase interaction
            interaction = {
                'user_id': ObjectId(user_id),
                'product_id': ObjectId(item['id']),
                'type': 'purchase',
                'timestamp': datetime.now()
            }
            mongo.db.interactions.insert_one(interaction)
        
        # Insert order
        order_id = mongo.db.orders.insert_one(order).inserted_id
        
        # Clear cart
        clear_cart()
        
        return True, "Order placed successfully", str(order_id)
    except Exception as e:
        return False, f"Error during checkout: {str(e)}", None