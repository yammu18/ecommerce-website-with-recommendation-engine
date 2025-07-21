from flask import Blueprint, render_template, request, redirect, url_for, flash, session, g
from bson.objectid import ObjectId
from services.product_service import (
    get_products, get_product, get_product_categories,
    create_product, update_product, delete_product,
    record_product_view, get_related_products
)
from controllers.auth_controller import login_required
from utils.decorators import admin_required
from recommender.engine import get_recommendations
import os
from werkzeug.utils import secure_filename

# Create blueprint
product_bp = Blueprint('product', __name__, url_prefix='/products')

@product_bp.route('/')
def index():
    """Display product listing"""
    # Get query parameters for filtering and pagination
    page = int(request.args.get('page', 1))
    category = request.args.get('category')
    search = request.args.get('search')
    sort_by = request.args.get('sort_by', 'name')
    sort_dir = request.args.get('sort_dir', 'asc')
    
    # Get products
    products, total_pages, current_page = get_products(
        page=page,
        category=category,
        search=search,
        sort_by=sort_by,
        sort_dir=sort_dir
    )
    
    # For debug purposes, you can check what's being returned
    # print(f"Debug: {len(products)} products found")
    # for product in products:
    #     print(f"Product: {product.get('name')}, ID: {product.get('_id', product.get('id'))}")
    
    # Get all categories for filter sidebar
    categories = get_product_categories()
    
    return render_template(
        'products/list.html',
        products=products,
        categories=categories,
        current_category=category,
        current_search=search,
        current_sort_by=sort_by,
        current_sort_dir=sort_dir,
        total_pages=total_pages,
        current_page=current_page
    )

# Rest of the file remains the same
@product_bp.route('/<product_id>')
def detail(product_id):
    """Display product detail"""
    # Get product
    product = get_product(product_id)
    if not product:
        flash('Product not found', 'danger')
        return redirect(url_for('product.index'))
    
    # Record view if user is logged in
    if g.user:
        record_product_view(str(g.user['_id']), product_id)
    
    # Get related products
    related_products = get_related_products(product_id)
    
    # Get personalized recommendations for this product if user is logged in
    recommended_products = []
    if g.user:
        user_id = str(g.user['_id'])
        recommended_products = get_recommendations(user_id, seed_product_id=product_id, limit=4)
    
    return render_template(
        'products/detail.html',
        product=product,
        related_products=related_products,
        recommended_products=recommended_products
    )

@product_bp.route('/search')
def search():
    """Handle product search"""
    query = request.args.get('q', '')
    if not query:
        return redirect(url_for('product.index'))
    
    # Redirect to index with search parameter
    return redirect(url_for('product.index', search=query))

@product_bp.route('/category/<category>')
def category(category):
    """Display products by category"""
    # Redirect to index with category parameter
    return redirect(url_for('product.index', category=category))



@product_bp.route('/admin')
@admin_required
def admin_index():
    """Admin product listing"""
    # Get all products without pagination
    products, _, _ = get_products(per_page=100)
    
    return render_template('admin/products/index.html', products=products)

@product_bp.route('/admin/create', methods=['GET', 'POST'])
@admin_required
def admin_create():
    """Create a new product (admin)"""
    if request.method == 'POST':
        # Process form data
        data = {
            'name': request.form.get('name'),
            'description': request.form.get('description'),
            'price': request.form.get('price'),
            'category': request.form.get('category'),
            'subcategory': request.form.get('subcategory'),
            'stock_quantity': request.form.get('stock_quantity', 0),
            'features': request.form.getlist('features')
        }
        
        # Handle image upload
        if 'image' in request.files:
            image = request.files['image']
            if image.filename:
                # Save image
                filename = secure_filename(image.filename)
                image_path = os.path.join('static', 'uploads', 'products', filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image.save(image_path)
                data['images'] = [image_path]
        
        success, message, product_id = create_product(data)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('product.admin_index'))
        else:
            flash(message, 'danger')
    
    # Get all categories for the form
    categories = get_product_categories()
    
    return render_template('admin/products/create.html', categories=categories)

@product_bp.route('/admin/edit/<product_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit(product_id):
    """Edit a product (admin)"""
    # Get product
    product = get_product(product_id)
    if not product:
        flash('Product not found', 'danger')
        return redirect(url_for('product.admin_index'))
    
    if request.method == 'POST':
        # Process form data
        data = {
            'name': request.form.get('name'),
            'description': request.form.get('description'),
            'price': request.form.get('price'),
            'category': request.form.get('category'),
            'subcategory': request.form.get('subcategory'),
            'stock_quantity': request.form.get('stock_quantity', 0),
            'features': request.form.getlist('features')
        }
        
        # Handle image upload
        if 'image' in request.files:
            image = request.files['image']
            if image.filename:
                # Save image
                filename = secure_filename(image.filename)
                image_path = os.path.join('static', 'uploads', 'products', filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image.save(image_path)
                data['images'] = product.get('images', []) + [image_path]
        
        success, message = update_product(product_id, data)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('product.admin_index'))
        else:
            flash(message, 'danger')
    
    # Get all categories for the form
    categories = get_product_categories()
    
    return render_template(
        'admin/products/edit.html',
        product=product,
        categories=categories
    )

@product_bp.route('/admin/delete/<product_id>', methods=['POST'])
@admin_required
def admin_delete(product_id):
    """Delete a product (admin)"""
    success, message = delete_product(product_id)
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'danger')
    
    return redirect(url_for('product.admin_index'))