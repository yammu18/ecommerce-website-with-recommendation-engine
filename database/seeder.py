"""
Database seeder to populate initial data for development
"""
import json
import os
import random
from datetime import datetime, timedelta
from bson.objectid import ObjectId
from flask_bcrypt import Bcrypt
from database.db import mongo

def seed_database(app):
    """
    Seed the database with initial data for development
    
    Args:
        app: Flask application
    """
    with app.app_context():
        bcrypt = Bcrypt(app)
        
        # Check if database is already seeded
        if mongo.db.users.count_documents({}) > 0:
            print("Database already seeded. Skipping...")
            return
        
        print("Seeding database...")
        
        # Seed users
        seed_users(bcrypt)
        
        # Seed products
        seed_products()
        
        # Seed interactions
        seed_interactions()
        
        # Seed orders
        seed_orders()
        
        print("Database seeding completed!")

def seed_users(bcrypt):
    """Seed users collection"""
    print("Seeding users...")
    
    # Create admin user
    admin_user = {
        '_id': ObjectId(),
        'username': 'admin',
        'email': 'admin@example.com',
        'password': bcrypt.generate_password_hash('admin123').decode('utf-8'),
        'roles': ['admin', 'user'],
        'full_name': 'Admin User',
        'phone': '555-123-4567',
        'address': {
            'street': '123 Admin St',
            'city': 'Admin City',
            'state': 'AS',
            'postal_code': '12345',
            'country': 'US'
        },
        'created_at': datetime.now() - timedelta(days=30),
        'updated_at': datetime.now(),
        'last_login': datetime.now()
    }
    
    # Create demo user
    demo_user = {
        '_id': ObjectId(),
        'username': 'demo',
        'email': 'demo@example.com',
        'password': bcrypt.generate_password_hash('password123').decode('utf-8'),
        'roles': ['user'],
        'full_name': 'Demo User',
        'phone': '555-987-6543',
        'address': {
            'street': '456 User Ave',
            'city': 'Demo City',
            'state': 'DS',
            'postal_code': '67890',
            'country': 'US'
        },
        'created_at': datetime.now() - timedelta(days=15),
        'updated_at': datetime.now(),
        'last_login': datetime.now()
    }
    
    # Create regular users
    regular_users = []
    for i in range(1, 21):
        user = {
            '_id': ObjectId(),
            'username': f'user{i}',
            'email': f'user{i}@example.com',
            'password': bcrypt.generate_password_hash(f'password{i}').decode('utf-8'),
            'roles': ['user'],
            'full_name': f'User {i}',
            'phone': f'555-{random.randint(100, 999)}-{random.randint(1000, 9999)}',
            'address': {
                'street': f'{random.randint(100, 999)} Main St',
                'city': random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']),
                'state': random.choice(['NY', 'CA', 'IL', 'TX', 'AZ']),
                'postal_code': f'{random.randint(10000, 99999)}',
                'country': 'US'
            },
            'created_at': datetime.now() - timedelta(days=random.randint(1, 30)),
            'updated_at': datetime.now(),
            'last_login': datetime.now() - timedelta(days=random.randint(0, 10))
        }
        regular_users.append(user)
    
    # Insert users
    mongo.db.users.insert_many([admin_user, demo_user] + regular_users)
    
    print(f"Inserted {mongo.db.users.count_documents({})} users")


def seed_products():
    """Seed products collection"""
    print("Seeding products...")
    
    categories = [
        'electronics', 'clothing', 'books', 'home', 'beauty', 
        'sports', 'toys', 'jewelry', 'health', 'food'
    ]
    
    products = []
    
    # Electronics
    electronics_products = [
        {
            'name': 'Smartphone X Pro',
            'description': 'Latest smartphone with amazing features. High resolution camera, long battery life, and powerful processor.',
            'price': 799.99,
            'category': 'electronics',
            'subcategory': 'smartphones',
            'stock_quantity': random.randint(50, 200),
            'features': ['5G Compatible', 'Triple Camera', '8GB RAM', '128GB Storage'],
            'ratings_average': round(random.uniform(3.5, 5.0), 1),
            'images': ['images/products/smartphone_1.jpg']
        },
        {
            'name': 'Laptop Ultra Slim',
            'description': 'Lightweight laptop for professionals and students. Fast performance and long battery life.',
            'price': 1299.99,
            'category': 'electronics',
            'subcategory': 'laptops',
            'stock_quantity': random.randint(25, 100),
            'features': ['Intel i7 Processor', '16GB RAM', '512GB SSD', 'Backlit Keyboard'],
            'ratings_average': round(random.uniform(4.0, 5.0), 1),
            'images': ['images/products/laptop_1.jpg']
        },
        {
            'name': 'Wireless Headphones',
            'description': 'Premium wireless headphones with noise cancellation technology. Incredible sound quality and comfort.',
            'price': 249.99,
            'category': 'electronics',
            'subcategory': 'audio',
            'stock_quantity': random.randint(75, 250),
            'features': ['Noise Cancellation', 'Bluetooth 5.0', '30 Hour Battery Life', 'Voice Assistant'],
            'ratings_average': round(random.uniform(4.0, 5.0), 1),
            'images': ['images/products/headphones_1.jpg']
        },
        {
            'name': 'Smart Watch Series 5',
            'description': 'Advanced smart watch with health monitoring, GPS, and seamless connectivity with your phone.',
            'price': 349.99,
            'category': 'electronics',
            'subcategory': 'wearables',
            'stock_quantity': random.randint(50, 150),
            'features': ['Heart Rate Monitor', 'GPS', 'Water Resistant', 'Fitness Tracking'],
            'ratings_average': round(random.uniform(3.8, 4.9), 1),
            'images': ['images/products/smartwatch_1.jpg']
        },
        {
            'name': '4K Ultra HD Smart TV',
            'description': 'Immersive viewing experience with crystal clear 4K resolution and smart features.',
            'price': 699.99,
            'category': 'electronics',
            'subcategory': 'televisions',
            'stock_quantity': random.randint(30, 80),
            'features': ['4K Resolution', 'Smart TV', 'HDR', 'Voice Control'],
            'ratings_average': round(random.uniform(4.2, 4.8), 1),
            'images': ['images/products/tv_1.jpg']
        }
    ]
    products.extend(electronics_products)
    
    # Clothing
    clothing_products = [
        {
            'name': 'Classic Denim Jeans',
            'description': 'Premium quality denim jeans with perfect fit and comfort. Durable and stylish for everyday wear.',
            'price': 59.99,
            'category': 'clothing',
            'subcategory': 'jeans',
            'stock_quantity': random.randint(100, 300),
            'features': ['100% Cotton', 'Slim Fit', 'Multiple Colors Available'],
            'ratings_average': round(random.uniform(4.0, 4.8), 1),
            'images': ['images/products/jeans_1.jpg']
        },
        {
            'name': 'Casual T-Shirt',
            'description': 'Soft and comfortable t-shirt for everyday wear. Made from high-quality material that lasts.',
            'price': 24.99,
            'category': 'clothing',
            'subcategory': 't-shirts',
            'stock_quantity': random.randint(150, 400),
            'features': ['100% Organic Cotton', 'Regular Fit', 'Machine Washable'],
            'ratings_average': round(random.uniform(3.9, 4.7), 1),
            'images': ['images/products/tshirt_1.jpg']
        },
        {
            'name': 'Formal Dress Shirt',
            'description': 'Elegant dress shirt for formal occasions. Tailored fit and wrinkle-resistant fabric.',
            'price': 49.99,
            'category': 'clothing',
            'subcategory': 'shirts',
            'stock_quantity': random.randint(75, 200),
            'features': ['Non-Iron', 'Tailored Fit', 'Cotton Blend'],
            'ratings_average': round(random.uniform(4.1, 4.9), 1),
            'images': ['images/products/dress_shirt_1.jpg']
        },
        {
            'name': 'Wool Winter Sweater',
            'description': 'Warm and cozy sweater for cold winter days. Luxurious wool blend that keeps you comfortable.',
            'price': 79.99,
            'category': 'clothing',
            'subcategory': 'sweaters',
            'stock_quantity': random.randint(60, 150),
            'features': ['Wool Blend', 'Ribbed Cuffs', 'Hand Wash'],
            'ratings_average': round(random.uniform(4.0, 4.6), 1),
            'images': ['images/products/sweater_1.jpg']
        },
        {
            'name': 'Athletic Running Shoes',
            'description': 'Lightweight and supportive running shoes. Perfect for jogging, training, or everyday wear.',
            'price': 89.99,
            'category': 'clothing',
            'subcategory': 'shoes',
            'stock_quantity': random.randint(80, 250),
            'features': ['Breathable Mesh', 'Cushioned Insole', 'Rubber Outsole'],
            'ratings_average': round(random.uniform(4.2, 4.9), 1),
            'images': ['images/products/running_shoes_1.jpg']
        }
    ]
    products.extend(clothing_products)
    
    # Books
    book_products = [
        {
            'name': 'The Art of Programming',
            'description': 'Comprehensive guide to programming fundamentals and advanced concepts. Perfect for beginners and experienced developers.',
            'price': 34.99,
            'category': 'books',
            'subcategory': 'programming',
            'stock_quantity': random.randint(100, 300),
            'features': ['Paperback', '500 Pages', 'Beginner to Advanced'],
            'ratings_average': round(random.uniform(4.3, 4.9), 1),
            'images': ['images/products/book_programming_1.jpg']
        },
        {
            'name': 'Mystery at Midnight',
            'description': 'Thrilling mystery novel that will keep you on the edge of your seat until the very last page.',
            'price': 19.99,
            'category': 'books',
            'subcategory': 'fiction',
            'stock_quantity': random.randint(120, 350),
            'features': ['Hardcover', '320 Pages', 'Award Winning'],
            'ratings_average': round(random.uniform(4.0, 4.8), 1),
            'images': ['images/products/book_mystery_1.jpg']
        },
        {
            'name': 'Healthy Cooking Made Easy',
            'description': 'Collection of nutritious and delicious recipes for the health-conscious cook. Simple instructions and beautiful photography.',
            'price': 29.99,
            'category': 'books',
            'subcategory': 'cooking',
            'stock_quantity': random.randint(90, 250),
            'features': ['Hardcover', '200 Recipes', 'Nutritional Information'],
            'ratings_average': round(random.uniform(4.2, 4.7), 1),
            'images': ['images/products/book_cooking_1.jpg']
        },
        {
            'name': 'Financial Freedom Guide',
            'description': 'Step-by-step guide to achieving financial independence. Practical advice on saving, investing, and building wealth.',
            'price': 24.99,
            'category': 'books',
            'subcategory': 'finance',
            'stock_quantity': random.randint(85, 200),
            'features': ['Paperback', '250 Pages', 'Beginner Friendly'],
            'ratings_average': round(random.uniform(4.1, 4.6), 1),
            'images': ['images/products/book_finance_1.jpg']
        },
        {
            'name': 'World History Encyclopedia',
            'description': 'Comprehensive reference book covering major historical events from ancient civilizations to modern times.',
            'price': 49.99,
            'category': 'books',
            'subcategory': 'reference',
            'stock_quantity': random.randint(50, 150),
            'features': ['Hardcover', '1000+ Pages', 'Illustrated'],
            'ratings_average': round(random.uniform(4.4, 4.9), 1),
            'images': ['images/products/book_history_1.jpg']
        }
    ]
    products.extend(book_products)
    
    # Add more product categories...
    # Home products
    home_products = [
        {
            'name': 'Modern Coffee Table',
            'description': 'Elegant coffee table with modern design. Perfect centerpiece for any living room.',
            'price': 199.99,
            'category': 'home',
            'subcategory': 'furniture',
            'stock_quantity': random.randint(30, 100),
            'features': ['Solid Wood', 'Easy Assembly', 'Spacious Storage'],
            'ratings_average': round(random.uniform(4.0, 4.7), 1),
            'images': ['images/products/coffee_table_1.jpg']
        },
        {
            'name': 'Luxury Bedding Set',
            'description': 'Premium quality bedding set with soft sheets, pillowcases, and duvet cover. Ultimate comfort for better sleep.',
            'price': 129.99,
            'category': 'home',
            'subcategory': 'bedding',
            'stock_quantity': random.randint(70, 200),
            'features': ['100% Egyptian Cotton', 'Queen Size', 'Machine Washable'],
            'ratings_average': round(random.uniform(4.2, 4.8), 1),
            'images': ['images/products/bedding_1.jpg']
        },
        {
            'name': 'Smart Home Security Camera',
            'description': 'High-definition security camera with motion detection, night vision, and smartphone alerts.',
            'price': 89.99,
            'category': 'home',
            'subcategory': 'security',
            'stock_quantity': random.randint(60, 180),
            'features': ['1080p HD', 'Motion Detection', 'Two-way Audio', 'Cloud Storage'],
            'ratings_average': round(random.uniform(4.1, 4.9), 1),
            'images': ['images/products/security_camera_1.jpg']
        },
        {
            'name': 'Ceramic Dinner Set',
            'description': 'Beautiful ceramic dinner set for 6 people. Elegant design for everyday use or special occasions.',
            'price': 149.99,
            'category': 'home',
            'subcategory': 'kitchenware',
            'stock_quantity': random.randint(40, 120),
            'features': ['18-Piece Set', 'Dishwasher Safe', 'Microwave Safe'],
            'ratings_average': round(random.uniform(4.0, 4.7), 1),
            'images': ['images/products/dinner_set_1.jpg']
        },
        {
            'name': 'Aromatherapy Essential Oil Diffuser',
            'description': 'Ultrasonic diffuser that spreads essential oils throughout your space. Creates a relaxing atmosphere with soothing light.',
            'price': 39.99,
            'category': 'home',
            'subcategory': 'decor',
            'stock_quantity': random.randint(90, 250),
            'features': ['300ml Capacity', '7 LED Light Colors', 'Auto Shut-off'],
            'ratings_average': round(random.uniform(4.2, 4.8), 1),
            'images': ['images/products/diffuser_1.jpg']
        }
    ]
    products.extend(home_products)
    
    # Beauty products
    beauty_products = [
        {
            'name': 'Premium Skincare Set',
            'description': 'Complete skincare routine with cleanser, toner, moisturizer, and serum. Made with natural ingredients.',
            'price': 79.99,
            'category': 'beauty',
            'subcategory': 'skincare',
            'stock_quantity': random.randint(80, 220),
            'features': ['Natural Ingredients', 'Paraben Free', 'For All Skin Types'],
            'ratings_average': round(random.uniform(4.3, 4.9), 1),
            'images': ['images/products/skincare_1.jpg']
        },
        {
            'name': 'Professional Hair Dryer',
            'description': 'Salon-quality hair dryer with multiple heat and speed settings. Ionic technology for smooth, frizz-free results.',
            'price': 59.99,
            'category': 'beauty',
            'subcategory': 'hair',
            'stock_quantity': random.randint(70, 190),
            'features': ['1800 Watts', 'Ionic Technology', '3 Heat Settings', '2 Speed Settings'],
            'ratings_average': round(random.uniform(4.0, 4.7), 1),
            'images': ['images/products/hair_dryer_1.jpg']
        },
        {
            'name': 'Luxury Perfume',
            'description': 'Elegant fragrance with floral and woody notes. Long-lasting scent for special occasions.',
            'price': 69.99,
            'category': 'beauty',
            'subcategory': 'fragrance',
            'stock_quantity': random.randint(60, 180),
            'features': ['50ml Bottle', 'Long Lasting', 'Signature Scent'],
            'ratings_average': round(random.uniform(4.2, 4.8), 1),
            'images': ['images/products/perfume_1.jpg']
        },
        {
            'name': 'Natural Organic Makeup Set',
            'description': 'Complete makeup set with foundation, concealer, blush, and eyeshadow. Made with organic ingredients.',
            'price': 89.99,
            'category': 'beauty',
            'subcategory': 'makeup',
            'stock_quantity': random.randint(75, 200),
            'features': ['Organic Ingredients', 'Cruelty Free', 'Long Lasting'],
            'ratings_average': round(random.uniform(4.1, 4.7), 1),
            'images': ['images/products/makeup_1.jpg']
        },
        {
            'name': 'Electric Shaver',
            'description': 'Advanced electric shaver for a smooth and comfortable shave. Rechargeable battery for cordless use.',
            'price': 49.99,
            'category': 'beauty',
            'subcategory': 'shaving',
            'stock_quantity': random.randint(65, 180),
            'features': ['Rechargeable', 'Waterproof', '60 Minute Runtime'],
            'ratings_average': round(random.uniform(4.0, 4.6), 1),
            'images': ['images/products/shaver_1.jpg']
        }
    ]
    products.extend(beauty_products)
    
    # Add timestamps and ObjectId to products
    for product in products:
        product['_id'] = ObjectId()
        product['created_at'] = datetime.now() - timedelta(days=random.randint(1, 30))
        product['updated_at'] = product['created_at']
    
    # Insert products
    mongo.db.products.insert_many(products)
    
    print(f"Inserted {mongo.db.products.count_documents({})} products")


def seed_interactions():
    """Seed interactions collection"""
    print("Seeding interactions...")
    
    # Get user and product IDs
    users = list(mongo.db.users.find({}, {'_id': 1}))
    products = list(mongo.db.products.find({}, {'_id': 1, 'category': 1}))
    
    # Interaction types and their weights
    interaction_types = {
        'view': 10,   # Most common
        'add_to_cart': 5,
        'purchase': 3,
        'rating': 2    # Least common
    }
    
    interactions = []
    
    # Generate interactions for each user
    for user in users:
        # Number of interactions per user (random between 10-50)
        n_interactions = random.randint(10, 50)
        
        # Select random products for this user
        user_products = random.sample(products, min(n_interactions, len(products)))
        
        # Generate interactions
        for product in user_products:
            # Determine interaction type based on weights
            interaction_types_list = []
            for itype, weight in interaction_types.items():
                interaction_types_list.extend([itype] * weight)
            
            interaction_type = random.choice(interaction_types_list)
            
            # Create interaction
            interaction = {
                '_id': ObjectId(),
                'user_id': user['_id'],
                'product_id': product['_id'],
                'type': interaction_type,
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 30), 
                                                      hours=random.randint(0, 23), 
                                                      minutes=random.randint(0, 59))
            }
            
            # Add rating value if it's a rating interaction
            if interaction_type == 'rating':
                interaction['rating'] = random.randint(1, 5)
            
            interactions.append(interaction)
    
    # Sort interactions by timestamp
    interactions.sort(key=lambda x: x['timestamp'])
    
    # Insert interactions
    mongo.db.interactions.insert_many(interactions)
    
    print(f"Inserted {mongo.db.interactions.count_documents({})} interactions")


def seed_orders():
    """Seed orders collection"""
    print("Seeding orders...")
    
    # Get users
    users = list(mongo.db.users.find({}, {'_id': 1, 'address': 1}))
    
    # Get purchase interactions and group by user
    purchases = list(mongo.db.interactions.find({'type': 'purchase'}, 
                                             {'user_id': 1, 'product_id': 1, 'timestamp': 1}))
    
    # Group purchases by user
    user_purchases = {}
    for purchase in purchases:
        user_id = str(purchase['user_id'])
        if user_id not in user_purchases:
            user_purchases[user_id] = []
        user_purchases[user_id].append(purchase)
    
    # Order statuses with weights
    statuses = {
        'delivered': 5,
        'shipped': 2,
        'processing': 1,
        'pending': 1,
        'cancelled': 1
    }
    
    orders = []
    
    # Generate orders for each user that has purchases
    for user in users:
        user_id_str = str(user['_id'])
        
        if user_id_str not in user_purchases:
            continue
        
        # Group purchases by approximate date (day)
        user_purchase_days = {}
        for purchase in user_purchases[user_id_str]:
            day = purchase['timestamp'].strftime('%Y-%m-%d')
            if day not in user_purchase_days:
                user_purchase_days[day] = []
            user_purchase_days[day].append(purchase)
        
        # Create an order for each day with purchases
        for day, day_purchases in user_purchase_days.items():
            # Determine order status based on weights
            status_list = []
            for status, weight in statuses.items():
                status_list.extend([status] * weight)
            order_status = random.choice(status_list)
            
            # Use the earliest timestamp for the order date
            order_date = min(p['timestamp'] for p in day_purchases)
            
            # Get products for this order
            order_products = []
            order_total = 0
            
            for purchase in day_purchases:
                # Get product details
                product = mongo.db.products.find_one({'_id': purchase['product_id']})
                
                # Skip if product not found
                if not product:
                    continue
                
                # Random quantity between 1 and 3
                quantity = random.randint(1, 3)
                
                # Add to order products
                order_product = {
                    'product_id': product['_id'],
                    'name': product['name'],
                    'price': product['price'],
                    'quantity': quantity,
                    'subtotal': product['price'] * quantity
                }
                order_products.append(order_product)
                
                # Add to order total
                order_total += order_product['subtotal']
            
            # Create the order
            order = {
                '_id': ObjectId(),
                'user_id': user['_id'],
                'products': order_products,
                'total_price': order_total,
                'order_date': order_date,
                'status': order_status,
                'shipping_address': user['address'],
                'payment_method': random.choice(['credit_card', 'paypal', 'cash_on_delivery'])
            }
            
            orders.append(order)
    
    # Insert orders
    if orders:
        mongo.db.orders.insert_many(orders)
    
    print(f"Inserted {mongo.db.orders.count_documents({})} orders")


if __name__ == '__main__':
    """Run seeder directly"""
    from flask import Flask
    from flask_pymongo import PyMongo
    
    app = Flask(__name__)
    app.config['MONGO_URI'] = 'mongodb://localhost:27017/ecommerce_dev'
    
    mongo_instance = PyMongo(app)
    mongo.mongo = mongo_instance
    
    seed_database(app)