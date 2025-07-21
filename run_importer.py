import os
import sys
import hashlib
import pandas as pd
import numpy as np
import datetime
import pymongo
from bson.objectid import ObjectId

# Connect to MongoDB directly
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ecommerce"]  # Use your actual database name
mongo = type('obj', (object,), {'db': db})

def generate_consistent_objectid(input_str):
    """
    Generate an ObjectId that is consistent for the same input string
    by using the first 12 bytes of the MD5 hash
    """
    # Convert to string and encode to bytes
    input_bytes = str(input_str).encode('utf-8')
    # Generate MD5 hash
    hash_bytes = hashlib.md5(input_bytes).digest()
    # Use first 12 bytes for ObjectId
    return ObjectId(hash_bytes[:12])

def import_static_dataset():
    """Import static dataset from CSV into MongoDB with enhanced product fields"""
    try:
        # First, clear existing static dataset products (those with external_id field)
        print("Removing existing static dataset products...")
        products_removed = db.products.delete_many({"external_id": {"$exists": True}})
        print(f"Removed {products_removed.deleted_count} existing static dataset products")
        
        # Try different possible file paths
        csv_paths = [
            'data/raw/reduced_data.csv',
            'data/reduced_data.csv',
            'reduced_data.csv',
            r'C:\Users\Yamunaa.S.A\Downloads\ecommerce\data\raw\reduced_data.csv'
        ]
        
        df = None
        for path in csv_paths:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    print(f"Loaded {len(df)} records from {path}")
                    break
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        if df is None:
            print("Could not find or load the dataset CSV file.")
            return False
        
        # Rename columns if necessary
        if 'prod_id' in df.columns and 'product_id' not in df.columns:
            df = df.rename(columns={'prod_id': 'product_id'})
        
        # Validate required columns
        required_columns = ['user_id', 'product_id', 'rating']
        if not all(col in df.columns for col in required_columns):
            print(f"CSV is missing required columns. Found: {df.columns.tolist()}")
            return False
        
        # Summary statistics
        unique_users = df['user_id'].nunique()
        unique_products = df['product_id'].nunique()
        print(f"Dataset contains {unique_users} unique users and {unique_products} unique products")
        
        # Predefined categories for random assignment
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', 'Toys', 'Health', 'Automotive', 'Garden']
        subcategories = {
            'Electronics': ['Smartphones', 'Laptops', 'Audio', 'Cameras', 'Accessories'],
            'Clothing': ['Men', 'Women', 'Children', 'Shoes', 'Accessories'],
            'Books': ['Fiction', 'Non-fiction', 'Educational', 'Comics', 'Magazines'],
            'Home': ['Kitchen', 'Furniture', 'Decor', 'Appliances', 'Bedding'],
            'Sports': ['Fitness', 'Outdoor', 'Team Sports', 'Water Sports', 'Winter Sports'],
            'Beauty': ['Skincare', 'Makeup', 'Haircare', 'Fragrance', 'Bath & Body'],
            'Toys': ['Action Figures', 'Dolls', 'Educational', 'Games', 'Outdoor Toys'],
            'Health': ['Vitamins', 'Supplements', 'Personal Care', 'Medical Supplies', 'Fitness Equipment'],
            'Automotive': ['Interior', 'Exterior', 'Tools', 'Parts', 'Accessories'],
            'Garden': ['Plants', 'Tools', 'Furniture', 'Decor', 'Irrigation']
        }
        
        # Create users first (only if they don't already exist)
        print("Checking user documents...")
        existing_user_count = db.users.count_documents({})
        if existing_user_count < unique_users:
            print(f"Creating missing user documents... (existing: {existing_user_count}, required: {unique_users})")
            users = []
            all_user_ids = df['user_id'].unique()
            
            # Get existing external IDs to avoid duplicates
            existing_ext_ids = set(str(u['external_id']) for u in db.users.find({}, {'external_id': 1}))
            
            for external_user_id in all_user_ids:
                # Skip if already exists
                if str(external_user_id) in existing_ext_ids:
                    continue
                    
                # Generate deterministic ObjectId from user_id
                user_id = generate_consistent_objectid(external_user_id)
                
                # Create user document
                user = {
                    '_id': user_id,
                    'external_id': str(external_user_id),
                    'username': f'User_{external_user_id}',
                    'email': f'user_{external_user_id}@example.com',
                    'age': np.random.randint(18, 65),
                    'gender': np.random.choice(['male', 'female']),
                    'created_at': datetime.datetime.now()
                }
                users.append(user)
                
                # Insert in batches
                if len(users) >= 1000:
                    try:
                        db.users.insert_many(users)
                        print(f"Inserted batch of {len(users)} users")
                        users = []
                    except Exception as e:
                        print(f"Error inserting users: {e}")
            
            # Insert remaining users
            if users:
                try:
                    db.users.insert_many(users)
                    print(f"Inserted final batch of {len(users)} users")
                except Exception as e:
                    print(f"Error inserting final users: {e}")
        else:
            print(f"Sufficient users already exist: {existing_user_count}")
        
        # Create products with enhanced fields
        print("Creating enhanced product documents...")
        products = []
        all_product_ids = df['product_id'].unique()
        
        # Pre-compute average ratings for each product
        print("Computing product ratings from interactions...")
        product_ratings = df.groupby('product_id')['rating'].agg(['mean', 'count']).reset_index()
        ratings_dict = dict(zip(product_ratings['product_id'], product_ratings['mean']))
        
        for external_product_id in all_product_ids:
            # Generate deterministic ObjectId from product_id
            product_id = generate_consistent_objectid(external_product_id)
            
            # Pick consistent category based on product_id
            category_index = hash(str(external_product_id)) % len(categories)
            category = categories[category_index]
            
            # Pick subcategory if available
            subcategory = None
            if category in subcategories:
                subcat_index = hash(str(external_product_id) + '_sub') % len(subcategories[category])
                subcategory = subcategories[category][subcat_index]
            
            # Get average rating for this product
            ratings_average = ratings_dict.get(external_product_id, 0)
            if ratings_average > 0:
                # Round to 1 decimal place
                ratings_average = round(ratings_average, 1)
            
            # Generate created and updated timestamps
            created_at = datetime.datetime.now() - datetime.timedelta(days=np.random.randint(1, 365))
            updated_at = created_at + datetime.timedelta(days=np.random.randint(0, 30))
            
            # Generate features array
            num_features = np.random.randint(2, 6)
            features = []
            possible_features = [
                "Waterproof", "Eco-friendly", "Portable", "Rechargeable", 
                "Wireless", "Foldable", "Adjustable", "Durable", 
                "Lightweight", "Compact", "Multi-functional", "Energy-efficient"
            ]
            for _ in range(num_features):
                feature = np.random.choice(possible_features)
                if feature not in features:
                    features.append(feature)
            
            # Generate stock quantity
            stock_quantity = np.random.randint(1, 100)
            
            # Generate images array
            num_images = np.random.randint(1, 4)
            images = [f"image_{external_product_id}_{i}.jpg" for i in range(num_images)]
            
            # Enhanced name that's more descriptive
            name = f"Product {external_product_id}"
            if category and subcategory:
                name = f"{subcategory} {name}"
            
            # Enhanced description
            description = f"High-quality {category.lower()} product with " + ", ".join(features).lower() + " features."
            
            # Create enhanced product document
            product = {
                '_id': product_id,
                'external_id': str(external_product_id),
                'name': name,
                'category': category,
                'subcategory': subcategory,
                'price': round(np.random.uniform(10, 500), 2),
                'description': description,
                'stock_quantity': stock_quantity,
                'features': features,
                'ratings_average': ratings_average,
                'images': images,
                'created_at': created_at,
                'updated_at': updated_at,
                'in_stock': stock_quantity > 0
            }
            products.append(product)
            
            # Insert in batches
            if len(products) >= 1000:
                try:
                    db.products.insert_many(products)
                    print(f"Inserted batch of {len(products)} products")
                    products = []
                except Exception as e:
                    print(f"Error inserting products: {e}")
        
        # Insert remaining products
        if products:
            try:
                db.products.insert_many(products)
                print(f"Inserted final batch of {len(products)} products")
            except Exception as e:
                print(f"Error inserting final products: {e}")
        
        # Create interactions (only if needed)
        existing_interaction_count = db.interactions.count_documents({})
        if existing_interaction_count < len(df):
            print(f"Creating missing interaction documents... (existing: {existing_interaction_count}, total: {len(df)})")
            
            # Should we delete existing interactions?
            print("Clearing existing interactions linked to static dataset...")
            db.interactions.delete_many({"user_id": {"$in": [generate_consistent_objectid(uid) for uid in df['user_id'].unique()]}})
            
            interactions = []
            batch_size = 10000  # Larger batch size for interactions
            
            # Create mapping dictionaries for faster lookups
            user_id_map = {str(u['external_id']): u['_id'] for u in db.users.find({}, {'external_id': 1})}
            product_id_map = {str(p['external_id']): p['_id'] for p in db.products.find({}, {'external_id': 1})}
            
            for _, row in df.iterrows():
                # Get user and product IDs from mapping
                user_id = user_id_map.get(str(row['user_id']))
                product_id = product_id_map.get(str(row['product_id']))
                
                # Skip if user or product not found
                if not user_id or not product_id:
                    continue
                
                # Create interaction
                timestamp = datetime.datetime.fromtimestamp(row['timestamp']) if 'timestamp' in row else datetime.datetime.now()
                
                interaction = {
                    '_id': ObjectId(),
                    'user_id': user_id,
                    'product_id': product_id,
                    'type': 'rating',
                    'rating': float(row['rating']),
                    'timestamp': timestamp
                }
                
                interactions.append(interaction)
                
                # Insert in batches
                if len(interactions) >= batch_size:
                    try:
                        db.interactions.insert_many(interactions)
                        print(f"Inserted batch of {len(interactions)} interactions")
                        interactions = []
                    except Exception as e:
                        print(f"Error inserting interactions: {e}")
            
            # Insert remaining interactions
            if interactions:
                try:
                    db.interactions.insert_many(interactions)
                    print(f"Inserted final batch of {len(interactions)} interactions")
                except Exception as e:
                    print(f"Error inserting final interactions: {e}")
        else:
            print(f"Sufficient interactions already exist: {existing_interaction_count}")
        
        # Final count check
        user_count = db.users.count_documents({})
        product_count = db.products.count_documents({})
        interaction_count = db.interactions.count_documents({})
        
        print(f"Import complete. Database now contains:")
        print(f"- {user_count} users (target: {unique_users})")
        print(f"- {product_count} products (target: {unique_products})")
        print(f"- {interaction_count} interactions")
        
        print("Enhanced static dataset import completed successfully")
        return True
    except Exception as e:
        print(f"Error importing static dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run the import function
if __name__ == "__main__":
    import_static_dataset()