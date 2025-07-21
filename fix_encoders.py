import os
import pymongo
import pickle
from sklearn.preprocessing import LabelEncoder
from bson.objectid import ObjectId

def force_encoder_initialization():
    """Force initialization of encoders with full product set"""
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["ecommerce"]  # Use your actual database name
        
        # Get all user and product IDs
        print("Fetching user IDs from database...")
        user_ids = [str(u["_id"]) for u in db.users.find({}, {"_id": 1})]
        
        print("Fetching product IDs from database...")
        product_ids = [str(p["_id"]) for p in db.products.find({}, {"_id": 1})]
        
        print(f"Found {len(user_ids)} users and {len(product_ids)} products in database")
        
        # Create and train encoders
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        print("Training user encoder...")
        user_encoder.fit(user_ids)
        
        print("Training item encoder...")
        item_encoder.fit(product_ids)
        
        print(f"Initialized user encoder with {len(user_encoder.classes_)} classes")
        print(f"Initialized item encoder with {len(item_encoder.classes_)} classes")
        
        # Find model directories and clear any existing encoder files
        possible_dirs = [
            'data/models',
            'data/data/models',
            os.path.join(os.getcwd(), 'data', 'models')
        ]
        
        models_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                models_dir = dir_path
                print(f"Found models directory: {models_dir}")
                
                # Check for existing encoder files
                encoder_file = os.path.join(dir_path, 'encoders.pkl')
                if os.path.exists(encoder_file):
                    print(f"Removing existing encoder file: {encoder_file}")
                    os.remove(encoder_file)
                break
        
        if models_dir is None:
            models_dir = os.path.join(os.getcwd(), 'data', 'models')
            os.makedirs(models_dir, exist_ok=True)
            print(f"Created models directory: {models_dir}")
        
        # Save encoders
        encoder_file = os.path.join(models_dir, 'encoders.pkl')
        with open(encoder_file, 'wb') as f:
            pickle.dump({
                'user_encoder': user_encoder,
                'item_encoder': item_encoder,
                'scaler': None  # Will be initialized later
            }, f)
        
        print(f"Saved encoders to {encoder_file}")
        return True
    
    except Exception as e:
        print(f"Error initializing encoders: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_interactions():
    """
    Fix the interactions collection to use the correct product IDs
    """
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["ecommerce"]  # Use your actual database name
        
        # Check if interactions collection exists and has content
        interaction_count = db.interactions.count_documents({})
        if interaction_count > 0:
            print(f"Interactions collection already has {interaction_count} documents")
            return True
        
        # Load CSV file for interactions
        import pandas as pd
        
        csv_paths = [
            'data/raw/reduced_data.csv',
            'data/reduced_data.csv',
            'reduced_data.csv'
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
            
        # Hash function for consistent ObjectIDs
        def generate_consistent_objectid(input_str):
            import hashlib
            input_bytes = str(input_str).encode('utf-8')
            hash_bytes = hashlib.md5(input_bytes).digest()
            return ObjectId(hash_bytes[:12])
            
        # Create interactions
        print("Creating interaction documents...")
        interactions = []
        batch_size = 10000  # Larger batch size for interactions
        
        # Instead of using the user and product mapping, we'll generate the IDs directly
        for _, row in df.iterrows():
            # Generate consistent ObjectIDs for user and product
            user_id = generate_consistent_objectid(row['user_id'])
            product_id = generate_consistent_objectid(row['product_id'])
            
            # Create interaction
            timestamp = pd.to_datetime(row['timestamp'], unit='s') if 'timestamp' in row else datetime.datetime.now()
            
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
                    return False
        
        # Insert remaining interactions
        if interactions:
            try:
                db.interactions.insert_many(interactions)
                print(f"Inserted final batch of {len(interactions)} interactions")
            except Exception as e:
                print(f"Error inserting final interactions: {e}")
                return False
                
        return True
    
    except Exception as e:
        print(f"Error fixing interactions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # First, fix the interactions if needed
    if fix_interactions():
        print("Interactions fixed successfully")
    else:
        print("Failed to fix interactions")
        
    # Then initialize the encoders
    if force_encoder_initialization():
        print("Encoders initialized successfully")
    else:
        print("Failed to initialize encoders")