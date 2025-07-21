import os
import pymongo
import pandas as pd
import hashlib
from bson.objectid import ObjectId
import datetime

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ecommerce"]  # Use your actual database name

def import_interactions():
    """Import interactions from CSV"""
    # Check if interactions collection exists
    interaction_count = db.interactions.count_documents({})
    if interaction_count > 0:
        print(f"Interactions collection already contains {interaction_count} documents")
        response = input("Do you want to drop and recreate it? (y/n): ")
        if response.lower() != 'y':
            return False
        
        # Drop interactions collection
        db.interactions.drop()
        print("Interactions collection dropped")
    
    # Find CSV file
    csv_paths = [
        'data/raw/reduced_data.csv',
        'data/reduced_data.csv',
        'reduced_data.csv',r'C:\Users\Yamunaa.S.A\Downloads\ecommerce\data\raw\reduced_data.csv'
    ]
    
    df = None
    for path in csv_paths:
        if os.path.exists(path):
            try:
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
    
    # Function for consistent ID generation
    def generate_consistent_objectid(input_str):
        input_bytes = str(input_str).encode('utf-8')
        hash_bytes = hashlib.md5(input_bytes).digest()
        return ObjectId(hash_bytes[:12])
    
    # Create interactions
    print("Creating interaction documents...")
    interactions = []
    batch_size = 10000  # Larger batch size for interactions
    
    # Process all interactions from CSV
    count = 0
    for _, row in df.iterrows():
        # Generate IDs directly from the user and product IDs in the CSV
        user_id = generate_consistent_objectid(row['user_id'])
        product_id = generate_consistent_objectid(row['product_id']) 
        
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
        count += 1
        
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
    
    final_count = db.interactions.count_documents({})
    print(f"Successfully imported {final_count} interactions out of {count} records")
    return True

if __name__ == "__main__":
    import_interactions()