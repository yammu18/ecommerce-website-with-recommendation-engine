# init_recommender.py
import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from sklearn.preprocessing import LabelEncoder
from config import config
from database.db import init_db

def create_encoders_pkl():
    """
    Create encoders.pkl file with user and item encoders
    """
    print("Creating encoders.pkl file...")
    
    # Create a minimal Flask app for initialization
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Initialize MongoDB
    mongo = PyMongo(app)
    
    # Initialize database
    init_db(app, mongo)
    
    # Create models directory if it doesn't exist
    models_dir = app.config.get('RECOMMENDER_MODELS_DIR', os.path.join(os.getcwd(), 'data', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, 'hybrid_recommender'), exist_ok=True)
    
    # Get all users
    users = list(mongo.db.users.find())
    user_ids = [str(user['_id']) for user in users]
    
    # Get all products
    products = list(mongo.db.products.find())
    product_ids = [str(product['_id']) for product in products]
    
    # Create encoders
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    # Fit encoders (ensure we have at least some IDs)
    if not user_ids:
        user_ids = ['dummy_user_1', 'dummy_user_2']
    if not product_ids:
        product_ids = ['dummy_product_1', 'dummy_product_2']
    
    user_encoder.fit(user_ids)
    item_encoder.fit(product_ids)
    
    # Create data stats
    data_stats = {
        'total_users': len(user_ids),
        'total_items': len(product_ids),
        'rating_mean': 3.5,
        'rating_std': 1.0
    }
    
    # Create encoders dictionary
    encoders_data = {
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'rating_median': 3.0,
        'rating_iqr': 2.0,
        'data_stats': data_stats
    }
    
    # Save encoders to file
    encoders_path = os.path.join(models_dir, 'encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders_data, f)
    
    print(f"Encoders saved to {encoders_path}")
    print(f"User encoder classes: {len(user_encoder.classes_)}")
    print(f"Item encoder classes: {len(item_encoder.classes_)}")
    
    return True

def create_hybrid_metadata_pkl():
    """
    Create metadata.pkl file for hybrid recommender
    """
    print("Creating hybrid recommender metadata.pkl file...")
    
    # Create models directory path
    try:
        # Try to access config attributes
        if hasattr(config, 'RECOMMENDER_MODELS_DIR'):
            models_dir = config.RECOMMENDER_MODELS_DIR
        else:
            models_dir = os.path.join(os.getcwd(), 'data', 'models')
    except:
        # Fallback to default path
        models_dir = os.path.join(os.getcwd(), 'data', 'models')
        
    hybrid_dir = os.path.join(models_dir, 'hybrid_recommender')
    os.makedirs(hybrid_dir, exist_ok=True)
    
    # Create metadata
    metadata = {
        'popularity_weight': 0.3,
        'category_weight': 0.3,
        'similarity_weight': 0.4,
        'cold_start_threshold': 5,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save metadata to file
    metadata_path = os.path.join(hybrid_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Hybrid recommender metadata saved to {metadata_path}")
    
    return True
if __name__ == "__main__":
    # Create encoders.pkl file
    encoders_created = create_encoders_pkl()
    
    # Create hybrid recommender metadata.pkl file
    metadata_created = create_hybrid_metadata_pkl()
    
    if encoders_created and metadata_created:
        print("Successfully created all necessary PKL files for the recommendation engine.")
    else:
        print("Error creating PKL files.")