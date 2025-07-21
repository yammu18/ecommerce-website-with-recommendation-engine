# recommender/encoder_utils.py
import os
import pandas as pd
import logging
from models.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('recommender.encoder_utils')

def load_encoders_and_initialize(app_config, mongo_instance=None):
    """
    Explicitly load encoders and initialize them with all users and items from database
    
    Args:
        app_config: Application configuration
        mongo_instance: MongoDB connection
        
    Returns:
        Initialized DataProcessor object
    """
    # Create data processor
    data_processor = DataProcessor(app_config)
    
    # Set MongoDB instance
    if mongo_instance is not None:
        data_processor.mongo = mongo_instance
    
    # Path to saved encoders
    encoders_path = os.path.join(app_config.RECOMMENDER_MODELS_DIR, 'encoders.pkl')
    
    # First try to load saved encoders
    if os.path.exists(encoders_path):
        logger.info(f"Loading encoders from {encoders_path}")
        success = data_processor.load_encoders(encoders_path)
        if success:
            logger.info("Encoders loaded successfully from disk")
            
            # Validate encoder dimensions
            if hasattr(data_processor.user_encoder, 'classes_') and hasattr(data_processor.item_encoder, 'classes_'):
                logger.info(f"User encoder has {len(data_processor.user_encoder.classes_)} classes")
                logger.info(f"Item encoder has {len(data_processor.item_encoder.classes_)} classes")
                
                # Verify expected counts
                interaction_count = 0
                user_count = 0
                product_count = 0
                
                try:
                    if mongo_instance:
                        interaction_count = mongo_instance.db.interactions.count_documents({})
                        user_count = mongo_instance.db.users.count_documents({})
                        product_count = mongo_instance.db.products.count_documents({})
                        logger.info(f"Database contains {interaction_count} interactions, {user_count} users, {product_count} products")
                except Exception as e:
                    logger.error(f"Error querying counts: {e}")
                
                # Check if encoder dimensions match database counts
                encoder_users = len(data_processor.user_encoder.classes_)
                encoder_items = len(data_processor.item_encoder.classes_)
                
                # If significant mismatch, reinitialize
                if user_count > 0 and product_count > 0:
                    if encoder_users < user_count * 0.8 or encoder_items < product_count * 0.8:
                        logger.warning("Encoder dimensions don't match database counts, reinitializing...")
                        success = False
                    
            return data_processor if success else None
    
    # If loading fails or validation fails, initialize encoders with all users and items
    logger.info("Initializing encoders with all users and items from database")
    
    # Get all products and users
    product_df = data_processor.get_product_data()
    user_df = data_processor.get_user_data()
    
    # Get all interactions to ensure we capture all products and users
    interaction_df = data_processor.get_interaction_data()
    
    # Collect all user IDs and product IDs
    all_product_ids = set()
    all_user_ids = set()
    
    # Add from product data
    if not product_df.empty and 'product_id' in product_df.columns:
        all_product_ids.update(product_df['product_id'].apply(str).unique())
    elif not product_df.empty and '_id' in product_df.columns:
        all_product_ids.update(product_df['_id'].apply(str).unique())
    
    # Add from user data
    if not user_df.empty and 'user_id' in user_df.columns:
        all_user_ids.update(user_df['user_id'].apply(str).unique())
    elif not user_df.empty and '_id' in user_df.columns:
        all_user_ids.update(user_df['_id'].apply(str).unique())
    
    # Add from interaction data
    if not interaction_df.empty:
        if 'product_id' in interaction_df.columns:
            all_product_ids.update(interaction_df['product_id'].apply(str).unique())
        if 'user_id' in interaction_df.columns:
            all_user_ids.update(interaction_df['user_id'].apply(str).unique())
    
    # Convert to lists
    product_ids = list(all_product_ids)
    user_ids = list(all_user_ids)
    
    logger.info(f"Fitting encoders on {len(user_ids)} users and {len(product_ids)} products")
    
    # Verify we have data to fit
    if len(user_ids) == 0 or len(product_ids) == 0:
        logger.warning("No users or products found. Using dummy data for initialization.")
        # Create dummy data to avoid errors
        if len(user_ids) == 0:
            user_ids = [f"dummy_user_{i}" for i in range(10)]
        if len(product_ids) == 0:
            product_ids = [f"dummy_product_{i}" for i in range(100)]
    
    # Fit encoders
    data_processor.user_encoder.fit(user_ids)
    data_processor.item_encoder.fit(product_ids)
    
    # Verify encoders are properly initialized
    if not hasattr(data_processor.user_encoder, 'classes_') or not hasattr(data_processor.item_encoder, 'classes_'):
        logger.error("Encoders not properly initialized after fitting")
        return None
    
    # Save the initialized encoders
    data_processor.save_encoders(encoders_path)
    logger.info(f"Encoders saved to {encoders_path}")
    
    return data_processor

def verify_encoder_consistency(data_processor, mongo_instance=None):
    """
    Verify that encoders are consistent with the current database state
    
    Args:
        data_processor: DataProcessor object with encoders
        mongo_instance: MongoDB connection
        
    Returns:
        Boolean indicating whether encoders are consistent
    """
    if not hasattr(data_processor.user_encoder, 'classes_') or not hasattr(data_processor.item_encoder, 'classes_'):
        logger.error("Encoders not initialized")
        return False
    
    if mongo_instance is None:
        logger.error("MongoDB instance not provided")
        return False
    
    try:
        # Count items in database
        user_count = mongo_instance.db.users.count_documents({})
        product_count = mongo_instance.db.products.count_documents({})
        
        # Count items in encoders
        encoder_user_count = len(data_processor.user_encoder.classes_)
        encoder_product_count = len(data_processor.item_encoder.classes_)
        
        logger.info(f"Database: {user_count} users, {product_count} products")
        logger.info(f"Encoders: {encoder_user_count} users, {encoder_product_count} products")
        
        # Check if counts match approximately
        user_match = encoder_user_count >= user_count * 0.8  # Allow for some missing users
        product_match = encoder_product_count >= product_count * 0.8  # Allow for some missing products
        
        if not user_match:
            logger.warning(f"User count mismatch: encoder has {encoder_user_count}, db has {user_count}")
        
        if not product_match:
            logger.warning(f"Product count mismatch: encoder has {encoder_product_count}, db has {product_count}")
        
        return user_match and product_match
    except Exception as e:
        logger.error(f"Error verifying encoder consistency: {e}")
        return False