# models/data_processor.py
# models/data_processor.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Import privacy module
from recommender.privacy import LocalDifferentialPrivacy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('recommender.data_processor')

class DataProcessor:
    """Data processor for recommendation models with privacy features"""
    
    def __init__(self, config, min_interactions=5, apply_privacy=True, epsilon=1.0, privacy_mechanism='laplace'):
        """
        Initialize data processor
        
        Args:
            config: Application configuration
            min_interactions: Minimum number of interactions for users/items to be included
            apply_privacy: Whether to apply privacy-preserving techniques
            epsilon: Privacy budget parameter (lower = more privacy)
            privacy_mechanism: Privacy mechanism to use ('laplace', 'gaussian', or 'randomized_response')
        """
        self.config = config
        self.min_interactions = min_interactions
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.mongo = None
        
        # For scaling ratings
        self.rating_median = None
        self.rating_iqr = None
        
        # Data statistics
        self.data_stats = {
            'total_users': 0,
            'total_items': 0,
            'rating_mean': 0,
            'rating_std': 0
        }
        
        # Privacy settings
        self.apply_privacy = apply_privacy
        if self.apply_privacy:
            self.ldp = LocalDifferentialPrivacy(epsilon=epsilon, mechanism=privacy_mechanism)
            logger.info(f"Privacy enabled with mechanism: {privacy_mechanism}, epsilon: {epsilon}")
    
    def prepare_training_data(self):
        """
        Prepare training data for recommendation models with privacy
        
        Returns:
            DataFrame with processed training data
        """
        try:
            # Fetch raw interaction data
            raw_data = self.get_interaction_data()
            
            if raw_data is None or len(raw_data) == 0:
                logger.error("No interaction data available")
                return None
            
            logger.info(f"Raw data: {len(raw_data)} interactions")
            
            # Apply privacy to raw data if enabled
            if self.apply_privacy:
                logger.info("Applying privacy transformations to raw data")
                raw_data = self.ldp.perturb_user_item_interactions(raw_data, rating_col='rating')
            
            # Process the data
            processed_data = self.fit_transform(raw_data)
            
            return processed_data
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_validation_data(self, validation_ratio=0.2):
        """
        Prepare validation data for model evaluation with privacy
        
        Args:
            validation_ratio: Portion of data to use for validation
            
        Returns:
            DataFrame with processed validation data
        """
        try:
            # Fetch raw interaction data
            raw_data = self.get_interaction_data()
            
            if raw_data is None or len(raw_data) == 0:
                logger.error("No interaction data available")
                return None
            
            # Sort by timestamp if available
            if 'timestamp' in raw_data.columns:
                raw_data = raw_data.sort_values('timestamp')
            
            # Split data
            train_size = int(len(raw_data) * (1 - validation_ratio))
            validation_data = raw_data.iloc[train_size:]
            
            # Apply privacy to validation data if enabled
            if self.apply_privacy:
                logger.info("Applying privacy transformations to validation data")
                validation_data = self.ldp.perturb_user_item_interactions(validation_data, rating_col='rating')
            
            # Process validation data using fitted encoders
            processed_validation = self.transform(validation_data)
            
            return processed_validation
        except Exception as e:
            logger.error(f"Error preparing validation data: {e}")
            return None
    
    # Original methods remain unchanged
    def get_interaction_data(self):
        """
        Fetch interaction data from MongoDB
        
        Returns:
            DataFrame of interaction data
        """
        if self.mongo is None:
            logger.error("MongoDB not connected")
            return None
        
        try:
            # Fetch all interactions from MongoDB
            interactions = list(self.mongo.db.interactions.find())
            
            if not interactions:
                logger.warning("No interactions found in database")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(interactions)
            
            # Rename columns if needed
            if '_id' in df.columns:
                df = df.rename(columns={'_id': 'interaction_id'})
            
            # Convert ObjectId to string
            for col in ['user_id', 'product_id']:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            
            # Only keep rating interactions
            if 'type' in df.columns:
                df = df[df['type'] == 'rating']
            
            # Extract only needed columns
            needed_cols = ['user_id', 'product_id', 'rating']
            if 'timestamp' in df.columns:
                needed_cols.append('timestamp')
            
            df = df[needed_cols]
            
            return df
        except Exception as e:
            logger.error(f"Error fetching interaction data: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    # Add a method to perturb embeddings when needed
    def perturb_embeddings(self, embeddings, l2_norm_bound=1.0):
        """
        Apply privacy to user or item embeddings if privacy is enabled
        
        Args:
            embeddings: Embedding vectors to perturb
            l2_norm_bound: Maximum L2 norm of the embeddings
            
        Returns:
            Perturbed embedding vectors
        """
        if self.apply_privacy:
            logger.info(f"Applying privacy to embeddings with shape {embeddings.shape}")
            return self.ldp.perturb_embeddings(embeddings, l2_norm_bound)
        return embeddings

    # Rest of the methods remain the same
    # ...
    def get_product_data(self):
        """
        Fetch product data from MongoDB
        
        Returns:
            DataFrame of product data
        """
        if self.mongo is None:
            logger.error("MongoDB not connected")
            return None
        
        try:
            # Fetch all products from MongoDB
            products = list(self.mongo.db.products.find())
            
            if not products:
                logger.warning("No products found in database")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(products)
            
            # Convert ObjectId to string
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching product data: {e}")
            return None
    
    def get_user_data(self):
        """
        Fetch user data from MongoDB
        
        Returns:
            DataFrame of user data
        """
        if self.mongo is None:
            logger.error("MongoDB not connected")
            return None
        
        try:
            # Fetch all users from MongoDB
            users = list(self.mongo.db.users.find())
            
            if not users:
                logger.warning("No users found in database")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(users)
            
            # Convert ObjectId to string
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching user data: {e}")
            return None
    
    def fit_transform(self, df):
        """
        Process and transform data for model training
        
        Args:
            df: DataFrame with raw interaction data
            
        Returns:
            DataFrame with processed data
        """
        # Create a copy to avoid warnings
        df = df.copy()
        
        # Filter users and items with minimum interactions
        user_counts = df['user_id'].value_counts()
        item_counts = df['product_id'].value_counts()
        
        filtered_df = df[
            df['user_id'].isin(user_counts[user_counts >= self.min_interactions].index) & 
            df['product_id'].isin(item_counts[item_counts >= self.min_interactions].index)
        ]
        
        if len(filtered_df) == 0:
            logger.warning("No data left after filtering by min_interactions")
            return filtered_df
        
        # Encode users and items
        filtered_df.loc[:, 'user_idx'] = self.user_encoder.fit_transform(filtered_df['user_id'])
        filtered_df.loc[:, 'item_idx'] = self.item_encoder.fit_transform(filtered_df['product_id'])
        
        # Update data statistics
        self.data_stats['total_users'] = len(self.user_encoder.classes_)
        self.data_stats['total_items'] = len(self.item_encoder.classes_)
        
        # Robust scaling for ratings
        self.rating_median = filtered_df['rating'].median()
        self.rating_iqr = filtered_df['rating'].quantile(0.75) - filtered_df['rating'].quantile(0.25)
        
        # Avoid division by zero
        if self.rating_iqr == 0:
            self.rating_iqr = 1.0
        
        # Scale ratings using robust scaling
        filtered_df.loc[:, 'rating_scaled'] = (filtered_df['rating'] - self.rating_median) / self.rating_iqr
        
        # Update rating statistics
        self.data_stats['rating_mean'] = filtered_df['rating'].mean()
        self.data_stats['rating_std'] = filtered_df['rating'].std()
        
        # Feature engineering
        user_counts = filtered_df.groupby('user_idx').size().reset_index(name='user_interaction_count')
        item_counts = filtered_df.groupby('item_idx').size().reset_index(name='item_interaction_count')
        
        filtered_df = pd.merge(filtered_df, user_counts, on='user_idx', how='left')
        filtered_df = pd.merge(filtered_df, item_counts, on='item_idx', how='left')
        
        # Normalize interaction counts
        user_count_mean = filtered_df['user_interaction_count'].mean()
        user_count_std = filtered_df['user_interaction_count'].std()
        if user_count_std == 0:
            user_count_std = 1.0
            
        item_count_mean = filtered_df['item_interaction_count'].mean()
        item_count_std = filtered_df['item_interaction_count'].std()
        if item_count_std == 0:
            item_count_std = 1.0
        
        filtered_df.loc[:, 'user_interaction_norm'] = (filtered_df['user_interaction_count'] - user_count_mean) / user_count_std
        filtered_df.loc[:, 'item_interaction_norm'] = (filtered_df['item_interaction_count'] - item_count_mean) / item_count_std
        
        return filtered_df
    
    def transform(self, df):
        """
        Transform new data using fitted encoders
        
        Args:
            df: DataFrame with raw data
            
        Returns:
            DataFrame with transformed data
        """
        # Create a copy to avoid warnings
        df = df.copy()
        
        # Apply encoders (for new users/items not in training, handle accordingly)
        user_ids = df['user_id'].values
        item_ids = df['product_id'].values
        
        # Transform user_ids that exist in the encoder's classes
        mask_user = np.isin(user_ids, self.user_encoder.classes_)
        df.loc[mask_user, 'user_idx'] = self.user_encoder.transform(user_ids[mask_user])
        # For user_ids that don't exist, assign a default value
        df.loc[~mask_user, 'user_idx'] = -1
        
        # Transform item_ids that exist in the encoder's classes
        mask_item = np.isin(item_ids, self.item_encoder.classes_)
        df.loc[mask_item, 'item_idx'] = self.item_encoder.transform(item_ids[mask_item])
        # For item_ids that don't exist, assign a default value
        df.loc[~mask_item, 'item_idx'] = -1
        
        # Filter out rows with unknown users or items
        df = df[(df['user_idx'] != -1) & (df['item_idx'] != -1)]
        
        if len(df) == 0:
            logger.warning("No data left after filtering out unknown users/items")
            return df
        
        # Scale ratings using robust scaling
        df.loc[:, 'rating_scaled'] = (df['rating'] - self.rating_median) / self.rating_iqr
        
        # Feature engineering
        user_counts = df.groupby('user_idx').size().reset_index(name='user_interaction_count')
        item_counts = df.groupby('item_idx').size().reset_index(name='item_interaction_count')
        
        df = pd.merge(df, user_counts, on='user_idx', how='left')
        df = pd.merge(df, item_counts, on='item_idx', how='left')
        
        # Normalize interaction counts
        user_count_mean = df['user_interaction_count'].mean()
        user_count_std = df['user_interaction_count'].std()
        if user_count_std == 0:
            user_count_std = 1.0
            
        item_count_mean = df['item_interaction_count'].mean()
        item_count_std = df['item_interaction_count'].std()
        if item_count_std == 0:
            item_count_std = 1.0
        
        df.loc[:, 'user_interaction_norm'] = (df['user_interaction_count'] - user_count_mean) / user_count_std
        df.loc[:, 'item_interaction_norm'] = (df['item_interaction_count'] - item_count_mean) / item_count_std
        
        return df
    
    def save_encoders(self, file_path):
        """
        Save encoders to file
        
        Args:
            file_path: Path to save encoders
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'user_encoder': self.user_encoder,
                    'item_encoder': self.item_encoder,
                    'rating_median': self.rating_median,
                    'rating_iqr': self.rating_iqr,
                    'data_stats': self.data_stats
                }, f)
            logger.info(f"Encoders saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving encoders: {e}")
            return False
    
    def load_encoders(self, file_path):
        """
        Load encoders from file
        
        Args:
            file_path: Path to load encoders from
            
        Returns:
            Boolean indicating success
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Encoder file not found: {file_path}")
                return False
                
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            self.user_encoder = data['user_encoder']
            self.item_encoder = data['item_encoder']
            self.rating_median = data['rating_median']
            self.rating_iqr = data['rating_iqr']
            self.data_stats = data['data_stats']
            
            logger.info(f"Encoders loaded from {file_path}")
            logger.info(f"Loaded {len(self.user_encoder.classes_)} users, {len(self.item_encoder.classes_)} items")
            
            return True
        except Exception as e:
            logger.error(f"Error loading encoders: {e}")
            return False
    
    def get_n_users(self):
        """Get number of users"""
        if hasattr(self.user_encoder, 'classes_'):
            return len(self.user_encoder.classes_)
        return 0
    
    def get_n_items(self):
        """Get number of items"""
        if hasattr(self.item_encoder, 'classes_'):
            return len(self.item_encoder.classes_)
        return 0
    
    def encode_user(self, user_id):
        """
        Encode user ID to internal index
        
        Args:
            user_id: User ID to encode
            
        Returns:
            Encoded user index or None if not found
        """
        if not hasattr(self.user_encoder, 'classes_'):
            logger.error("User encoder not fitted")
            return None
            
        try:
            user_id = str(user_id)
            if user_id in self.user_encoder.classes_:
                return self.user_encoder.transform([user_id])[0]
            return None
        except Exception as e:
            logger.error(f"Error encoding user: {e}")
            return None
    
    def encode_item(self, item_id):
        """
        Encode item ID to internal index
        
        Args:
            item_id: Item ID to encode
            
        Returns:
            Encoded item index or None if not found
        """
        if not hasattr(self.item_encoder, 'classes_'):
            logger.error("Item encoder not fitted")
            return None
            
        try:
            item_id = str(item_id)
            if item_id in self.item_encoder.classes_:
                return self.item_encoder.transform([item_id])[0]
            return None
        except Exception as e:
            logger.error(f"Error encoding item: {e}")
            return None
    
    def encode_items(self, item_ids):
        """
        Encode multiple item IDs to internal indices
        
        Args:
            item_ids: List of item IDs to encode
            
        Returns:
            List of encoded item indices
        """
        if not hasattr(self.item_encoder, 'classes_'):
            logger.error("Item encoder not fitted")
            return []
            
        try:
            # Convert to strings
            item_ids = [str(iid) for iid in item_ids]
            
            # Filter out items not in encoder
            valid_items = [iid for iid in item_ids if iid in self.item_encoder.classes_]
            
            if not valid_items:
                return []
                
            return self.item_encoder.transform(valid_items).tolist()
        except Exception as e:
            logger.error(f"Error encoding items: {e}")
            return []
    
    def decode_items(self, item_indices):
        """
        Decode item indices to item IDs
        
        Args:
            item_indices: List of item indices to decode
            
        Returns:
            List of decoded item IDs
        """
        if not hasattr(self.item_encoder, 'classes_'):
            logger.error("Item encoder not fitted")
            return []
            
        try:
            return self.item_encoder.inverse_transform(item_indices).tolist()
        except Exception as e:
            logger.error(f"Error decoding items: {e}")
            return []
    
    def get_user_interactions(self, user_id):
        """
        Get all items a user has interacted with
        
        Args:
            user_id: User ID
            
        Returns:
            List of item indices the user has interacted with
        """
        if self.mongo is None:
            logger.error("MongoDB not connected")
            return []
            
        try:
            # Encode user ID
            user_idx = self.encode_user(user_id)
            
            if user_idx is None:
                return []
                
            # Fetch interactions from MongoDB
            user_interactions = list(self.mongo.db.interactions.find({"user_id": user_id}))
            
            # Extract product IDs
            product_ids = [str(interaction["product_id"]) for interaction in user_interactions]
            
            # Encode product IDs
            item_indices = self.encode_items(product_ids)
            
            return item_indices
        except Exception as e:
            logger.error(f"Error getting user interactions: {e}")
            return []
    
    def get_all_items(self):
        """
        Get all valid item indices
        
        Returns:
            List of all item indices
        """
        if not hasattr(self.item_encoder, 'classes_'):
            logger.error("Item encoder not fitted")
            return []
            
        try:
            return list(range(len(self.item_encoder.classes_)))
        except Exception as e:
            logger.error(f"Error getting all items: {e}")
            return []
    
    def inverse_scale_rating(self, scaled_rating):
        """
        Convert scaled rating back to original scale
        
        Args:
            scaled_rating: Scaled rating value
            
        Returns:
            Original scale rating
        """
        if self.rating_median is None or self.rating_iqr is None:
            logger.error("Rating scaler not fitted")
            return scaled_rating
            
        return (scaled_rating * self.rating_iqr) + self.rating_median