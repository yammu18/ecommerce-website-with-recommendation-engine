# recommender/privacy.py
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('recommender.privacy')

class LocalDifferentialPrivacy:
    """
    Implements Local Differential Privacy techniques for recommender systems.
    Enhances privacy by adding controlled noise to user data before processing.
    """
    
    def __init__(self, epsilon=1.0, mechanism='laplace', delta=1e-5):
        """
        Initialize the LDP module.
        
        Args:
            epsilon: Privacy budget parameter (lower = more privacy, less accuracy)
            mechanism: Privacy mechanism to use ('laplace', 'gaussian', or 'randomized_response')
            delta: Additional privacy parameter for Gaussian mechanism (usually small)
        """
        self.epsilon = epsilon
        self.mechanism = mechanism
        self.delta = delta
        
        logger.info(f"Initialized LDP with epsilon={epsilon}, mechanism={mechanism}, delta={delta}")
    
    def _laplace_mechanism(self, data, sensitivity=1.0):
        """
        Apply Laplace mechanism to add noise to numerical data.
        
        Args:
            data: Numerical data to perturb
            sensitivity: Maximum change in the function output when one record changes
            
        Returns:
            Perturbed data with Laplace noise
        """
        # Scale parameter is sensitivity/epsilon
        scale = sensitivity / self.epsilon
        
        # Add Laplace noise to the data
        if isinstance(data, np.ndarray):
            noise = np.random.laplace(0, scale, size=data.shape)
            return data + noise
        else:
            # For scalar values
            noise = np.random.laplace(0, scale)
            return data + noise
    
    def _gaussian_mechanism(self, data, sensitivity=1.0):
        """
        Apply Gaussian mechanism to add noise to numerical data.
        
        Args:
            data: Numerical data to perturb
            sensitivity: Maximum change in the function output when one record changes
            
        Returns:
            Perturbed data with Gaussian noise
        """
        # Calculate sigma based on epsilon and delta
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        
        # Add Gaussian noise to the data
        if isinstance(data, np.ndarray):
            noise = np.random.normal(0, sigma, size=data.shape)
            return data + noise
        else:
            # For scalar values
            noise = np.random.normal(0, sigma)
            return data + noise
    
    def _randomized_response(self, data, categories=None):
        """
        Apply randomized response for categorical data.
        
        Args:
            data: Categorical data to perturb
            categories: List of possible categories (if None, extracts from data)
            
        Returns:
            Perturbed categorical data
        """
        if categories is None:
            if isinstance(data, np.ndarray):
                categories = np.unique(data)
            else:
                # Assume it's a single value
                return self._randomized_response_value(data)
        
        # Apply randomized response to array
        if isinstance(data, np.ndarray):
            result = np.zeros_like(data)
            for i in range(len(data)):
                result[i] = self._randomized_response_value(data[i], categories)
            return result
        else:
            return self._randomized_response_value(data, categories)
    
    def _randomized_response_value(self, value, categories=None):
        """
        Apply randomized response to a single value.
        
        Args:
            value: Value to perturb
            categories: List of possible categories
            
        Returns:
            Perturbed value
        """
        if categories is None:
            # Binary case
            categories = [0, 1]
            
        n_categories = len(categories)
        
        # Probability of keeping the true value
        p_keep = np.exp(self.epsilon) / (np.exp(self.epsilon) + n_categories - 1)
        
        # Probability of choosing a random value (including the true one)
        p_random = 1 - p_keep
        
        # Generate a random value to decide whether to keep or randomize
        if np.random.random() < p_keep:
            # Keep the original value
            return value
        else:
            # Choose a random category with uniform probability
            return np.random.choice(categories)
    
    def perturb_ratings(self, ratings, rating_range=(1, 5)):
        """
        Add privacy-preserving noise to user ratings.
        
        Args:
            ratings: Array of user ratings to protect
            rating_range: Tuple of (min_rating, max_rating)
            
        Returns:
            Privacy-protected ratings
        """
        min_rating, max_rating = rating_range
        sensitivity = 1.0  # Assuming a user can change rating by at most 1
        
        # Log the privacy operation
        logger.info(f"Perturbing ratings with {self.mechanism} mechanism, epsilon={self.epsilon}")
        logger.info(f"Rating range: {rating_range}, sensitivity: {sensitivity}")
        
        if self.mechanism == 'laplace':
            perturbed = self._laplace_mechanism(ratings, sensitivity)
        elif self.mechanism == 'gaussian':
            perturbed = self._gaussian_mechanism(ratings, sensitivity)
        elif self.mechanism == 'randomized_response':
            # For ratings, create discrete categories
            categories = np.arange(min_rating, max_rating + 1, 0.5)
            perturbed = self._randomized_response(ratings, categories)
        else:
            logger.warning(f"Unknown mechanism: {self.mechanism}, using laplace as default")
            perturbed = self._laplace_mechanism(ratings, sensitivity)
        
        # Clip to valid rating range
        clipped = np.clip(perturbed, min_rating, max_rating)
        
        # Log privacy impact
        if isinstance(ratings, np.ndarray) and len(ratings) > 0:
            avg_noise = np.mean(np.abs(clipped - ratings))
            logger.info(f"Average noise magnitude: {avg_noise:.4f}")
        
        return clipped
    
    def perturb_embeddings(self, embeddings, l2_norm_bound=1.0):
        """
        Apply privacy-preserving noise to embeddings (useful for protecting user latent factors).
        
        Args:
            embeddings: User or item embedding vectors
            l2_norm_bound: Maximum L2 norm of the embeddings
            
        Returns:
            Privacy-protected embeddings
        """
        # Early exit if no embeddings
        if embeddings is None or len(embeddings) == 0:
            return embeddings
            
        logger.info(f"Perturbing embeddings with shape {embeddings.shape}")
        
        # Clip embeddings by L2 norm if needed
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        scaling = np.minimum(1.0, l2_norm_bound / (norms + 1e-8))  # Avoid division by zero
        clipped_embeddings = embeddings * scaling
        
        # Add noise based on the chosen mechanism
        if self.mechanism == 'laplace':
            perturbed = self._laplace_mechanism(clipped_embeddings, l2_norm_bound)
        elif self.mechanism == 'gaussian':
            perturbed = self._gaussian_mechanism(clipped_embeddings, l2_norm_bound)
        else:
            # Default to Laplace for embeddings if randomized response was selected
            logger.warning(f"Using Laplace mechanism for embeddings instead of {self.mechanism}")
            perturbed = self._laplace_mechanism(clipped_embeddings, l2_norm_bound)
        
        # Log privacy impact
        avg_noise = np.mean(np.linalg.norm(perturbed - clipped_embeddings, axis=1))
        logger.info(f"Average embedding noise magnitude: {avg_noise:.4f}")
            
        return perturbed
    
    def perturb_user_item_interactions(self, df, rating_col='rating'):
        """
        Apply privacy-preserving noise to user-item interaction data.
        
        Args:
            df: DataFrame with user-item interactions
            rating_col: Column name containing ratings
            
        Returns:
            DataFrame with privacy-protected interactions
        """
        try:
            # Early exit if dataframe is empty
            if df is None or len(df) == 0:
                return df
                
            # Create a copy to avoid modifying the original
            df_private = df.copy()
            
            # Apply noise to ratings
            if rating_col in df_private.columns:
                ratings = df_private[rating_col].values
                
                # Determine rating range from data
                min_rating = df_private[rating_col].min()
                max_rating = df_private[rating_col].max()
                
                # Apply privacy mechanism
                private_ratings = self.perturb_ratings(
                    ratings, 
                    rating_range=(min_rating, max_rating)
                )
                
                # Update the dataframe
                df_private[rating_col] = private_ratings
                
                # Log privacy impact
                avg_noise = np.mean(np.abs(private_ratings - ratings))
                logger.info(f"Applied {self.mechanism} noise to {len(ratings)} ratings "
                          f"with avg magnitude: {avg_noise:.4f}")
            else:
                logger.warning(f"Rating column '{rating_col}' not found in DataFrame")
            
            return df_private
        except Exception as e:
            logger.error(f"Error applying privacy to interactions: {e}")
            import traceback
            traceback.print_exc()
            return df

    def create_privacy_layer(self, sensitivity=1.0):
        """
        Create a Keras Lambda layer that adds privacy noise during model training
        
        Args:
            sensitivity: The sensitivity of the data
            
        Returns:
            Lambda layer that adds privacy noise
        """
        def add_noise_to_outputs(x):
            # Determine which mechanism to use
            if self.mechanism == 'laplace':
                scale = sensitivity / self.epsilon
                noise = tf.random.stateless_normal(
                    tf.shape(x), 
                    seed=[42, 43],  # Fixed seed for reproducibility
                    mean=0.0, 
                    stddev=scale
                )
            elif self.mechanism == 'gaussian':
                sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
                noise = tf.random.stateless_normal(
                    tf.shape(x), 
                    seed=[42, 43],  # Fixed seed for reproducibility
                    mean=0.0, 
                    stddev=sigma
                )
            else:
                # Default to Laplace
                scale = sensitivity / self.epsilon
                noise = tf.random.stateless_normal(
                    tf.shape(x), 
                    seed=[42, 43],  # Fixed seed for reproducibility
                    mean=0.0, 
                    stddev=scale
                )
            
            # Only add noise during training
            return x + noise * tf.cast(tf.keras.backend.learning_phase(), tf.float32)
        
        return Lambda(add_noise_to_outputs)

class PrivacyAwareRecommender:
    """
    Wrapper for recommendation models to add privacy-preserving features.
    This class adds local differential privacy to existing recommendation models.
    """
    
    def __init__(self, base_model, epsilon=1.0, mechanism='laplace', delta=1e-5):
        """
        Initialize privacy-aware recommender.
        
        Args:
            base_model: Base recommendation model to wrap
            epsilon: Privacy budget parameter
            mechanism: Privacy mechanism to use
            delta: Additional privacy parameter for Gaussian mechanism
        """
        self.base_model = base_model
        self.ldp = LocalDifferentialPrivacy(epsilon=epsilon, mechanism=mechanism, delta=delta)
        self.epsilon = epsilon
        self.mechanism = mechanism
        self.delta = delta
        
        logger.info(f"Initialized privacy-aware recommender wrapper with epsilon={epsilon}")
    
    def _add_privacy_layers_to_model(self, model):
        """
        Add privacy noise layers to a Keras model
        
        Args:
            model: Keras model to modify
            
        Returns:
            Modified model with privacy layers
        """
        # This is more complex to implement and requires modifying model architecture
        # For simplicity, we'll keep the original model and just add noise to data
        return model
    
    def train(self, df, **kwargs):
        """
        Train the model with privacy-preserving data.
        
        Args:
            df: DataFrame with interaction data
            **kwargs: Additional arguments to pass to the base model's train method
            
        Returns:
            Training history
        """
        # Apply privacy-preserving noise to the training data
        logger.info(f"Training with privacy (ε={self.epsilon}, mechanism={self.mechanism})")
        private_df = self.ldp.perturb_user_item_interactions(df)
        
        # Train the base model with private data
        try:
            # Log metrics for comparison
            if 'rating' in df.columns and 'rating' in private_df.columns:
                original_mean = df['rating'].mean()
                private_mean = private_df['rating'].mean()
                original_std = df['rating'].std()
                private_std = private_df['rating'].std()
                
                logger.info(f"Original data: mean={original_mean:.4f}, std={original_std:.4f}")
                logger.info(f"Private data: mean={private_mean:.4f}, std={private_std:.4f}")
            
            # Train with private data
            return self.base_model.train(private_df, **kwargs)
        except Exception as e:
            logger.error(f"Error in privacy-aware training: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to non-private training
            logger.warning("Falling back to non-private training")
            return self.base_model.train(df, **kwargs)
    
    def recommend_for_user(self, user_id, top_k=10):
        """
        Generate privacy-preserving recommendations.
        
        Args:
            user_id: User ID to generate recommendations for
            top_k: Number of recommendations to generate
            
        Returns:
            List of recommended item IDs
        """
        # Get base recommendations
        if hasattr(self.base_model, 'recommend_for_user'):
            recommendations, scores = self.base_model.recommend_for_user(user_id, top_k)
        else:
            logger.error("Base model doesn't have recommend_for_user method")
            return [], []
        
        # Add privacy noise to scores
        private_scores = self.ldp.perturb_ratings(scores, rating_range=(min(scores) if len(scores) > 0 else 0, 
                                                                       max(scores) if len(scores) > 0 else 5))
        
        # Re-sort based on private scores
        if len(recommendations) > 0:
            private_recommendations = [rec for _, rec in sorted(zip(private_scores, recommendations), reverse=True)]
            return private_recommendations[:top_k], private_scores[:top_k]
        else:
            return [], []

    def find_similar_items(self, item_id, top_k=10):
        """
        Find similar items with privacy protection.
        
        Args:
            item_id: Item ID to find similar items for
            top_k: Number of similar items to return
            
        Returns:
            List of similar item IDs
        """
        # If base model has find_similar_items method, use it
        if hasattr(self.base_model, 'find_similar_items'):
            similar_items, similarity_scores = self.base_model.find_similar_items(item_id, top_k)
            
            # Add privacy noise to similarity scores
            if len(similarity_scores) > 0:
                private_scores = self.ldp.perturb_ratings(
                    similarity_scores, 
                    rating_range=(min(similarity_scores), max(similarity_scores))
                )
                
                # Re-sort based on private scores
                private_similar_items = [item for _, item in sorted(zip(private_scores, similar_items), reverse=True)]
                return private_similar_items[:top_k], private_scores[:top_k]
            
        return [], []