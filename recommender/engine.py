import os
import pickle
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from recommender.custom_objects import get_custom_objects, load_model_with_custom_objects
from tensorflow.keras.models import load_model
import logging
from threading import Thread
from datetime import datetime, timedelta
from scipy.sparse import csr_matrix
from apscheduler.schedulers.background import BackgroundScheduler

# Import models
from models.ncf_model import NCFModel
from models.deepfm_model import DeepFMModel
# Removed hybrid model import
from models.data_processor import DataProcessor

# Import privacy components
from recommender.privacy import LocalDifferentialPrivacy, PrivacyAwareRecommender

# Create a global variable for the recommender instance
recommender = None
mongo_instance = None
scheduler = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('recommender.engine')

def init_recommender(config, mongo=None, apply_privacy=True, epsilon=1.0, privacy_mechanism='laplace'):
    """
    Initialize the recommendation engine with configuration and MongoDB connection
    
    Args:
        config: Application configuration
        mongo: MongoDB connection
        apply_privacy: Whether to apply privacy-preserving techniques
        epsilon: Privacy budget parameter (lower = more privacy)
        privacy_mechanism: Privacy mechanism to use ('laplace', 'gaussian', or 'randomized_response')
    """
    global recommender, mongo_instance
    
    # Store MongoDB instance for later use
    mongo_instance = mongo
    
    try:
        # Check if recommender already exists
        if recommender is not None:
            logger.info("Recommender already initialized")
            return True
            
        # Create models directory if it doesn't exist
        # Handle missing configuration gracefully
        models_dir = getattr(config, 'RECOMMENDER_MODELS_DIR', os.path.join(os.getcwd(), 'data', 'models'))
        os.makedirs(models_dir, exist_ok=True)
        
        # Create a config dictionary to pass to the recommender
        config_dict = {
            'RECOMMENDER_MODELS_DIR': models_dir,
            'RECOMMENDER_BATCH_SIZE': getattr(config, 'RECOMMENDER_BATCH_SIZE', 100),
            'RECOMMENDER_TOP_K': getattr(config, 'RECOMMENDER_TOP_K', 10),
            'NCF_FACTORS': getattr(config, 'NCF_FACTORS', 64),
            'NCF_LAYERS': getattr(config, 'NCF_LAYERS', [128, 64, 32]),
            'DEEPFM_EMBEDDING_SIZE': getattr(config, 'DEEPFM_EMBEDDING_SIZE', 64),
            'VAE_LATENT_DIM': getattr(config, 'VAE_LATENT_DIM', 32),
            'NUM_WORKERS': getattr(config, 'NUM_WORKERS', 4),
            'APPLY_PRIVACY': apply_privacy,
            'PRIVACY_EPSILON': epsilon,
            'PRIVACY_MECHANISM': privacy_mechanism
        }
        
        # Create a simple object to hold the configuration
        class ConfigObject:
            pass
        
        config_obj = ConfigObject()
        for key, value in config_dict.items():
            setattr(config_obj, key, value)
        
        # Initialize the recommender with our config object and privacy settings
        recommender = RecommendationEngine(config_obj, mongo, 
                                         apply_privacy=apply_privacy, 
                                         epsilon=epsilon, 
                                         privacy_mechanism=privacy_mechanism)
        
        # Load model if available, otherwise schedule training
        if recommender.load():
            logger.info("Loaded pre-trained recommendation models")
        else:
            logger.info("No pre-trained models found, scheduling training")
            schedule_training()
            
        return True
    except Exception as e:
        logger.error(f"Error initializing recommender: {e}")
        import traceback
        traceback.print_exc()
        return False


def schedule_training(delay_minutes=1):
    """
    Schedule model training to run in the background
    
    Args:
        delay_minutes: Minutes to wait before training
    """
    global scheduler
    
    if scheduler is None:
        scheduler = BackgroundScheduler()
        scheduler.start()
    
    # Schedule the training to run after a delay
    run_time = datetime.now() + timedelta(minutes=delay_minutes)
    scheduler.add_job(train_models, 'date', run_date=run_time, id='train_models')
    
    logger.info(f"Scheduled model training to run at {run_time}")
    
def train_models():
    """Train recommendation models in the background"""
    global recommender
    
    if recommender is None:
        logger.error("Cannot train models: Recommender not initialized")
        return {"success": False, "error": "Recommender not initialized"}
    
    try:
        # Train in a separate thread to avoid blocking the main thread
        thread = Thread(target=_train_models_thread)
        thread.daemon = True
        thread.start()
        
        return {"success": True, "message": "Model training started in background"}
    except Exception as e:
        logger.error(f"Error scheduling model training: {e}")
        return {"success": False, "error": str(e)}

def force_train_models():
    """Force immediate training of recommendation models"""
    global recommender
    
    if recommender is None:
        logger.error("Cannot train models: Recommender not initialized")
        return {"success": False, "error": "Recommender not initialized"}
    
    try:
        # Run training directly
        return _train_models_thread()
    except Exception as e:
        logger.error(f"Error in force training models: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def _train_models_thread():
    """Internal function to train models in a separate thread"""
    global recommender
    
    try:
        logger.info("Starting model training process")
        start_time = time.time()
        
        # Train the models
        results = recommender.train()
        
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Save the models
        recommender.save()
        
        return {
            "success": True, 
            "results": results, 
            "training_time": training_time
        }
    except Exception as e:
        logger.error(f"Error training models: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
        
def get_recommendations(user_id, seed_product_id=None, seed_product_ids=None, limit=10):
    """
    Get personalized product recommendations for a user
    
    Args:
        user_id: User ID to get recommendations for
        seed_product_id: Single product ID to use as seed for recommendations
        seed_product_ids: List of product IDs to use as seeds for recommendations
        limit: Maximum number of recommendations to return
        
    Returns:
        List of recommended products with scores
    """
    global recommender, mongo_instance
    
    if recommender is None or mongo_instance is None:
        logger.error("Recommender or MongoDB not initialized")
        return []
    
    try:
        # Convert seed_product_id to a list if provided
        if seed_product_id is not None:
            seed_ids = [seed_product_id]
        elif seed_product_ids is not None:
            seed_ids = seed_product_ids
        else:
            seed_ids = None
            
        # Get raw recommendations
        if seed_ids:
            # Logic for seed-based recommendations
            # This would depend on your implementation, but might involve:
            # 1. Getting similar products to each seed
            # 2. Combining these with user preferences
            # 3. Sorting by combined scores
            
            # Example placeholder implementation:
            all_similar_products = []
            for seed_id in seed_ids:
                similar_ids, scores = recommender.find_similar_items(seed_id, top_k=limit)
                for sid, score in zip(similar_ids, scores):
                    all_similar_products.append((sid, score))
                    
            # Combine with user personalization if needed
            # This is just an example - your actual implementation would depend on your needs
            user_recs, user_scores = recommender.recommend_for_user(user_id, top_k=limit)
            
            # Combine and sort by score
            combined_recs = {}
            for pid, score in all_similar_products:
                if pid in combined_recs:
                    combined_recs[pid] += score * 0.7  # Weight for similarity
                else:
                    combined_recs[pid] = score * 0.7
                    
            for pid, score in zip(user_recs, user_scores):
                if pid in combined_recs:
                    combined_recs[pid] += score * 0.3  # Weight for user personalization
                else:
                    combined_recs[pid] = score * 0.3
                    
            # Sort by score
            sorted_recs = sorted(combined_recs.items(), key=lambda x: x[1], reverse=True)
            product_ids = [pid for pid, _ in sorted_recs[:limit]]
            scores = [score for _, score in sorted_recs[:limit]]
        else:
            # Original user-based recommendations
            product_ids, scores = recommender.recommend_for_user(user_id, top_k=limit)
        
        if not product_ids or len(product_ids) == 0:
            logger.info(f"No recommendations generated for user {user_id}")
            return get_popular_products(limit)
        
        # Convert to strings for MongoDB query
        product_ids = [str(pid) for pid in product_ids]
        
        # Fetch product details from MongoDB
        products = list(mongo_instance.db.products.find({"_id": {"$in": product_ids}}))
        
        # Sort products by recommendation score
        product_score_map = {str(pid): score for pid, score in zip(product_ids, scores)}
        products.sort(key=lambda p: product_score_map.get(str(p["_id"]), 0), reverse=True)
        
        return products[:limit]
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return get_popular_products(limit)

def get_similar_products(product_id, limit=10):
    """
    Get similar products to a given product
    
    Args:
        product_id: Product ID to find similar items for
        limit: Maximum number of similar products to return
        
    Returns:
        List of similar products
    """
    global recommender, mongo_instance
    
    if recommender is None or mongo_instance is None:
        logger.error("Recommender or MongoDB not initialized")
        return []
    
    try:
        # Get raw similar products
        similar_ids, scores = recommender.find_similar_items(product_id, top_k=limit)
        
        if not similar_ids or len(similar_ids) == 0:
            return get_popular_products(limit)
        
        # Convert to strings for MongoDB query
        similar_ids = [str(sid) for sid in similar_ids]
        
        # Fetch product details from MongoDB
        products = list(mongo_instance.db.products.find({"_id": {"$in": similar_ids}}))
        
        # Sort products by similarity score
        product_score_map = {str(pid): score for pid, score in zip(similar_ids, scores)}
        products.sort(key=lambda p: product_score_map.get(str(p["_id"]), 0), reverse=True)
        
        return products[:limit]
    except Exception as e:
        logger.error(f"Error getting similar products: {e}")
        return get_popular_products(limit)

def get_popular_products(limit=10):
    """
    Get popular products based on ratings and interaction count
    
    Args:
        limit: Maximum number of popular products to return
        
    Returns:
        List of popular products
    """
    global mongo_instance
    
    if mongo_instance is None:
        logger.error("MongoDB not initialized")
        return []
    
    try:
        # Query products with highest ratings average and with at least 5 interactions
        pipeline = [
            {"$match": {"ratings_average": {"$exists": True, "$gt": 0}}},
            {"$sort": {"ratings_average": -1}},
            {"$limit": limit * 2}  # Get more than needed to ensure we have enough after filtering
        ]
        
        popular_products = list(mongo_instance.db.products.aggregate(pipeline))
        
        # Randomly select from top products to add variety
        if len(popular_products) > limit:
            import random
            random.shuffle(popular_products)
            popular_products = popular_products[:limit]
        
        return popular_products
    except Exception as e:
        logger.error(f"Error getting popular products: {e}")
        return []

class RecommendationEngine:
    """Recommendation engine combining multiple recommendation models with privacy"""
    
    def __init__(self, config, mongo=None, apply_privacy=True, epsilon=1.0, privacy_mechanism='laplace'):
        """
        Initialize recommendation engine
        
        Args:
            config: Application configuration
            mongo: MongoDB connection
            apply_privacy: Whether to apply privacy-preserving techniques
            epsilon: Privacy budget parameter (lower = more privacy)
            privacy_mechanism: Privacy mechanism to use ('laplace', 'gaussian', or 'randomized_response')
        """
        self.config = config
        
        # Initialize data processor with privacy settings
        self.data_processor = DataProcessor(
            config, 
            apply_privacy=apply_privacy, 
            epsilon=epsilon, 
            privacy_mechanism=privacy_mechanism
        )
        
        # Store privacy settings
        self.apply_privacy = apply_privacy
        self.epsilon = epsilon
        self.privacy_mechanism = privacy_mechanism
        
        if self.apply_privacy:
            logger.info(f"Initializing recommendation engine with privacy: {privacy_mechanism}, epsilon: {epsilon}")
        
        # Set MongoDB instance
        if mongo is not None:
            self.data_processor.mongo = mongo
        elif mongo_instance is not None:
            self.data_processor.mongo = mongo_instance
        
        # Initialize models - removed hybrid model
        self.models = {
            'ncf': None,
            'deepfm': None
        }
        
        # Initialize privacy wrappers if privacy is enabled
        self.privacy_wrappers = {
            'ncf': None,
            'deepfm': None
        }
        
        self.model_files = {
            'ncf': os.path.join(config.RECOMMENDER_MODELS_DIR, 'ncf_model.h5'),
            'deepfm': os.path.join(config.RECOMMENDER_MODELS_DIR, 'deepfm_model.h5')
        }
        
        # Load encoders if they exist
        encoders_path = os.path.join(config.RECOMMENDER_MODELS_DIR, 'encoders.pkl')
        if os.path.exists(encoders_path):
            self.data_processor.load_encoders(encoders_path)
        
        # Flag to track training status
        self.is_trained = False
    
    # Load method with privacy support
    def load(self):
        """
        Load pre-trained models from disk and wrap with privacy if enabled
        
        Returns:
            Boolean indicating if all models were loaded successfully
        """
        all_loaded = True
        
        try:
            # Load encoders
            encoders_path = os.path.join(self.config.RECOMMENDER_MODELS_DIR, 'encoders.pkl')
            if os.path.exists(encoders_path):
                success = self.data_processor.load_encoders(encoders_path)
                if not success:
                    return False
            else:
                # Initialize encoders
                from recommender.encoder_utils import load_encoders_and_initialize
                self.data_processor = load_encoders_and_initialize(self.config, mongo_instance)
                if self.data_processor is None:
                    logger.error("Failed to initialize encoders")
                    return False
            
            # Get custom objects for model loading
            from recommender.custom_objects import get_custom_objects, load_model_with_custom_objects
            
            # Check if models exist and load them with custom objects
            for model_name, model_path in self.model_files.items():
                if os.path.exists(model_path):
                    try:
                        # Load the model with custom objects
                        model = load_model_with_custom_objects(model_path)
                        
                        if model is not None:
                            self.models[model_name] = model
                            
                            # Wrap with privacy if enabled
                            if self.apply_privacy:
                                self.privacy_wrappers[model_name] = PrivacyAwareRecommender(
                                    model, 
                                    epsilon=self.epsilon,
                                    mechanism=self.privacy_mechanism
                                )
                                logger.info(f"Wrapped {model_name} model with privacy")
                            
                            logger.info(f"Loaded {model_name} model from {model_path}")
                        else:
                            # Special handling for NCF model which has serialization issues
                            if model_name == 'ncf':
                                logger.warning("Attempting to fix NCF model...")
                                from recommender.custom_objects import fix_ncf_model
                                fix_success = fix_ncf_model(model_path)
                                if fix_success:
                                    # Try loading again
                                    model = load_model_with_custom_objects(model_path)
                                    if model is not None:
                                        self.models[model_name] = model
                                        
                                        # Wrap with privacy if enabled
                                        if self.apply_privacy:
                                            self.privacy_wrappers[model_name] = PrivacyAwareRecommender(
                                                model, 
                                                epsilon=self.epsilon,
                                                mechanism=self.privacy_mechanism
                                            )
                                            logger.info(f"Wrapped fixed {model_name} model with privacy")
                                            
                                        logger.info(f"Successfully fixed and loaded {model_name} model")
                                    else:
                                        logger.error(f"Failed to load fixed {model_name} model")
                                        all_loaded = False
                                else:
                                    logger.error(f"Failed to fix {model_name} model")
                                    all_loaded = False
                            else:
                                logger.error(f"Failed to load {model_name} model")
                                all_loaded = False
                    except Exception as e:
                        logger.error(f"Error loading {model_name} model: {e}")
                        all_loaded = False
                else:
                    logger.warning(f"{model_name} model file not found at {model_path}")
                    all_loaded = False
            
            # Load model weights file if it exists
            weights_path = os.path.join(self.config.RECOMMENDER_MODELS_DIR, 'model_weights.pkl')
            if os.path.exists(weights_path):
                try:
                    with open(weights_path, 'rb') as f:
                        weights = pickle.load(f)
                        # Update weights to only include NCF and DeepFM
                        if 'hybrid' in weights:
                            hybrid_weight = weights.pop('hybrid', 0)
                            # Redistribute hybrid weight proportionally to NCF and DeepFM
                            total = weights['ncf'] + weights['deepfm']
                            if total > 0:
                                weights['ncf'] += hybrid_weight * (weights['ncf'] / total)
                                weights['deepfm'] += hybrid_weight * (weights['deepfm'] / total)
                            else:
                                # Equal distribution if both have weight 0
                                weights['ncf'] += hybrid_weight / 2
                                weights['deepfm'] += hybrid_weight / 2
                        self.model_weights = weights
                        logger.info(f"Loaded and adjusted model weights from {weights_path}")
                except Exception as e:
                    logger.error(f"Error loading model weights: {e}")
                    self.model_weights = {'ncf': 0.5, 'deepfm': 0.5}
            else:
                logger.info("No model weights file found, using equal weights")
                self.model_weights = {'ncf': 0.5, 'deepfm': 0.5}
            
            # Set trained flag
            self.is_trained = all_loaded
            
            return all_loaded
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Train method with privacy integration
    def train(self):
        """
        Train recommendation models with privacy protection
        
        Returns:
            Dictionary of training results
        """
        try:
            # Fetch and preprocess data with privacy
            logger.info("Fetching and preprocessing data with privacy")
            training_data = self.data_processor.prepare_training_data()
            
            if training_data is None or len(training_data) == 0:
                logger.error("No training data available")
                return {"success": False, "error": "No training data"}
            
            # Get data statistics
            n_users = self.data_processor.get_n_users()
            n_items = self.data_processor.get_n_items()
            
            logger.info(f"Training models with {len(training_data)} interactions, {n_users} users, {n_items} items")
            logger.info(f"Privacy enabled: {self.apply_privacy}, Mechanism: {self.privacy_mechanism}, Epsilon: {self.epsilon}")
            
            # Train NCF model
            logger.info("Training NCF model")
            ncf_model = NCFModel(
                n_users=n_users,
                n_items=n_items,
                embedding_size=self.config.NCF_FACTORS
            )
            
            # Apply privacy to NCF model if enabled
            if self.apply_privacy:
                logger.info("Wrapping NCF model with privacy")
                ncf_privacy_wrapper = PrivacyAwareRecommender(
                    ncf_model,
                    epsilon=self.epsilon/2,  # Split privacy budget between models - updated from /3 to /2
                    mechanism=self.privacy_mechanism
                )
                ncf_history = ncf_privacy_wrapper.train(training_data)
                self.models['ncf'] = ncf_model.model
                self.privacy_wrappers['ncf'] = ncf_privacy_wrapper
            else:
                ncf_history = ncf_model.train(training_data)
                self.models['ncf'] = ncf_model.model
            
            # Train DeepFM model
            logger.info("Training DeepFM model")
            deepfm_model = DeepFMModel(
                n_users=n_users,
                n_items=n_items,
                embedding_size=self.config.DEEPFM_EMBEDDING_SIZE
            )
            
            # Apply privacy to DeepFM model if enabled
            if self.apply_privacy:
                logger.info("Wrapping DeepFM model with privacy")
                deepfm_privacy_wrapper = PrivacyAwareRecommender(
                    deepfm_model,
                    epsilon=self.epsilon/2,  # Split privacy budget - updated from /3 to /2
                    mechanism=self.privacy_mechanism
                )
                deepfm_history = deepfm_privacy_wrapper.train(training_data)
                self.models['deepfm'] = deepfm_model.model
                self.privacy_wrappers['deepfm'] = deepfm_privacy_wrapper
            else:
                deepfm_history = deepfm_model.train(training_data)
                self.models['deepfm'] = deepfm_model.model
            
            # Removed hybrid model training
            
            # Evaluate models on a validation set
            validation_data = self.data_processor.prepare_validation_data()
            
            # Set model weights based on validation performance
            self.model_weights = self._compute_model_weights(validation_data)
            
            # Mark as trained
            self.is_trained = True
            
            return {
                "success": True, 
                "trained_users": n_users,
                "trained_items": n_items,
                "trained_interactions": len(training_data),
                "model_weights": self.model_weights,
                "privacy_applied": self.apply_privacy
            }
        except Exception as e:
            logger.error(f"Error training models: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    # Update recommend_for_user method to use privacy wrappers
    def recommend_for_user(self, user_id, top_k=10):
        """
        Generate recommendations for a user with privacy protection
        
        Args:
            user_id: User ID to generate recommendations for
            top_k: Number of recommendations to generate
            
        Returns:
            Tuple of (product_ids, scores)
        """
        if not self.is_trained:
            logger.warning("Models not trained, loading pre-trained models")
            if not self.load():
                logger.error("Failed to load pre-trained models")
                return [], []
        
        try:
            # Convert user ID to internal index
            user_idx = self.data_processor.encode_user(user_id)
            
            if user_idx is None:
                logger.warning(f"Unknown user ID: {user_id}")
                return [], []
            
            # Get all items the user hasn't interacted with
            interacted_items = self.data_processor.get_user_interactions(user_id)
            all_items = self.data_processor.get_all_items()
            
            # Items to consider for recommendations
            candidate_items = list(set(all_items) - set(interacted_items))
            
            if len(candidate_items) == 0:
                logger.warning(f"User {user_id} has interacted with all items")
                return [], []
            
            # Prepare input data for prediction
            n_candidates = len(candidate_items)
            user_input = np.array([user_idx] * n_candidates)
            item_input = np.array(self.data_processor.encode_items(candidate_items))
            
            # Add placeholder interaction features
            user_interaction = np.zeros((n_candidates, 1))
            item_interaction = np.zeros((n_candidates, 1))
            
            # Prepare model input
            X = [user_input, item_input, user_interaction, item_interaction]
            
            # Generate predictions from each model
            predictions = {}
            for model_name, model in self.models.items():
                if model is not None:
                    try:
                        # Use privacy wrapper if available and privacy is enabled
                        if self.apply_privacy and self.privacy_wrappers[model_name] is not None:
                            # For recommendation, we don't actually use the privacy wrapper's recommend_for_user
                            # method because it already assumes candidate items. Instead, we predict directly
                            # but add noise to the scores
                            y_pred = model.predict(X).flatten()
                            
                            # Apply Laplace/Gaussian noise to predictions
                            ldp = LocalDifferentialPrivacy(epsilon=self.epsilon/2, mechanism=self.privacy_mechanism)
                            y_pred = ldp.perturb_ratings(y_pred, rating_range=(y_pred.min(), y_pred.max()))
                        else:
                            y_pred = model.predict(X).flatten()
                            
                        predictions[model_name] = y_pred
                    except Exception as e:
                        logger.error(f"Error predicting with {model_name}: {e}")
            
            # Combine predictions using weighted average
            if len(predictions) == 0:
                logger.error("No models available for prediction")
                return [], []
            
            final_scores = np.zeros(n_candidates)
            for model_name, pred in predictions.items():
                weight = self.model_weights.get(model_name, 0.5)
                final_scores += weight * pred
            
            # Get top-k items
            top_indices = np.argsort(-final_scores)[:top_k]
            recommended_items = [candidate_items[i] for i in top_indices]
            recommended_scores = final_scores[top_indices]
            
            # Convert item indices back to product IDs
            recommended_product_ids = self.data_processor.decode_items(recommended_items)
            
            return recommended_product_ids, recommended_scores
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    # Update find_similar_items method to apply privacy to embeddings
    def find_similar_items(self, product_id, top_k=10):
        """
        Find similar products based on embeddings with privacy protection
        
        Args:
            product_id: Product ID to find similar items for
            top_k: Number of similar items to return
            
        Returns:
            Tuple of (similar_product_ids, similarity_scores)
        """
        if not self.is_trained:
            logger.warning("Models not trained, loading pre-trained models")
            if not self.load():
                logger.error("Failed to load pre-trained models")
                return [], []
        
        try:
            # Convert product ID to internal index
            item_idx = self.data_processor.encode_item(product_id)
            
            if item_idx is None:
                logger.warning(f"Unknown product ID: {product_id}")
                return [], []
            
            # Get all items except the target item
            all_items = self.data_processor.get_all_items()
            candidate_items = [i for i in all_items if i != item_idx]
            
            # Extract item embeddings from NCF model since hybrid is removed
            model_name = 'ncf'
            model = self.models[model_name]
            
            if model is None:
                # Try DeepFM as fallback
                model_name = 'deepfm'
                model = self.models[model_name]
                
            if model is None:
                logger.error("No models available for similarity calculation")
                return [], []
            
            # Extract item embeddings layer
            try:
                item_embedding_layer = None
                for layer in model.layers:
                    if 'item_embedding' in layer.name:
                        item_embedding_layer = layer
                        break
                
                if item_embedding_layer is None:
                    logger.error("Item embedding layer not found")
                    return [], []
                
                # Get target item embedding
                # Get target item embedding
                target_embedding = item_embedding_layer(np.array([item_idx])).numpy()
                
                # Get candidate item embeddings
                candidate_indices = np.array(candidate_items)
                candidate_embeddings = item_embedding_layer(candidate_indices).numpy()
                
                # Apply privacy to embeddings if enabled
                if self.apply_privacy:
                    logger.info("Applying privacy to item embeddings for similarity search")
                    # We don't apply privacy to the target embedding to maintain utility
                    # Apply privacy only to candidate embeddings
                    candidate_embeddings = self.data_processor.perturb_embeddings(candidate_embeddings)
                
                # Calculate cosine similarity
                target_norm = np.linalg.norm(target_embedding)
                candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
                
                dot_products = np.dot(candidate_embeddings, target_embedding.T).flatten()
                similarities = dot_products / (candidate_norms * target_norm + 1e-8)
                
                # Get top-k similar items
                top_indices = np.argsort(-similarities)[:top_k]
                similar_items = [candidate_items[i] for i in top_indices]
                similarity_scores = similarities[top_indices]
                
                # Convert item indices back to product IDs
                similar_product_ids = self.data_processor.decode_items(similar_items)
                
                return similar_product_ids, similarity_scores
            except Exception as e:
                logger.error(f"Error extracting embeddings: {e}")
                return [], []
        except Exception as e:
            logger.error(f"Error finding similar items: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    # Update _compute_model_weights method to handle privacy
    def _compute_model_weights(self, validation_data):
        """
        Compute weights for each model based on validation performance with privacy
        
        Args:
            validation_data: Validation dataset
            
        Returns:
            Dictionary of model weights
        """
        if validation_data is None or len(validation_data) == 0:
            logger.warning("No validation data available, using default weights")
            return {'ncf': 0.5, 'deepfm': 0.5}
        
        try:
            X_val = [
                validation_data['user_idx'].values,
                validation_data['item_idx'].values,
                validation_data['user_interaction_norm'].values.reshape(-1, 1),
                validation_data['item_interaction_norm'].values.reshape(-1, 1)
            ]
            y_val = validation_data['rating_scaled'].values
            
            # Calculate error for each model
            errors = {}
            for model_name, model in self.models.items():
                if model is not None:
                    # If privacy is enabled, predict with noise
                    if self.apply_privacy and self.privacy_wrappers[model_name] is not None:
                        # We can't directly use the privacy wrapper for prediction because it expects
                        # a different format, so we'll add noise to the predictions manually
                        y_pred = model.predict(X_val).flatten()
                        
                        # Only add a small amount of noise for validation to maintain accuracy
                        # of weight computation - using a higher epsilon (less noise)
                        ldp = LocalDifferentialPrivacy(epsilon=self.epsilon*2, mechanism=self.privacy_mechanism)
                        y_pred = ldp.perturb_ratings(y_pred, rating_range=(y_pred.min(), y_pred.max()))
                    else:
                        y_pred = model.predict(X_val).flatten()
                    
                    # Use mean absolute error
                    mae = np.mean(np.abs(y_pred - y_val))
                    errors[model_name] = mae
            
            # If any model is missing, use default weights
            if len(errors) < 2:  # Changed from 3 to 2 since we only have 2 models now
                logger.warning("Some models missing, using default weights")
                return {'ncf': 0.5, 'deepfm': 0.5}
            
            # Compute inverse error (lower error = higher weight)
            inv_errors = {m: 1.0/e for m, e in errors.items()}
            
            # Normalize to sum to 1
            total = sum(inv_errors.values())
            weights = {m: e/total for m, e in inv_errors.items()}
            
            logger.info(f"Computed model weights with privacy: {weights}")
            return weights
        except Exception as e:
            logger.error(f"Error computing model weights: {e}")
            return {'ncf': 0.5, 'deepfm': 0.5}
    
    # Update save method to save privacy settings
    def save(self):
        """Save trained models and privacy settings to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.config.RECOMMENDER_MODELS_DIR, exist_ok=True)
            
            # Save encoders
            encoders_path = os.path.join(self.config.RECOMMENDER_MODELS_DIR, 'encoders.pkl')
            self.data_processor.save_encoders(encoders_path)
            
            # Import custom object utilities
            from recommender.custom_objects import save_model_with_custom_objects
            
            # Save models
            for model_name, model in self.models.items():
                if model is not None:
                    model_path = self.model_files[model_name]
                    try:
                        # Use the custom saving function
                        success = save_model_with_custom_objects(model, model_path)
                        if success:
                            logger.info(f"Saved {model_name} model to {model_path}")
                        else:
                            logger.error(f"Failed to save {model_name} model")
                    except Exception as e:
                        logger.error(f"Error saving {model_name} model: {e}")
            
            # Save model weights
            weights_path = os.path.join(self.config.RECOMMENDER_MODELS_DIR, 'model_weights.pkl')
            with open(weights_path, 'wb') as f:
                pickle.dump(self.model_weights, f)
            
            # Save privacy settings
            privacy_settings_path = os.path.join(self.config.RECOMMENDER_MODELS_DIR, 'privacy_settings.pkl')
            with open(privacy_settings_path, 'wb') as f:
                privacy_settings = {
                    'apply_privacy': self.apply_privacy,
                    'epsilon': self.epsilon,
                    'privacy_mechanism': self.privacy_mechanism
                }
                pickle.dump(privacy_settings, f)
                logger.info(f"Saved privacy settings to {privacy_settings_path}")
            
            logger.info("All models and settings saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False