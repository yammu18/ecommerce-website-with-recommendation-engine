import os
from datetime import timedelta
import tensorflow as tf

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://localhost:27017/ecommerce'
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
    
    # TensorFlow settings
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Simplified eager execution
    tf.config.run_functions_eagerly(True)
    
    # Recommendation engine settings
    RECOMMENDER_MODELS_DIR = os.path.join(os.getcwd(), 'data', 'models')
    RECOMMENDER_BATCH_SIZE = 100
    RECOMMENDER_TOP_K = 10

    # Model training settings
    NCF_FACTORS = 64
    NCF_LAYERS = [128, 64, 32]
    DEEPFM_EMBEDDING_SIZE = 64
    VAE_LATENT_DIM = 32
    
    # Parallel processing
    NUM_WORKERS = 4
    
    # Privacy settings
    APPLY_PRIVACY = True  # Enable privacy by default
    PRIVACY_EPSILON = 1.0  # Privacy budget (lower = more privacy, less accuracy)
    PRIVACY_MECHANISM = 'laplace'  # Options: 'laplace', 'gaussian', 'randomized_response'
    PRIVACY_DELTA = 1e-5  # Delta parameter for Gaussian mechanism

# Use a single configuration
config = Config()