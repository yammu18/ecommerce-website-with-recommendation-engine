# recommender/custom_objects.py
import tensorflow as tf
from tensorflow.keras.layers import Layer
import os
class ElementWiseMultiplyLayer(Layer):
    """Custom layer for element-wise multiplication"""
    def __init__(self, **kwargs):
        super(ElementWiseMultiplyLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        """Apply element-wise multiplication"""
        # Make sure to unpack the inputs properly
        user_latent, item_latent = inputs
        return user_latent * item_latent
    
    def get_config(self):
        """Get layer configuration for serialization"""
        return super(ElementWiseMultiplyLayer, self).get_config()
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration"""
        return cls(**config)

def get_custom_objects():
    """
    Get a dictionary of custom objects for model loading/saving
    
    Returns:
        Dictionary of custom objects
    """
    # Create dictionary of custom objects
    from tensorflow.keras.losses import MeanSquaredError
    
    custom_objects = {
        'ElementWiseMultiplyLayer': ElementWiseMultiplyLayer,
        'MeanSquaredError': MeanSquaredError,
    }
    
    return custom_objects

def save_model_with_custom_objects(model, filepath):
    """
    Save a model with custom objects
    
    Args:
        model: Keras model to save
        filepath: Path to save the model
        
    Returns:
        Boolean indicating success
    """
    try:
        custom_objects = get_custom_objects()
        
        # This is important for TF 2.18+
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        # This can help in some cases to reset any cached state
        tf.keras.backend.clear_session()
        
        with tf.keras.utils.custom_object_scope(custom_objects):
            model.save(filepath, save_format='h5')
        
        return True
    except Exception as e:
        import logging
        logger = logging.getLogger('recommender.custom_objects')
        logger.error(f"Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_model_with_custom_objects(filepath):
    """
    Load a model with custom objects
    
    Args:
        filepath: Path to load the model from
        
    Returns:
        Loaded model or None if loading failed
    """
    try:
        custom_objects = get_custom_objects()
        
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(filepath)
        
        return model
    except Exception as e:
        import logging
        logger = logging.getLogger('recommender.custom_objects')
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def fix_ncf_model(model_path):
    """
    Fix the NCF model by rebuilding and saving it with correct loss function
    
    Args:
        model_path: Path to the NCF model file
        
    Returns:
        Boolean indicating success
    """
    try:
        import os
        import logging
        from models.data_processor import DataProcessor
        from models.ncf_model import NCFModel
        from recommender.engine import recommender
        
        logger = logging.getLogger('recommender.custom_objects')
        
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"NCF model not found at {model_path}")
            return False
            
        # Get dimensions from recommender if available
        if recommender and recommender.data_processor:
            n_users = recommender.data_processor.get_n_users()
            n_items = recommender.data_processor.get_n_items()
        else:
            # Default values
            logger.warning("Using default values for users and items")
            n_users = 1540  # From your logs
            n_items = 5689  # From your logs
            
        # Create a new NCF model with the same dimensions
        logger.info(f"Rebuilding NCF model with {n_users} users and {n_items} items")
        ncf_model = NCFModel(n_users, n_items)
        
        # Save the model with proper serialization
        return save_model_with_custom_objects(ncf_model.model, model_path)
    except Exception as e:
        import logging
        logger = logging.getLogger('recommender.custom_objects')
        logger.error(f"Error fixing NCF model: {e}")
        import traceback
        traceback.print_exc()
        return False