# models/ncf_model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout, BatchNormalization, Layer, Dot, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
# At the top of ncf_model.py, add this import:
from tensorflow.keras.losses import MeanSquaredError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('recommender.ncf_model')

class ElementWiseMultiplyLayer(Layer):
    """Custom layer for element-wise multiplication"""
    def __init__(self, **kwargs):
        super(ElementWiseMultiplyLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        """Apply element-wise multiplication"""
        return inputs[0] * inputs[1]

class NCFModel:
    """Neural Collaborative Filtering (NCF) Model"""
    
    def __init__(self, n_users, n_items, embedding_size=200):
        """
        Initialize NCF model
        
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_size: Size of embedding vectors
        """
        self.n_users = max(1, n_users)  # Ensure at least 1 user
        self.n_items = max(1, n_items)  # Ensure at least 1 item
        self.embedding_size = embedding_size
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the NCF model architecture
        
        Returns:
            Compiled Keras model
        """
        # Inputs
        user_input = Input(shape=(1,), name='user_input')
        item_input = Input(shape=(1,), name='item_input')
        user_interaction_input = Input(shape=(1,), name='user_interaction')
        item_interaction_input = Input(shape=(1,), name='item_interaction')
        
        # Embedding layers
        user_embedding = Embedding(
            input_dim=self.n_users, 
            output_dim=self.embedding_size,
            embeddings_regularizer=l2(0.01),
            name='user_embedding'
        )(user_input)
        
        item_embedding = Embedding(
            input_dim=self.n_items, 
            output_dim=self.embedding_size,
            embeddings_regularizer=l2(0.01),
            name='item_embedding'
        )(item_input)
        
        # Flatten embedding vectors
        user_latent = Flatten()(user_embedding)
        item_latent = Flatten()(item_embedding)
        
        # NCF: Element-wise multiplication of user and item embeddings
        element_product = ElementWiseMultiplyLayer()([user_latent, item_latent])
        
        # Concatenate all features
        concat = Concatenate()([
            user_latent, 
            item_latent, 
            element_product, 
            user_interaction_input, 
            item_interaction_input
        ])
        
        # Multi-layer perceptron
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(concat)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        
        # Output layer
        output = Dense(1, activation='linear')(x)
        
        # Create and compile model
        model = Model(
            inputs=[user_input, item_input, user_interaction_input, item_interaction_input], 
            outputs=output
        )
        
        # Compile model
        model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss=MeanSquaredError(),
            metrics=['mae']
        )
        
        return model
    
    def train(self, df, epochs=150, validation_split=0.1, batch_size=1024):
        """
        Train the NCF model
        
        Args:
            df: DataFrame with processed interaction data
            epochs: Maximum number of training epochs
            validation_split: Portion of data to use for validation
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        try:
            # Prepare training data
            X = [
                df['user_idx'].values, 
                df['item_idx'].values,
                df['user_interaction_norm'].values.reshape(-1, 1),
                df['item_interaction_norm'].values.reshape(-1, 1)
            ]
            y = df['rating_scaled'].values
            
            # Callbacks for early stopping and learning rate reduction
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=5,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
            
            # Train the model
            history = self.model.fit(
                X, y, 
                epochs=epochs, 
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            return history
        except Exception as e:
            logger.error(f"Error training NCF model: {e}")
            import traceback
            traceback.print_exc()
            return None