# models/deepfm_model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout, BatchNormalization, Dot, Add, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('recommender.deepfm_model')

class DeepFMModel:
    """DeepFM Recommendation Model"""
    
    def __init__(self, n_users, n_items, embedding_size=64):
        """
        Initialize DeepFM model
        
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
        Build the DeepFM model architecture with best values from image:
        - Activation: sigmoid
        - Loss Function: huber
        - Optimizer: RMSprop
        
        Returns:
            Compiled Keras model
        """
        # Keep the same input interface but modify the internal architecture
        user_input = Input(shape=(1,), name='user_input')
        item_input = Input(shape=(1,), name='item_input')
        user_interaction_input = Input(shape=(1,), name='user_interaction')
        item_interaction_input = Input(shape=(1,), name='item_interaction')
        
        # Embedding layers for FM component
        user_embedding = Embedding(
            input_dim=self.n_users, 
            output_dim=self.embedding_size,
            embeddings_regularizer=l2(1e-4),
            name='user_embedding'
        )(user_input)
        
        item_embedding = Embedding(
            input_dim=self.n_items, 
            output_dim=self.embedding_size,
            embeddings_regularizer=l2(1e-4),
            name='item_embedding'
        )(item_input)
        
        # Bias terms for FM component
        user_bias = Embedding(self.n_users, 1, embeddings_regularizer=l2(1e-4), name='user_bias')(user_input)
        item_bias = Embedding(self.n_items, 1, embeddings_regularizer=l2(1e-4), name='item_bias')(item_input)
        
        # Flatten embeddings
        user_latent = Flatten()(user_embedding)
        item_latent = Flatten()(item_embedding)
        
        # FM Component: Dot product for pairwise interaction
        fm_interaction = Dot(axes=1, name='fm_interaction')([user_latent, item_latent])
        
        # Deep Component - modified to match pasted code
        # Instead of using all 4 inputs, just use user and item embeddings
        deep_input = Concatenate(name='deep_input')([
            user_latent, 
            item_latent
            # Not using interaction inputs anymore
        ])
        
        # First dense layer with BatchNorm and Sigmoid activation
        deep = Dense(256, kernel_regularizer=l2(1e-4))(deep_input)
        deep = BatchNormalization()(deep)
        deep = Activation('sigmoid')(deep)  # Using sigmoid activation as per best values
        deep = Dropout(0.5)(deep)
        
        # Second dense layer with BatchNorm and Sigmoid activation
        deep = Dense(128, kernel_regularizer=l2(1e-4))(deep)
        deep = BatchNormalization()(deep)
        deep = Activation('sigmoid')(deep)  # Using sigmoid activation as per best values
        deep = Dropout(0.5)(deep)
        
        # Combine FM and Deep components like in the pasted code
        combined = Concatenate(name='combined_features')([fm_interaction, deep])
        output = Dense(1, activation='linear', name='prediction')(combined)
        
        # Add bias terms
        output = Add(name='add_bias')([output, Flatten()(user_bias), Flatten()(item_bias)])
        
        # Create and compile model - keeping the original inputs to maintain interface compatibility
        model = Model(
            inputs=[user_input, item_input, user_interaction_input, item_interaction_input], 
            outputs=output
        )
        
        # Compile model with Huber loss and RMSprop optimizer as per best values
        model.compile(
            optimizer=RMSprop(learning_rate=0.0001),
            loss=Huber(delta=0.8),
            metrics=['mae']
        )
        
        return model
    
    def train(self, df, epochs=250, validation_split=0.1, batch_size=10100):
        """
        Train the DeepFM model with 250 epochs as per best values
        
        Args:
            df: DataFrame with processed interaction data
            epochs: Maximum number of training epochs (250 based on best values)
            validation_split: Portion of data to use for validation
            batch_size: Batch size for training (10100 from original code)
            
        Returns:
            Training history
        """
        try:
            # Keep the original input preparation to maintain compatibility
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
                patience=19,
                restore_best_weights=True,
                min_delta=1e-5
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.8,
                patience=5,
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
            logger.error(f"Error training DeepFM model: {e}")
            import traceback
            traceback.print_exc()
            return None