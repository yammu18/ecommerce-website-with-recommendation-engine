import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('recommender_evaluation')

# Import project modules
from models.data_processor import DataProcessor
from models.ncf_model import NCFModel
from models.deepfm_model import DeepFMModel
from recommender.privacy import LocalDifferentialPrivacy
from config import config

# Hardcoded configuration
DATA_PATH = "data/raw/reduced_data.csv"  # Path to your data file
OUTPUT_DIR = "evaluation_results"  # Directory to store results
MODELS_TO_EVALUATE = ["NCF", "DeepFM"]  # Models to evaluate
USE_TEMPORAL_SPLIT = True  # Use temporal split instead of random split

class ModelEvaluator:
    """Evaluates recommendation models using MSE, RMSE, MAE, and privacy metrics"""
    
    def __init__(self, output_dir=OUTPUT_DIR, 
                 epsilon_values=[0.1, 1.0, 10.0], 
                 privacy_mechanisms=['laplace', 'gaussian']):
        """
        Initialize the evaluator with privacy evaluation settings
        
        Args:
            output_dir: Directory to save evaluation results
            epsilon_values: Privacy budget values to test
            privacy_mechanisms: Privacy mechanisms to evaluate
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.epsilon_values = epsilon_values
        self.privacy_mechanisms = privacy_mechanisms
    
    def load_data(self, interactions_file=DATA_PATH):
        """
        Load data from CSV file
        
        Args:
            interactions_file: Path to interactions CSV
            
        Returns:
            DataFrame with interactions
        """
        try:
            # Ensure file exists
            if not os.path.exists(interactions_file):
                logger.error(f"File not found: {interactions_file}")
                return None
            
            # Read CSV with appropriate settings
            df = pd.read_csv(
                interactions_file, 
                skiprows=1,  # Skip header if present
                names=['user_id', 'product_id', 'rating', 'timestamp'],
                dtype={'user_id': str, 'product_id': str, 'rating': float, 'timestamp': float}
            )
            
            logger.info(f"Loaded {len(df)} interactions from {interactions_file}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_data(self, df, test_size=0.2, random_state=42, temporal_split=USE_TEMPORAL_SPLIT):
        """
        Preprocess data and create train/test split
        
        Args:
            df: DataFrame with interactions
            test_size: Ratio of test set
            random_state: Random state for reproducibility
            temporal_split: Whether to split based on time
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Make a copy to avoid modifications to original
        df = df.copy()
        
        # Handle missing values
        df = df.dropna(subset=['user_id', 'product_id', 'rating'])
        
        # Ensure user_id and product_id are strings and rating is float
        df['user_id'] = df['user_id'].astype(str)
        df['product_id'] = df['product_id'].astype(str)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Drop any rows where conversion to float failed
        df = df.dropna(subset=['rating'])
        
        # Split data
        if temporal_split and 'timestamp' in df.columns:
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Take last test_size% as test set
            train_size = int(len(df) * (1 - test_size))
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            logger.info(f"Created temporal train/test split: {len(train_df)} train, {len(test_df)} test")
        else:
            # Random split
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
            logger.info(f"Created random train/test split: {len(train_df)} train, {len(test_df)} test")
        
        return train_df, test_df
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Calculate basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate MAPE with error handling
        try:
            non_zero_mask = (y_true != 0)
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            else:
                mape = np.nan
        except Exception:
            mape = np.nan
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape) if not np.isnan(mape) else None
        }
    
    def apply_privacy(self, y_pred, y_true, mechanism, epsilon):
        """
        Apply privacy mechanism to predictions
        
        Args:
            y_pred: Original predictions
            y_true: True ratings (for range determination)
            mechanism: Privacy mechanism to use
            epsilon: Privacy budget
            
        Returns:
            Perturbed predictions
        """
        # Create LDP instance
        ldp = LocalDifferentialPrivacy(epsilon=epsilon, mechanism=mechanism)
        
        # Determine rating range
        rating_min = y_true.min()
        rating_max = y_true.max()
        
        # Apply privacy mechanism
        return ldp.perturb_ratings(y_pred, rating_range=(rating_min, rating_max))
    
    def evaluate_model(self, model, train_df, test_df, data_processor):
        """
        Evaluate a model with privacy considerations
        
        Args:
            model: The model to evaluate
            train_df: Training data
            test_df: Test data
            data_processor: Data processor instance
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Process test data
        test_processed = data_processor.transform(test_df)
        
        # Prepare test data for prediction
        X_test = [
            test_processed['user_idx'].values,
            test_processed['item_idx'].values,
            test_processed['user_interaction_norm'].values.reshape(-1, 1),
            test_processed['item_interaction_norm'].values.reshape(-1, 1)
        ]
        y_true = test_processed['rating'].values
        
        # Predict ratings
        y_pred_scaled = model.model.predict(X_test).flatten()
        y_pred = np.array([data_processor.inverse_scale_rating(pred) for pred in y_pred_scaled])
        
        # Clip predictions to valid rating range
        rating_min = test_df['rating'].min()
        rating_max = test_df['rating'].max()
        y_pred = np.clip(y_pred, rating_min, rating_max)
        
        # Calculate original metrics
        original_metrics = self.calculate_metrics(y_true, y_pred)
        
        # Privacy evaluation results
        privacy_results = {'original': original_metrics}
        
        # Evaluate with different privacy mechanisms and epsilon values
        for mechanism in self.privacy_mechanisms:
            for epsilon in self.epsilon_values:
                try:
                    # Apply privacy mechanism
                    y_pred_private = self.apply_privacy(y_pred, y_true, mechanism, epsilon)
                    
                    # Calculate metrics for private predictions
                    privacy_key = f"{mechanism}_ε{epsilon}"
                    privacy_results[privacy_key] = self.calculate_metrics(y_true, y_pred_private)
                    
                    # Log privacy impact
                    logger.info(f"Privacy evaluation for {mechanism} (ε={epsilon}):")
                    logger.info(f"  MSE: {privacy_results[privacy_key]['mse']:.4f}")
                    logger.info(f"  RMSE: {privacy_results[privacy_key]['rmse']:.4f}")
                    logger.info(f"  MAE: {privacy_results[privacy_key]['mae']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error in privacy evaluation for {mechanism} (ε={epsilon}): {e}")
                    privacy_results[privacy_key] = None
        
        return privacy_results
    
    # Update this section in evaluator.py

    def evaluate_all_models(self, train_df, test_df, models_to_evaluate=MODELS_TO_EVALUATE):
        """
        Evaluate all specified models
        
        Args:
            train_df: Training data
            test_df: Test data
            models_to_evaluate: List of model names to evaluate
            
        Returns:
            Dictionary with results for each model
        """
        results = {}
        
        # Process data once
        data_processor = DataProcessor(config, min_interactions=3)
        train_processed = data_processor.fit_transform(train_df)
        
        # Train and evaluate each model
        for model_name in models_to_evaluate:
            try:
                logger.info(f"Training and evaluating {model_name} model")
                
                # Initialize the appropriate model
                if model_name == 'NCF':
                    model = NCFModel(
                        n_users=data_processor.get_n_users(),
                        n_items=data_processor.get_n_items()
                    )
                elif model_name == 'DeepFM':
                    model = DeepFMModel(
                        n_users=data_processor.get_n_users(),
                        n_items=data_processor.get_n_items()
                    )
                else:
                    logger.warning(f"Unknown model {model_name}, skipping")
                    continue
                
                # Train model
                model.train(train_processed)
                
                # Evaluate model with privacy considerations
                model_results = self.evaluate_model(
                    model, train_df, test_df, data_processor
                )
                
                # Store results
                results[model_name] = model_results
                
                # Log original model performance
                original_metrics = model_results['original']
                logger.info(f"{model_name} original performance:")
                logger.info(f"  MSE: {original_metrics['mse']:.4f}")
                logger.info(f"  RMSE: {original_metrics['rmse']:.4f}")
                logger.info(f"  MAE: {original_metrics['mae']:.4f}")
                logger.info(f"  MAPE: {original_metrics.get('mape', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return results

def main():
    """Main function to run the evaluation"""
    # Create evaluator
    evaluator = ModelEvaluator(
        output_dir=OUTPUT_DIR, 
        epsilon_values=[0.1, 1.0, 10.0],
        privacy_mechanisms=['laplace', 'gaussian']
    )
    
    # Load data
    df = evaluator.load_data(DATA_PATH)
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Preprocess data and create train/test split
    train_df, test_df = evaluator.preprocess_data(df, temporal_split=USE_TEMPORAL_SPLIT)
    
    # Evaluate models
    results = evaluator.evaluate_all_models(train_df, test_df)
    
    # Save results to JSON
    with open(os.path.join(OUTPUT_DIR, 'privacy_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\n===== PRIVACY EVALUATION SUMMARY =====")
    for model_name, privacy_results in results.items():
        print(f"\n{model_name} Model Performance:")
        
        # Print original (non-private) metrics
        original = privacy_results['original']
        print("  Original Performance:")
        print(f"    MSE:  {original['mse']:.4f}")
        print(f"    RMSE: {original['rmse']:.4f}")
        print(f"    MAE:  {original['mae']:.4f}")
        
        # Print privacy mechanism results
        for key, metrics in privacy_results.items():
            if key != 'original' and metrics is not None:
                print(f"  {key} Performance:")
                print(f"    MSE:  {metrics['mse']:.4f}")
                print(f"    RMSE: {metrics['rmse']:.4f}")
                print(f"    MAE:  {metrics['mae']:.4f}")

if __name__ == "__main__":
    main()