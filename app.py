from flask import Flask, render_template, session, g, jsonify, request
from flask_pymongo import PyMongo
from datetime import datetime
from bson.objectid import ObjectId
import os
from flask_bcrypt import Bcrypt
from services.cart_service import get_cart
# Import controllers
from controllers.auth_controller import auth_bp
from controllers.product_controller import product_bp
from controllers.user_controller import user_bp
from controllers.cart_controller import cart_bp
from controllers.order_controller import order_bp
from recommender.engine import init_recommender, train_models, schedule_training
# Import services
from services.auth_service import init_auth_service
from services.product_service import init_product_service
from services.user_service import init_user_service
from services.cart_service import init_cart_service
from services.order_service import init_order_service
from utils.helpers import get_product_categories
from flask import g, session
# Import database
from datetime import datetime, timedelta
from database.db import init_db

# Import configuration
from config import config

# Define the fix_recommender_initialization function before create_app
# Update the fix_recommender_initialization function to force training
from recommender.engine import get_recommendations, get_similar_products, get_popular_products, train_models, force_train_models
import os

def create_error_templates():
    """Create error template files if they don't exist"""
    templates_dir = os.path.join(os.getcwd(), 'templates')
    errors_dir = os.path.join(templates_dir, 'errors')
    
    # Create directories if they don't exist
    os.makedirs(errors_dir, exist_ok=True)
    
    # Create 404 template if it doesn't exist
    not_found_path = os.path.join(errors_dir, '404.html')
    if not os.path.exists(not_found_path):
        with open(not_found_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>404 - Page Not Found</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        h1 { color: #333; }
        p { color: #666; }
        .container { max-width: 600px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>404 - Page Not Found</h1>
        <p>The page you are looking for does not exist.</p>
        <p><a href="/">Go back to home page</a></p>
    </div>
</body>
</html>
""")
        print(f"Created 404 template at {not_found_path}")
    
    # Create 500 template if it doesn't exist
    server_error_path = os.path.join(errors_dir, '500.html')
    if not os.path.exists(server_error_path):
        with open(server_error_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>500 - Server Error</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        h1 { color: #333; }
        p { color: #666; }
        .container { max-width: 600px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>500 - Server Error</h1>
        <p>Something went wrong on our end. Please try again later.</p>
        <p><a href="/">Go back to home page</a></p>
    </div>
</body>
</html>
""")
        print(f"Created 500 template at {server_error_path}")

# Create privacy template directories
def create_privacy_templates():
    """Create privacy template directories if they don't exist"""
    templates_dir = os.path.join(os.getcwd(), 'templates')
    privacy_dir = os.path.join(templates_dir, 'privacy')
    admin_dir = os.path.join(templates_dir, 'admin')
    
    # Create directories if they don't exist
    os.makedirs(privacy_dir, exist_ok=True)
    os.makedirs(admin_dir, exist_ok=True)
    
    print(f"Created privacy template directories: {privacy_dir}, {admin_dir}")

# Call this function in app.py before create_app()
# Update the fix_recommender_initialization function to ensure proper dimensionality
def fix_recommender_initialization(app, mongo):
    """
    Ensure the recommender system is properly initialized with MongoDB instance
    and correctly loads or trains models with consistent dimensions.
    Also ensures encoders are properly initialized.
    """
    from recommender.engine import init_recommender, recommender, force_train_models
    from recommender.encoder_utils import load_encoders_and_initialize
    
    # Create encoder_utils.py file if it doesn't exist
    encoder_utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'recommender', 'encoder_utils.py')
    if not os.path.exists(encoder_utils_path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(encoder_utils_path), exist_ok=True)
        
        # Write encoder utils function to file
        with open(encoder_utils_path, 'w') as f:
            f.write("""
def load_encoders_and_initialize(app_config, mongo_instance=None):
    \"\"\"
    Explicitly load encoders and initialize them with all users and items from database
    
    Args:
        app_config: Application configuration
        mongo_instance: MongoDB connection
    \"\"\"
    import os
    from models.data_processor import DataProcessor
    import pandas as pd
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('recommender.encoder_utils')
    
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
                    from database.db import mongo
                    interaction_count = mongo.db.interactions.count_documents({})
                    user_count = mongo.db.users.count_documents({})
                    product_count = mongo.db.products.count_documents({})
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
""")
        print(f"Created {encoder_utils_path}")
    
    # Reinitialize the recommender with the mongo instance and proper encoders
    print("Reinitializing recommender with MongoDB instance...")
    
    # Initialize with privacy settings
    init_recommender(
        app.config, 
        mongo=mongo,
        apply_privacy=app.config.get('APPLY_PRIVACY', True),
        epsilon=app.config.get('PRIVACY_EPSILON', 1.0),
        privacy_mechanism=app.config.get('PRIVACY_MECHANISM', 'laplace')
    )
    
    # Explicitly verify encoders are initialized
    if not recommender or not hasattr(recommender, 'data_processor'):
        print("Recommender not properly initialized. Forcing initialization...")
        init_recommender(app.config, mongo)
    
    if not recommender or not hasattr(recommender, 'data_processor'):
        print("Failed to initialize recommender again. Skipping further steps.")
        return
    
    # Check if encoders are properly initialized
    if not hasattr(recommender.data_processor.user_encoder, 'classes_') or not hasattr(recommender.data_processor.item_encoder, 'classes_'):
        print("Encoders not properly initialized. Reinitializing...")
        recommender.data_processor = load_encoders_and_initialize(app.config, mongo)
    
    # Check if models exist
    models_dir = app.config['RECOMMENDER_MODELS_DIR']
    os.makedirs(models_dir, exist_ok=True)
    
    # Force training if encoders were just initialized
    if recommender.data_processor and (not hasattr(recommender.data_processor.user_encoder, 'classes_') or len(recommender.data_processor.user_encoder.classes_) == 0):
        print("Encoders still not initialized. Forcing model training...")
        try:
            result = force_train_models()
            print(f"Training result: {result}")
        except Exception as e:
            print(f"Error during forced model training: {e}")
            import traceback
            traceback.print_exc()
    
    # Check the database for at least one product and user
    product_count = mongo.db.products.count_documents({})
    user_count = mongo.db.users.count_documents({})
    
    if product_count == 0 or user_count == 0:
        print("Warning: No products or users found in database. Import sample data.")
    
    # Check if recommendation engine is properly initialized
    if recommender and recommender.data_processor:
        if hasattr(recommender.data_processor.user_encoder, 'classes_') and hasattr(recommender.data_processor.item_encoder, 'classes_'):
            print(f"Recommendation engine initialized with {len(recommender.data_processor.user_encoder.classes_)} users and {len(recommender.data_processor.item_encoder.classes_)} items")
            print("Loading recommender models...")
            loaded = recommender.load()
            print(f"Models loaded successfully: {loaded}")
        else:
            print("Recommender encoders not properly initialized.")
    else:
        print("Recommender not initialized properly.")

# Add to the top of create_app() function
create_error_templates()
create_privacy_templates()
# app.py (updated initialization part)

def create_app():
    """
    Create and configure the Flask application
    
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Initialize extensions
    mongo = PyMongo(app)
    bcrypt = Bcrypt(app)
    
    # Initialize database
    init_db(app, mongo)
    
    # Initialize services
    init_auth_service(app, mongo, bcrypt)
    init_product_service(app, mongo)
    init_user_service(app, mongo)
    init_cart_service(app, mongo)
    init_order_service(app, mongo)
    
    # Initialize recommendation engine with privacy settings
    from recommender.engine import init_recommender
    init_recommender(
        app.config, 
        mongo=mongo,
        apply_privacy=app.config.get('APPLY_PRIVACY', True),
        epsilon=app.config.get('PRIVACY_EPSILON', 1.0),
        privacy_mechanism=app.config.get('PRIVACY_MECHANISM', 'laplace')
    )
    
    # Fix recommender initialization if needed
    fix_recommender_initialization(app, mongo)
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(product_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(cart_bp)
    app.register_blueprint(order_bp)
    
    # Import data if needed
    
    @app.before_request
    def load_logged_in_user():
        user_id = session.get('user_id')
        if user_id is None:
            g.user = None
        else:
            try:
                from bson.objectid import ObjectId
                # Convert string ID back to ObjectId
                user_id_obj = ObjectId(user_id)
                g.user = mongo.db.users.find_one({"_id": user_id_obj})
            except Exception as e:
                print(f"Error loading user: {e}")
                g.user = None
                session.clear()  # Clear invalid session
                
    @app.context_processor
    def utility_processor():
        """
        Add utility functions to all templates
        """
        # Import functions to get cart and product categories
        from services.cart_service import get_cart
        from utils.helpers import get_product_categories
        
        return {
            'get_product_categories': get_product_categories,
            'get_cart': get_cart,
        }
    
    # Home route with recommendations
    @app.route('/')
    def home():
        # Get featured products and recommendations if user is logged in
        featured_products = list(mongo.db.products.find().limit(8))
        recommended_products = []
        
        if g.user:
            from recommender.engine import get_recommendations
            user_id = str(g.user['_id'])
            recommended_products = get_recommendations(user_id, limit=4)
        
        return render_template(
            'home.html',
            featured_products=featured_products,
            recommended_products=recommended_products
        )
        
    @app.template_filter('currency')
    def currency_filter(value):
        """Format a value as currency"""
        if value is None:
            return "$0.00"
        return f"${float(value):.2f}"
    
    @app.template_filter('format_date')
    def format_date_filter(date, format_string='%Y-%m-%d'):
        """Format a date according to the given format"""
        if date is None:
            return ""
        if isinstance(date, str):
            try:
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            except ValueError:
                return date
        return date.strftime(format_string)
        
    # Add this endpoint to manually train the models
    @app.route('/admin/train-models')
    def admin_train_models():
        try:
            print("Manually triggering model training...")
            from recommender.engine import train_models
            result = train_models()
            return jsonify({"success": True, "result": result})
        except Exception as e:
            print(f"Error training models: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "error": str(e)})
    
    # Add recommendations API endpoint
    @app.route('/api/recommendations')
    def api_recommendations():
        """Get recommendations for the current user or generic popular products"""
        limit = request.args.get('limit', 5, type=int)
        
        if g.user:
            from recommender.engine import get_recommendations
            user_id = str(g.user['_id'])
            products = get_recommendations(user_id, limit=limit)
        else:
            from recommender.engine import get_popular_products
            products = get_popular_products(limit=limit)
            
        # Convert ObjectId fields to string for JSON serialization
        for product in products:
            if isinstance(product.get('_id'), ObjectId):
                product['_id'] = str(product['_id'])
        
        return jsonify(products)
    
    # API endpoint for similar products
    @app.route('/api/similar-products/<product_id>')
    def api_similar_products(product_id):
        """Get similar products to a specific product"""
        limit = request.args.get('limit', 5, type=int)
        
        from recommender.engine import get_similar_products
        similar_products = get_similar_products(product_id, limit=limit)
        
        return jsonify(similar_products)
    
    @app.route('/direct-train-models')
        # Update this section in app.py

    def direct_train_models():
        """Directly train models with fixed implementations"""
        try:
            # Create models directory
            models_dir = os.path.join(os.getcwd(), 'data', 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Create a DataProcessor
            from models.data_processor import DataProcessor
            processor = DataProcessor(app.config)
            processor.mongo = mongo
            
            # Import custom object utilities
            from recommender.custom_objects import save_model_with_custom_objects
            
            # Get data
            print("Fetching data...")
            interaction_data = processor.get_interaction_data()
            if interaction_data is None or len(interaction_data) == 0:
                return jsonify({"success": False, "error": "No interaction data available"})
                    
            print(f"Processing {len(interaction_data)} interactions...")
            processed_data = processor.fit_transform(interaction_data)
            
            # Save encoders first
            print("Saving encoders...")
            encoders_path = os.path.join(models_dir, 'encoders.pkl')
            processor.save_encoders(encoders_path)
            
            # Get dimensions
            n_users = processor.get_n_users()
            n_items = processor.get_n_items()
            print(f"Training with {n_users} users and {n_items} items")
            
            # Train one model at a time to avoid memory issues
            results = {}
            
            try:
                # Train NCF model
                print("\n======= Training NCF model =======")
                from models.ncf_model import NCFModel
                ncf_model = NCFModel(n_users, n_items)
                ncf_history = ncf_model.train(processed_data, epochs=3)  # Fewer epochs for testing
                
                # Save NCF model with custom objects
                model_path = os.path.join(models_dir, 'ncf_model.h5')
                save_result = save_model_with_custom_objects(ncf_model.model, model_path)
                print(f"NCF model saved to {model_path}, result: {save_result}")
                results['ncf'] = "Success" if save_result else "Save failed"
            except Exception as e:
                print(f"Error training NCF model: {e}")
                import traceback
                traceback.print_exc()
                results['ncf'] = f"Failed: {str(e)}"
            
            try:
                # Train DeepFM model
                print("\n======= Training DeepFM model =======")
                from models.deepfm_model import DeepFMModel
                deepfm_model = DeepFMModel(n_users, n_items)
                deepfm_history = deepfm_model.train(processed_data, epochs=3)  # Fewer epochs for testing
                
                # Save DeepFM model
                model_path = os.path.join(models_dir, 'deepfm_model.h5')
                # DeepFM doesn't use custom layers, so we can save directly
                deepfm_model.model.save(model_path)
                print(f"DeepFM model saved to {model_path}")
                results['deepfm'] = "Success"
            except Exception as e:
                print(f"Error training DeepFM model: {e}")
                import traceback
                traceback.print_exc()
                results['deepfm'] = f"Failed: {str(e)}"
            
            # Remove hybrid model training
            
            # Save model weights (simple equal weighting or based on success)
            import pickle
            weights = {'ncf': 0.5, 'deepfm': 0.5}  # Updated to equal 50/50 weights between NCF and DeepFM
            
            with open(os.path.join(models_dir, 'model_weights.pkl'), 'wb') as f:
                pickle.dump(weights, f)
            
            # List files in models directory
            files = os.listdir(models_dir) if os.path.exists(models_dir) else []
            
            return jsonify({
                "success": True, 
                "message": "Training completed with some results",
                "results": results,
                "files_saved": files
            })
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error during direct training: {e}")
            print(error_trace)
            return jsonify({
                "success": False, 
                "error": str(e), 
                "traceback": error_trace
            })
    
    # Add a route to toggle privacy settings
    @app.route('/admin/toggle-privacy', methods=['POST'])
    def toggle_privacy():
        """Toggle privacy settings for the recommendation engine"""
        try:
            # Get current recommender
            from recommender.engine import recommender
            
            if not recommender:
                return jsonify({"success": False, "error": "Recommender not initialized"})
                
            # Toggle privacy setting
            current_setting = recommender.apply_privacy
            new_setting = not current_setting
            
            # Update recommender privacy setting
            recommender.apply_privacy = new_setting
            
            # Update data processor privacy setting
            recommender.data_processor.apply_privacy = new_setting
            
            # If turning on privacy and there's no LDP instance, create one
            if new_setting and not hasattr(recommender.data_processor, 'ldp'):
                from recommender.privacy import LocalDifferentialPrivacy
                epsilon = app.config.get('PRIVACY_EPSILON', 1.0)
                mechanism = app.config.get('PRIVACY_MECHANISM', 'laplace')
                recommender.data_processor.ldp = LocalDifferentialPrivacy(
                    epsilon=epsilon, 
                    mechanism=mechanism
                )
            
            # Save updated settings
            recommender.save()
            
            return jsonify({
                "success": True, 
                "privacy_enabled": new_setting,
                "message": f"Privacy {'enabled' if new_setting else 'disabled'}"
            })
        except Exception as e:
            app.logger.error(f"Error toggling privacy: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "error": str(e)})
    
    # Add route to adjust privacy parameters
    @app.route('/admin/update-privacy', methods=['POST'])
    def update_privacy_settings():
        """Update privacy settings for the recommendation engine"""
        try:
            import json
            from recommender.engine import recommender
            from recommender.privacy import LocalDifferentialPrivacy
            
            if not recommender:
                return jsonify({"success": False, "error": "Recommender not initialized"})
                
            # Get parameters from request
            data = request.json
            epsilon = data.get('epsilon', app.config.get('PRIVACY_EPSILON', 1.0))
            mechanism = data.get('mechanism', app.config.get('PRIVACY_MECHANISM', 'laplace'))
            
            # Validate epsilon (must be positive)
            try:
                epsilon = float(epsilon)
                if epsilon <= 0:
                    return jsonify({"success": False, "error": "Epsilon must be positive"})
            except (ValueError, TypeError):
                return jsonify({"success": False, "error": "Invalid epsilon value"})
                
            # Validate mechanism
            valid_mechanisms = ['laplace', 'gaussian', 'randomized_response']
            if mechanism not in valid_mechanisms:
                return jsonify({
                    "success": False, 
                    "error": f"Invalid mechanism. Choose from: {', '.join(valid_mechanisms)}"
                })
            
            # Update recommender privacy settings
            recommender.epsilon = epsilon
            recommender.privacy_mechanism = mechanism
            
            # Update data processor privacy settings
            recommender.data_processor.apply_privacy = True
            recommender.data_processor.ldp = LocalDifferentialPrivacy(
                epsilon=epsilon,
                mechanism=mechanism
            )
            
            # Save updated settings
            recommender.save()
            
            return jsonify({
                "success": True,
                "epsilon": epsilon,
                "mechanism": mechanism,
                "message": f"Privacy settings updated: ε={epsilon}, mechanism={mechanism}"
            })
        except Exception as e:
            app.logger.error(f"Error updating privacy settings: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "error": str(e)})
    
    # Add admin panel route for privacy settings
    @app.route('/admin/privacy-dashboard')
    def privacy_dashboard():
        """Admin dashboard for privacy settings"""
        try:
            from recommender.engine import recommender
            
            # If recommender not initialized, show default values
            if not recommender:
                privacy_status = {
                    "enabled": app.config.get('APPLY_PRIVACY', True),
                    "epsilon": app.config.get('PRIVACY_EPSILON', 1.0),
                    "mechanism": app.config.get('PRIVACY_MECHANISM', 'laplace')
                }
            else:
                privacy_status = {
                    "enabled": recommender.apply_privacy,
                    "epsilon": recommender.epsilon if hasattr(recommender, 'epsilon') else app.config.get('PRIVACY_EPSILON', 1.0),
                    "mechanism": recommender.privacy_mechanism if hasattr(recommender, 'privacy_mechanism') else app.config.get('PRIVACY_MECHANISM', 'laplace')
                }
            
            # Get training status
            training_status = {
                "is_trained": recommender.is_trained if recommender else False,
                "user_count": recommender.data_processor.get_n_users() if recommender and recommender.data_processor else 0,
                "item_count": recommender.data_processor.get_n_items() if recommender and recommender.data_processor else 0
            }
            
            return render_template(
                'admin/privacy_dashboard.html',
                privacy_status=privacy_status,
                training_status=training_status
            )
        except Exception as e:
            app.logger.error(f"Error displaying privacy dashboard: {e}")
            import traceback
            traceback.print_exc()
            return render_template('errors/500.html')

    # Add a route to expose privacy information for users
    @app.route('/api/privacy/status')
    def api_privacy_status():
        """Return privacy status information for users"""
        try:
            from recommender.engine import recommender
            
            if not recommender:
                return jsonify({
                    "privacy_enabled": False,
                    "message": "Recommendation system not initialized"
                })
            
            # Get privacy status
            privacy_enabled = recommender.apply_privacy if hasattr(recommender, 'apply_privacy') else False
            epsilon = recommender.epsilon if hasattr(recommender, 'epsilon') else None
            mechanism = recommender.privacy_mechanism if hasattr(recommender, 'privacy_mechanism') else None
            
            # Calculate privacy level (simplified for user-facing info)
            privacy_level = "high" if epsilon and epsilon < 1 else "medium" if epsilon and epsilon < 3 else "low"
            
            return jsonify({
                "privacy_enabled": privacy_enabled,
                "privacy_level": privacy_level,
                "mechanism": mechanism if privacy_enabled else None,
                "message": (
                    "Your data is protected with differential privacy" if privacy_enabled 
                    else "Standard recommendations without privacy enhancement"
                )
            })
        except Exception as e:
            app.logger.error(f"Error getting privacy status: {e}")
            return jsonify({
                "privacy_enabled": False,
                "message": "Error retrieving privacy information"
            })

    # Add a route to provide privacy-enhanced recommendations with explanations
    @app.route('/api/privacy/recommendations')
    def api_privacy_recommendations():
        """Get recommendations with privacy information"""
        try:
            from recommender.engine import recommender, get_recommendations, get_popular_products
            limit = request.args.get('limit', 5, type=int)
            
            # Check if user is logged in
            if not g.user:
                products = get_popular_products(limit=limit)
                return jsonify({
                    "recommendations": products,
                    "privacy_info": {
                        "type": "anonymous",
                        "message": "Popular products shown to anonymous users (no personalization)",
                        "privacy_enabled": False
                    }
                })
            
            # Get user ID
            user_id = str(g.user['_id'])
            
            # Get recommendations
            products = get_recommendations(user_id, limit=limit)
            
            # Get privacy info
            privacy_enabled = recommender.apply_privacy if recommender and hasattr(recommender, 'apply_privacy') else False
            privacy_level = (
                "high" if recommender and hasattr(recommender, 'epsilon') and recommender.epsilon < 1 
                else "medium" if recommender and hasattr(recommender, 'epsilon') and recommender.epsilon < 3 
                else "low"
            )
            
            # Convert ObjectId fields for JSON serialization
            for product in products:
                if isinstance(product.get('_id'), ObjectId):
                    product['_id'] = str(product['_id'])
            
            # Return recommendations with privacy info
            return jsonify({
                "recommendations": products,
                "privacy_info": {
                    "type": "personalized",
                    "message": (
                        f"Recommendations using {privacy_level} privacy protection" if privacy_enabled
                        else "Personalized recommendations without privacy enhancement"
                    ),
                    "privacy_enabled": privacy_enabled,
                    "privacy_level": privacy_level if privacy_enabled else None,
                    "data_used": [
                        "Your previous purchases", 
                        "Your product ratings", 
                        "Your browsing history"
                    ]
                }
            })
        except Exception as e:
            app.logger.error(f"Error getting privacy recommendations: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "recommendations": [],
                "privacy_info": {
                    "type": "error",
                    "message": "Error generating recommendations"
                }
            })

    # Add a route for users to opt out of recommendation system
    @app.route('/api/privacy/opt-out', methods=['POST'])
    def api_privacy_opt_out():
        """Let users opt out of the recommendation system"""
        try:
            if not g.user:
                return jsonify({
                    "success": False,
                    "message": "You must be logged in to change privacy settings"
                })
            
            # Check if the form data is provided
            data = request.json or {}
            opt_out = data.get('opt_out', True)  # Default to opt-out if not specified
            
            # Get user ID
            user_id = str(g.user['_id'])
            
            # Update user privacy settings in the database
            mongo.db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"privacy_settings.opted_out": opt_out}}
            )
            
            return jsonify({
                "success": True,
                "opted_out": opt_out,
                "message": (
                    "You have opted out of personalized recommendations" if opt_out
                    else "You have opted into personalized recommendations"
                )
            })
        except Exception as e:
            app.logger.error(f"Error processing opt-out: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "success": False,
                "message": f"Error processing privacy settings: {str(e)}"
            })

    # Add a route for a user-facing privacy explanation page
    @app.route('/privacy/recommendations')
    def privacy_recommendations_page():
        """Show privacy information about recommendations to users"""
        try:
            from recommender.engine import recommender
            
            # Get privacy status
            privacy_enabled = recommender.apply_privacy if recommender and hasattr(recommender, 'apply_privacy') else False
            epsilon = recommender.epsilon if recommender and hasattr(recommender, 'epsilon') else 1.0
            mechanism = recommender.privacy_mechanism if recommender and hasattr(recommender, 'privacy_mechanism') else 'laplace'
            
            # Calculate privacy level for display
            privacy_level = "high" if epsilon < 1 else "medium" if epsilon < 3 else "low"
            
            # Check if the user is opted out
            opted_out = False
            if g.user:
                user = mongo.db.users.find_one({"_id": ObjectId(str(g.user['_id']))})
                if user and user.get('privacy_settings', {}).get('opted_out'):
                    opted_out = True
            
            return render_template(
                'privacy/recommendations.html',
                privacy_enabled=privacy_enabled,
                privacy_level=privacy_level,
                privacy_mechanism=mechanism,
                epsilon=epsilon,
                opted_out=opted_out
            )
        except Exception as e:
            app.logger.error(f"Error displaying privacy page: {e}")
            import traceback
            traceback.print_exc()
            return render_template('errors/500.html')
            
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)