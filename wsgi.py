from app import create_app
from recommender.engine import init_recommender, schedule_training

# Create Flask application
app = create_app('production')

# Initialize recommendation engine
with app.app_context():
    init_recommender(app.config)
    
    # Schedule model training every 24 hours
    schedule_training(interval_hours=24)

# WSGI entry point
if __name__ == '__main__':
    app.run()