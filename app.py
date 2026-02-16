import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import Counter
import base64
from PIL import Image
import io
# Import necessary libraries for YOLO if not already imported globally
# Example:
# from some_yolo_library import YOLOModel, preprocess_image
from ultralytics import YOLO
from datetime import datetime
FEEDBACK_FILE = "feedback.txt"


FOOD_CATEGORIES = [
    'apple pie', 'burger', 'cheesecake', 'chicken curry',
    'chicken wings', 'chocolate cake', 'donuts', 'fries',
    'hot dog', 'ice cream', 'pizza', 'sandwich',
    'steak', 'tacos', 'waffles'
]

YOLO_TO_CATEGORY = {
    'apple pie': 'apple pie',
    'burger': 'burger',
    'cheeseburger': 'burger',
    'pizza': 'pizza',
    'sandwich': 'sandwich',
    'hot dog': 'hot dog',
    'taco': 'tacos',
    'steak': 'steak',
    'fries': 'fries',
    'waffle': 'waffles',
    'ice cream': 'ice cream',
    'donut': 'donuts',
    'cheesecake': 'cheesecake',
    'chicken wings': 'chicken wings',
    'chicken curry': 'chicken curry',
    'chocolate cake': 'chocolate cake'
}

def map_to_categories(detected_items):
    categories = []
    for item in detected_items:
        if item in YOLO_TO_CATEGORY:
            categories.append(YOLO_TO_CATEGORY[item])
        else:
            for yolo_class, category in YOLO_TO_CATEGORY.items():
                if yolo_class in item:
                    categories.append(category)
                    break
    return categories

class RestaurantRecommender:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self._preprocess_data()

    def _preprocess_data(self):
        self.df.dropna(subset=['Rating', 'Price', 'Latitude', 'Longitude'], inplace=True)
        self.df['Category'] = self.df['Meal'].apply(self._categorize_meal)

        if 'Full Address' in self.df.columns:
            self.df['Address'] = self.df['Full Address']
        else:
            self.df['Address'] = self.df.apply(
                lambda row: f"Near ({row['Latitude']:.3f}, {row['Longitude']:.3f})", axis=1
            )

        scaler = StandardScaler()
        features = self.df[['Price', 'Rating', 'Latitude', 'Longitude']]
        self.df[['Price_scaled', 'Rating_scaled', 'Lat_scaled', 'Lon_scaled']] = scaler.fit_transform(features)

        self.feature_matrix = self.df[['Price_scaled', 'Rating_scaled', 'Lat_scaled', 'Lon_scaled']].values
        self.similarity_matrix = cosine_similarity(self.feature_matrix)


    def _categorize_meal(self, meal_name):
        meal_lower = meal_name.lower()
        for category in FOOD_CATEGORIES:
            if category in meal_lower:
                return category
        return 'other'

    def recommend_by_category(self, categories, top_n=5):
        if not categories:
            return self.df.drop_duplicates('Restaurant').nlargest(top_n, 'Rating')

        relevant = self.df[self.df['Category'].isin(categories)]
        if len(relevant) == 0:
            return self.df.drop_duplicates('Restaurant').nlargest(top_n, 'Rating')

        category_counts = Counter(categories)

        def score_restaurant(row):
            score = 0
            for category, count in category_counts.items():
                if row['Category'] == category:
                    score += count
            return score

        relevant['MatchScore'] = relevant.apply(score_restaurant, axis=1)
        unique_restaurants = (relevant
                              .sort_values(['MatchScore', 'Rating'], ascending=[False, False])
                              .drop_duplicates('Restaurant')
                              .head(top_n))
        return unique_restaurants

    def recommend_by_restaurant(self, restaurant_name, top_n=5):
        if restaurant_name not in self.df['Restaurant'].values:
            return self.df.nlargest(top_n, 'Rating')

        idx = self.df[self.df['Restaurant'] == restaurant_name].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]

        restaurant_indices = [i[0] for i in sim_scores]
        return self.df.iloc[restaurant_indices]

    def get_recommendations(self, detected_items=None, restaurant_name=None, top_n=5):
        if detected_items:
            categories = map_to_categories(detected_items)
            return self.recommend_by_category(categories, top_n)
        elif restaurant_name:
            return self.recommend_by_restaurant(restaurant_name, top_n)
        else:
            return self.df.nlargest(top_n, 'Rating')

# Flask app
app = Flask(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "irish_restaurants_meals_final.csv")
recommender = RestaurantRecommender(data_path)

# Initialize your YOLO model here if it's a class or needs loading
# Eg:
yolo_model = YOLO("./weights/best.pt")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    image_data = data.get("image")

    if not image_data:
        return jsonify({"status": "error", "message": "No image data provided"})

    try:
        # Decode the base64 image data
        # The image data is expected to be in the format "data:image/jpeg;base64,..."
        header, base64_string = image_data.split(',', 1)
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))

        # TODO: Integrate your YOLO detection code here
        # Run your YOLO model on the 'image' variable (PIL Image object)
        # The YOLO model should return a list of detected object class names (strings)
        # Example:
        # results = yolo_model(image)
        # detected_items = [box.cls for box in results.xyxy[0]] # Adjust based on your YOLO output format

        # Example using the provided YOLO model object:
        results = yolo_model(image)
        detected_items = []
        for r in results:
            for c in r.boxes.cls:
                detected_items.append(yolo_model.names[int(c)])

        # detected_items = ['burger', 'fries']  # TEMP: Mock for now - REMOVE THIS LINE

        recommendations = recommender.get_recommendations(detected_items=detected_items)

        recommendation_list = recommendations[[
            'Restaurant', 'Meal', 'Price', 'Rating', 'Address', 'Latitude', 'Longitude'
        ]].to_dict(orient='records')

        # Include the original image data (or processed image data) in the response
        # You can send back the original base64 string or re-encode the processed image
        response_data = {
            "status": "success",
            "detected_items": detected_items,
            "recommendations": recommendation_list,
            "image": image_data # Include the image data in the response
        }


        return jsonify(response_data)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        username = data.get('username', 'anonymous')
        rating = data.get('rating')
        comments = data.get('comments', '')
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(FEEDBACK_FILE, 'a') as f:
            f.write(f"{timestamp},{username},{rating},{comments}\n")
        
        return jsonify({"status": "success"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get-feedback')
def get_feedback():
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            feedbacks = [line.strip().split(',', 3) for line in f.readlines()]
        return jsonify({"status": "success", "feedbacks": feedbacks})
    except FileNotFoundError:
        return jsonify({"status": "success", "feedbacks": []})


if __name__ == "__main__":
    # If running in Colab, you might need to use a different method to expose the Flask app
    # like ngrok or flask-ngrok.
    # For a standard local run, app.run() is fine.
    # Example for Colab with flask-ngrok:
    # from flask_ngrok import run_with_ngrok
    # run_with_ngrok(app)
    #app.run(debug=True)
    app.run(host='127.0.0.1', port=8080)
