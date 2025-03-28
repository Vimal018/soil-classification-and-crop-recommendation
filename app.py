import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import io
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)

genai.configure(api_key="AIzaSyCFCYZ1G_o-fVTJnfmNNv3GHBlWw1ZzhzQ")  # Replace with your actual API key

# Load trained models
crop_disease_model = load_model("model.h5")  # Crop disease detection model
soil_model = load_model("soil_classifier.h5")  # Soil classification model

# Define class labels
CROP_DISEASE_LABELS = {
    0: "American Bollworm on Cotton", 1: "Anthracnose on Cotton", 2: "Army worm",
    3: "Bacterial Blight in Rice", 4: "Brownspot", 5: "Common_Rust", 6: "Cotton Aphid",
    7: "Flag Smut", 8: "Gray_Leaf_Spot", 9: "Healthy Maize", 10: "Healthy Wheat",
    11: "Healthy Cotton", 12: "Leaf Curl", 13: "Leaf Smut", 14: "Mosaic Sugarcane",
    15: "RedRot Sugarcane", 16: "RedRust Sugarcane", 17: "Rice Blast", 18: "Sugarcane Healthy",
    19: "Tungro", 20: "Wheat Brown Leaf Rust", 21: "Wheat Stem Fly", 22: "Wheat Aphid",
    23: "Wheat Black Rust", 24: "Wheat Leaf Blight", 25: "Wheat Mite", 26: "Wheat Powdery Mildew",
    27: "Wheat Scab", 28: "Wheat Yellow Rust", 29: "Wilt", 30: "Yellow Rust Sugarcane",
    31: "Bacterial Blight in Cotton", 32: "Boll Rot on Cotton", 33: "Bollworm on Cotton",
    34: "Cotton Mealy Bug", 35: "Cotton Whitefly", 36: "Maize Ear Rot", 37: "Maize Fall Armyworm",
    38: "Maize Stem Borer", 39: "Pink Bollworm in Cotton", 40: "Red Cotton Bug", 41: "Thrips on Cotton"
}

SOIL_LABELS = ['Alluvial soil', 'Black', 'Chalky', 'Clay soil', 'Mary', 'Red soil', 'Sand', 'Silt']

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict_disease():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        prediction = crop_disease_model.predict(processed_image)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class = CROP_DISEASE_LABELS[predicted_class_index]
        confidence = round(float(np.max(prediction)) * 100, 2)
        
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/soil-predict', methods=['POST'])
def predict_soil():
    try:
        file = request.files['image']
        image = Image.open(file)
        processed_image = preprocess_image(image)
        
        predictions = soil_model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        
        return jsonify({"soil_type": SOIL_LABELS[predicted_class_index]})
    except Exception as e:
        return jsonify({"error": str(e)})

def get_district_info_from_csv(district, df):
    district_data = df[df['District'].str.lower() == district.lower()]
    return district_data.iloc[0] if not district_data.empty else None

def get_crop_recommendations(soil_type, district, groundwater, rainfall, soil_moisture):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"Recommend the best 3 crops for the {district} district based on:\n"
        f"- Soil Type: {soil_type}\n"
        f"- Groundwater Level: {groundwater}\n"
        f"- Rainfall: {rainfall}mm\n"
        f"- Soil Moisture: {soil_moisture}%\n"
        "Give precise crop recommendations."
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

@app.route('/recommend-crops', methods=['POST'])
def recommend_crops():
    try:
        data = request.json
        district = data.get("district")
        soil_type = data.get("soil_type")
        df = pd.read_csv("tn_final.csv")
        district_info = get_district_info_from_csv(district, df)
        
        if district_info is None:
            return jsonify({"error": "District data not found."})
        
        crops = get_crop_recommendations(
            soil_type, district,
            district_info['current water level'],
            district_info['Actual Rainfall (mm)'],
            district_info['Soil Moisture (%)']
        )
        
        response = (
            f"🌱 Crop Recommendation for {district}\n"
            f"📍 District: {district}\n"
            f"🏔️ Predicted Soil Type: {soil_type}\n"
            f"💧 Groundwater Level: {district_info['current water level']}m\n"
            f"🌧️ Actual Rainfall: {district_info['Actual Rainfall (mm)']}mm\n"
            f"🌿 Soil Moisture: {district_info['Soil Moisture (%)']}%\n\n"
            f"🌾 Recommended Crops: {crops}"
        )
        
        return jsonify({"recommended_crops": response})
    except Exception as e:
        return jsonify({"error": str(e)})
        
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
