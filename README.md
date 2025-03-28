# Crop Disease and Soil Classification API

This project is a Flask-based API that provides functionalities for:
- **Crop Disease Detection** using a trained deep learning model.
- **Soil Classification** to identify soil type.
- **Crop Recommendation** based on soil type, groundwater level, rainfall, and soil moisture using Gemini AI.

## Features
- Accepts image inputs for **crop disease detection** and **soil classification**.
- Provides **crop recommendations** based on environmental parameters.
- Uses **TensorFlow** models for prediction.
- Fetches district-wise environmental data from a CSV file.

---

## Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install Dependencies
Ensure you have Python installed, then install required libraries:
```bash
pip install -r requirements.txt
```

### 3️⃣ Add Your Hugging Face API Key
Update the `genai.configure(api_key="YOUR_API_KEY")` line in the code with your actual API key.

### 4️⃣ Run the Flask App
```bash
python app.py
```
The server will start at `http://0.0.0.0:5000/`.

---

## API Endpoints

### 🔹 1. Crop Disease Prediction
**Endpoint:** `/predict`
**Method:** `POST`
**Input:** Image file
**Output:** Predicted crop disease and confidence score
```json
{
    "predicted_class": "Bacterial Blight in Rice",
    "confidence": 92.5
}
```

### 🔹 2. Soil Type Prediction
**Endpoint:** `/soil-predict`
**Method:** `POST`
**Input:** Image file
**Output:** Predicted soil type
```json
{
    "soil_type": "Black Soil"
}
```

### 🔹 3. Crop Recommendation
**Endpoint:** `/recommend-crops`
**Method:** `POST`
**Input:** JSON (District & Soil Type)
**Output:** Recommended crops based on environmental conditions
```json
{
    "recommended_crops": "🌾 Recommended Crops: Rice, Sugarcane, Maize"
}
```

---

## Technologies Used
- **Flask** (Backend API)
- **TensorFlow/Keras** (Deep Learning Models)
- **Pandas** (Data Handling)
- **OpenCV, Pillow** (Image Processing)
- **Google Gemini AI** (Crop Recommendation)

---

## Future Enhancements
- Deploying models to **Hugging Face Spaces**.
- Enhancing **multi-language support**.
- Adding **real-time environmental data integration**.

---

## Contributing
Feel free to fork the repository, submit issues, and contribute improvements!

---

## License
This project is licensed under the MIT License.

