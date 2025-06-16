from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("churn_model.pkl")

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from request
    df = pd.DataFrame([data])  # Convert to DataFrame
    
    # Make prediction
    prediction = model.predict(df)
    churn_probability = model.predict_proba(df)[:,1]
    
    # Return result
    return jsonify({
        "Churn Prediction": int(prediction[0]), 
        "Churn Probability": float(churn_probability[0])
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
