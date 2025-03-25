import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load pre-trained model, scaler, and label encoders
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from form
        data = request.form

        # Convert input values to appropriate types
        age = int(data["age"])
        gender = data["gender"]
        tenure = int(data["tenure"])
        usage_frequency = float(data["usage_frequency"])
        support_calls = int(data["support_calls"])
        payment_delay = int(data["payment_delay"])
        subscription_type = data["subscription_type"]
        contract_length = data["contract_length"]
        total_spend = float(data["total_spend"])
        last_interaction = int(data["last_interaction"])

        # Encode categorical values
        gender_encoded = label_encoders["Gender"].transform([gender])[0]
        subscription_encoded = label_encoders["Subscription Type"].transform([subscription_type])[0]
        contract_encoded = label_encoders["Contract Length"].transform([contract_length])[0]

        # Create feature array
        features = np.array([
            age, gender_encoded, tenure, usage_frequency, support_calls, 
            payment_delay, subscription_encoded, contract_encoded, 
            total_spend, last_interaction
        ]).reshape(1, -1)

        # Scale the input data
        features_scaled = scaler.transform(features)

        # Predict churn probability
        churn_prob = model.predict_proba(features_scaled)[0][1]  # Probability of churn
        churn_percentage = round(churn_prob * 100, 2)  # Convert to percentage

        # Determine risk level
        if churn_prob >= 0.75:
            risk = "High Risk"
        elif churn_prob >= 0.50:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        return jsonify({
            "churn_probability": f"{churn_percentage}%",
            "risk_level": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
