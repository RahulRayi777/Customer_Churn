import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\naray\OneDrive\Pictures\Desktop\01-College_Stuff\Customer_Churn\customer_churn_dataset.csv")

# Encode categorical features
label_encoders = {
    "Gender": LabelEncoder(),
    "Subscription Type": LabelEncoder(),
    "Contract Length": LabelEncoder()
}

for col in label_encoders:
    df[col] = label_encoders[col].fit_transform(df[col])

# Define features and target
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# Save model, scaler, and encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))
