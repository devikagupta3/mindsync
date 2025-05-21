import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
dataset_path = "C:/flaskapp/mindsync5/dataset/mentaldata.csv"
df = pd.read_csv(dataset_path)

# Preprocessing
le = LabelEncoder()
categorical_cols = ['gender', 'ethnicity', 'education', 'employment']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop('mental_health_status', axis=1)
y = df['mental_health_status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/mentalhealth_model.pkl')

print("Model trained and saved successfully!")