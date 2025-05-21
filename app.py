from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change for production

# Load model
model_path = 'model/mentalhealth_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None  # Handle missing model gracefully

# Label Encoders
le_gender = LabelEncoder().fit(['Male', 'Female', 'Non-binary'])
le_ethnicity = LabelEncoder().fit(['Caucasian', 'African American', 'Asian', 'Hispanic', 'Other'])
le_education = LabelEncoder().fit(['High School', 'Some College', "Bachelor's", "Master's", "PhD"])
le_employment = LabelEncoder().fit(['Employed', 'Unemployed', 'Student', 'Retired'])

@app.route('/')
def welcome():
    return render_template('welcome.html')  # Renders the first page

@app.route('/demographics', methods=['GET', 'POST'])
def demographics():
    if request.method == 'POST':
        session.update(request.form.to_dict())
        return redirect(url_for('phq9'))
    return render_template('demographics.html')

@app.route('/phq9', methods=['GET', 'POST'])
def phq9():
    if request.method == 'POST':
        session.update(request.form.to_dict())
        return redirect(url_for('gad7'))
    return render_template('phq9.html')

@app.route('/gad7', methods=['GET', 'POST'])
def gad7():
    if request.method == 'POST':
        session.update(request.form.to_dict())
        return redirect(url_for('lifestyle'))
    return render_template('gad7.html')

@app.route('/lifestyle', methods=['GET', 'POST'])
def lifestyle():
    if request.method == 'POST':
        session.update(request.form.to_dict())
        return redirect(url_for('results'))
    return render_template('lifestyle.html')

@app.route('/results')
def results():
    try:
        # Ensure session contains required data
        if not session.get("phq1") or not session.get("gad1"):
            return redirect(url_for('phq9'))  # Redirect to PHQ-9 if missing data

        # Calculate PHQ-9 and GAD-7 scores
        phq9_score = sum(int(session.get(f'phq{q}', 0)) for q in range(1, 10))
        gad7_score = sum(int(session.get(f'gad{q}', 0)) for q in range(1, 8))

        # Prepare input data
        input_data = {
            'age': int(session.get('age', 30)),
            'gender': session.get('gender', 'Male'),
            'ethnicity': session.get('ethnicity', 'Caucasian'),
            'education': session.get('education', "Bachelor's"),
            'employment': session.get('employment', 'Employed'),
            'income': int(session.get('income', 50000)),
            'bmi': float(session.get('bmi', 24.5)),
            'sleep_quality': int(session.get('sleep_quality', 6)),
            'exercise_frequency': int(session.get('exercise_frequency', 3)),
            'chronic_conditions': int(session.get('chronic_conditions', 0)),
            'phq9_score': phq9_score,
            'gad7_score': gad7_score,
            'stress_score': int(session.get('stress_score', 10)),
            'resilience_score': int(session.get('resilience_score', 15)),
            'social_support': int(session.get('social_support', 5)),
            'loneliness_score': int(session.get('loneliness_score', 3)),
            'alcohol_consumption': int(session.get('alcohol_consumption', 2)),
            'screen_time': float(session.get('screen_time', 5.0)),
            'caffeine_intake': int(session.get('caffeine_intake', 2)),
            'work_balance': int(session.get('work_balance', 3)),
            'neighborhood_safety': int(session.get('neighborhood_safety', 4))
        }

        # Create DataFrame and encode categorical variables
        input_df = pd.DataFrame([input_data])
        input_df['gender'] = le_gender.transform([input_df['gender'][0]])[0]
        input_df['ethnicity'] = le_ethnicity.transform([input_df['ethnicity'][0]])[0]
        input_df['education'] = le_education.transform([input_df['education'][0]])[0]
        input_df['employment'] = le_employment.transform([input_df['employment'][0]])[0]

        # Default prediction message
        prediction = "No prediction available."

        # Predict mental health if model exists
        if model:
            prediction = model.predict(input_df)[0]

        return render_template('results.html',
                               prediction=prediction,
                               phq9_score=phq9_score,
                               gad7_score=gad7_score,
                               phq9_level=get_severity(phq9_score, 'phq9'),
                               gad7_level=get_severity(gad7_score, 'gad7'))

    except Exception as e:
        return render_template('results.html', prediction="Error occurred.", error=str(e))

def get_severity(score, test_type):
    if test_type == 'phq9':
        if score < 5:
            return "Minimal"
        elif score < 10:
            return "Mild"
        elif score < 15:
            return "Moderate"
        elif score < 20:
            return "Moderately severe"
        else:
            return "Severe"
    else:  # GAD-7
        if score < 5:
            return "Minimal"
        elif score < 10:
            return "Mild"
        elif score < 15:
            return "Moderate"
        else:
            return "Severe"

if __name__ == '__main__':
    app.run(debug=True)
