from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from fuzzywuzzy import fuzz

app = Flask(__name__)
DATABASE = "datas.db"

def create_connection():
    return sqlite3.connect(DATABASE)

# Fetch data from SQLite and normalize
def fetch_data():
    conn = create_connection()
    programme_query = "SELECT * FROM programme"
    subject_query = """
    SELECT subject.id, LOWER(subject.name) AS subject_name, programme.name AS programme_name
    FROM subject
    JOIN programme ON subject.programme_id = programme.id
    """
    programme_df = pd.read_sql_query(programme_query, conn)
    subject_df = pd.read_sql_query(subject_query, conn)
    conn.close()
    return programme_df, subject_df

# Prepare DataFrame for Machine Learning
def prepare_data(subject_df):
    mlb = MultiLabelBinarizer()
    grouped = subject_df.groupby("programme_name")["subject_name"].apply(list).reset_index()
    subject_matrix = mlb.fit_transform(grouped["subject_name"])
    subject_columns = mlb.classes_
    
    subject_df_ml = pd.DataFrame(subject_matrix, columns=subject_columns)
    programme_df_ml = pd.concat([grouped["programme_name"], subject_df_ml], axis=1)
    return programme_df_ml, subject_columns

# Train Random Forest Model
def train_model(data, subject_columns):
    X = data[subject_columns]
    y = data["programme_name"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to validate subjects with fuzzy matching
def validate_subjects_fuzzy(input_subjects, valid_subjects, threshold=70):
    valid_subjects = [subject.lower() for subject in valid_subjects]
    matched_subjects = []
    
    for input_subject in input_subjects:
        matched = False
        for valid_subject in valid_subjects:
            similarity = fuzz.ratio(input_subject.lower(), valid_subject)
            if similarity >= threshold:
                matched_subjects.append(valid_subject)
                matched = True
                break
        if not matched:
            return False, matched_subjects
    return True, matched_subjects

# Load and prepare data
programme_df, subject_df = fetch_data()
data, subject_columns = prepare_data(subject_df)
model = train_model(data, subject_columns)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    subjects_input = request.form.get('subjects')
    if not subjects_input:
        return jsonify({"error": "Please enter subjects."})
    
    student_subjects = [subject.strip().lower() for subject in subjects_input.split(",")]
    if len(student_subjects) < 3:
        return jsonify({"error": "Please enter at least 3 subjects."})
    
    valid_subjects = set(subject_columns)
    is_valid, matched_subjects = validate_subjects_fuzzy(student_subjects, valid_subjects)

    if not is_valid:
        return jsonify({"error": "Some entered subjects are not recognized.", "matched_subjects": matched_subjects})
    
    input_data = [1 if subject in matched_subjects else 0 for subject in subject_columns]
    probabilities = model.predict_proba([input_data])[0]
    recommendations = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
    
    top_recommendations = [{"program": program, "probability": f"{prob * 100:.1f}%"} for program, prob in recommendations[:5]]
    return jsonify(top_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
