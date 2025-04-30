from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

# Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), 'mlproject.csv')
INTRODUCE_NOISE = False
DEBUG_MODE = True

# Load and preprocess data
data = pd.read_csv(DATA_PATH)
target_column = 'Class'
feature_columns = [col for col in data.columns if col != target_column]
X = data[feature_columns].values
y = data[target_column].values

# Class mapping
class_labels = np.unique(y)
class_mapping = {int(label): str(label) for label in class_labels}

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if INTRODUCE_NOISE:
    np.random.seed(42)
    y_noisy = y.copy()
    num_noisy_labels = int(0.10 * len(y))
    noise_indices = np.random.choice(len(y), num_noisy_labels, replace=False)
    y_noisy[noise_indices] = np.random.choice(np.unique(y), num_noisy_labels)
else:
    y_noisy = y

X_scaled = X_scaled[:5000]
y_noisy = y_noisy[:5000]

kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Train models
def train_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    }

    param_grids = {
        "Logistic Regression": {'C': [0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']},
        "Random Forest": {'n_estimators': [100], 'max_depth': [10], 'min_samples_split': [2]},
        "SVM": {'C': [0.1, 1], 'gamma': ['scale'], 'kernel': ['rbf']}
    }

    trained_models = {}

    for name, model in models.items():
        search = RandomizedSearchCV(model, param_grids[name], n_iter=5, cv=kf, scoring='accuracy', n_jobs=-1)
        search.fit(X_scaled, y_noisy)
        trained_models[name] = search.best_estimator_

    bagging = BaggingClassifier(
        estimator=RandomForestClassifier(random_state=42),
        n_estimators=50,
        random_state=42
    )

    bag_search = RandomizedSearchCV(
        bagging,
        {'n_estimators': [50], 'estimator__max_depth': [5, 10], 'estimator__min_samples_split': [2]},
        n_iter=5,
        cv=kf,
        scoring='accuracy',
        n_jobs=-1
    )
    bag_search.fit(X_scaled, y_noisy)
    trained_models["Bagging"] = bag_search.best_estimator_

    stacking = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=1, penalty='l2', solver='lbfgs')),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('svc', SVC(C=1, gamma='scale', kernel='rbf', probability=True, random_state=42))
        ],
        final_estimator=LogisticRegression(max_iter=200)
    )
    stacking.fit(X_scaled, y_noisy)
    trained_models["Stacking"] = stacking

    return trained_models

models = train_models()

@app.route('/')
def home():
    return render_template('index.html', feature_columns=feature_columns)

@app.route('/form_predict', methods=['POST'])
def form_predict():
    try:
        input_data = []
        for col in feature_columns:
            val = request.form.get(col)
            if val is None:
                return jsonify({'error': f'Missing value for {col}'}), 400
            input_data.append(float(val))

        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        predictions = {}
        for name, model in models.items():
            pred = model.predict(input_scaled)[0]
            predictions[name] = class_mapping.get(int(pred), str(pred))

        return render_template('index.html', predictions=predictions, feature_columns=feature_columns)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json.get('features')
        if not input_data:
            return jsonify({'error': 'No features provided'}), 400

        if len(input_data) != len(feature_columns):
            return jsonify({'error': f'Expected {len(feature_columns)} features, got {len(input_data)}'}), 400

        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        predictions = {}
        for name, model in models.items():
            pred = model.predict(input_scaled)[0]
            predictions[name] = class_mapping.get(int(pred), str(pred))

        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        df = pd.read_csv(file)
        if not all(col in df.columns for col in feature_columns):
            return jsonify({'error': f'CSV file must contain the following columns: {feature_columns}'}), 400

        X_batch = df[feature_columns].values
        X_batch_scaled = scaler.transform(X_batch)

        results = []
        for i in range(X_batch_scaled.shape[0]):
            sample = X_batch_scaled[i].reshape(1, -1)
            sample_preds = {}
            for name, model in models.items():
                pred = model.predict(sample)[0]
                sample_preds[name] = class_mapping.get(int(pred), str(pred))
            results.append(sample_preds)

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=DEBUG_MODE)
