from flask import Flask, request, jsonify, render_template
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(cleaned_tokens)

def expand_abbreviations(text):
    abbrev_dict = {
        "MI": "Myocardial Infarction",
        "DM2": "Type 2 Diabetes",
        "HTN": "Hypertension",
        "CAD": "Coronary Artery Disease",
    }
    for abbrev, expansion in abbrev_dict.items():
        text = text.replace(abbrev, expansion)
    return text

def normalize_concepts(text):
    concept_map = {
        "heart attack": "Myocardial Infarction",
        "high blood pressure": "Hypertension",
        "type ii diabetes": "Type 2 Diabetes",
    }
    for concept, normalized in concept_map.items():
        text = text.replace(concept, normalized)
    return text

def extract_entities(text):
    if isinstance(text, str):
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    else:
        return []

@app.route('/html_sample.html')
def serve_html():
    return render_template('html_sample.html')

@app.route('/process', methods=['POST'])
def process_data():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        df['cleaned_description'] = df['description'].apply(clean_text)
        df['expanded_description'] = df['cleaned_description'].apply(expand_abbreviations)
        df['normalized_description'] = df['expanded_description'].apply(normalize_concepts)
        df['entities'] = df['description'].apply(extract_entities)
        df = df.dropna(subset=['diagnosis'])

        class_counts = df['diagnosis'].value_counts()
        single_instance_classes = class_counts[class_counts == 1].index
        df = df[~df['diagnosis'].isin(single_instance_classes)]

        X = df['normalized_description']
        y = df['diagnosis']

        if len(y.unique()) < 2:
            return jsonify({'error': "Not enough unique classes for stratification."})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        rf_model = RandomForestClassifier(random_state=42)
        rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=2, n_jobs=-1, verbose=0)
        rf_grid_search.fit(X_train_tfidf, y_train)
        rf_best_model = rf_grid_search.best_estimator_
        rf_y_pred = rf_best_model.predict(X_test_tfidf)

        lr_model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        lr_param_grid = {'C': [0.1, 1], 'penalty': ['l1', 'l2']}
        lr_grid_search = GridSearchCV(lr_model, lr_param_grid, cv=2, n_jobs=-1, verbose=0)
        lr_grid_search.fit(X_train_tfidf, y_train)
        lr_best_model = lr_grid_search.best_estimator_
        lr_y_pred = lr_best_model.predict(X_test_tfidf)

        rf_report = classification_report(y_test, rf_y_pred, zero_division=0)
        lr_report = classification_report(y_test, lr_y_pred, zero_division=0)

        results = [ # Changed from {} to []
        {'Model': 'Random Forest', 'Accuracy': accuracy_score(y_test, rf_y_pred)},
        {'Model': 'Logistic Regression', 'Accuracy': accuracy_score(y_test, lr_y_pred)}
        ]

        cm = confusion_matrix(y_test, rf_y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix (Random Forest)")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return jsonify({
            'rf_report': rf_report,
            'lr_report': lr_report,
            'model_comparison': results,
            'confusion_matrix_image': image_base64,
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)