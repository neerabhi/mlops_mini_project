# bow vs tfidf

# Import necessary libraries
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import nltk
import dagshub
import os

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Set MLflow tracking URI and initialize DagsHub
mlflow.set_tracking_uri('https://dagshub.com/neerabhi/mlops_mini_project.mlflow')
dagshub.init(repo_owner='neerabhi', repo_name='mlops_mini_project', mlflow=True)

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])

# Define text preprocessing functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

# Normalize and filter data
df = normalize_text(df)
df = df[df['sentiment'].isin(['happiness','sadness'])]
df['sentiment'] = df['sentiment'].replace({'sadness': 0, 'happiness': 1})

# Set the experiment
mlflow.set_experiment("BoW vs TF-IDF again")

# Define vectorizers and algorithms
vectorizers = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

algorithms = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

# Start MLflow parent run
with mlflow.start_run(run_name="All Experiments") as parent_run:
    for algo_name, algorithm in algorithms.items():
        for vec_name, vectorizer in vectorizers.items():
            with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                try:
                    # Vectorization and train-test split
                    X = vectorizer.fit_transform(df['content'])
                    y = df['sentiment']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Log params
                    mlflow.log_param("vectorizer", vec_name)
                    mlflow.log_param("algorithm", algo_name)
                    mlflow.log_param("test_size", 0.2)

                    # Train model
                    model = algorithm
                    model.fit(X_train, y_train)

                    # Log model hyperparameters
                    if algo_name == 'LogisticRegression':
                        mlflow.log_param("C", model.C)
                    elif algo_name == 'MultinomialNB':
                        mlflow.log_param("alpha", model.alpha)
                    elif algo_name == 'XGBoost':
                        mlflow.log_param("n_estimators", model.n_estimators)
                        mlflow.log_param("learning_rate", model.learning_rate)
                    elif algo_name == 'RandomForest':
                        mlflow.log_param("n_estimators", model.n_estimators)
                        mlflow.log_param("max_depth", model.max_depth)
                    elif algo_name == 'GradientBoosting':
                        mlflow.log_param("n_estimators", model.n_estimators)
                        mlflow.log_param("learning_rate", model.learning_rate)
                        mlflow.log_param("max_depth", model.max_depth)

                    # Evaluate model
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    # Log metrics
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1_score", f1)

                    # Skip model logging (DagsHub does not support it)
                    print("⚠️ Skipped logging model: DagsHub does not support model artifacts via this endpoint.")

                    # Log script if available
                    if "__file__" in globals():
                        mlflow.log_artifact(__file__)
                    else:
                        print("Skipped logging script source.")

                    # Print results
                    print(f"\n✅ {algo_name} with {vec_name}")
                    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

                except Exception as e:
                    print(f"\n❌ Error in {algo_name} with {vec_name}: {e}")
