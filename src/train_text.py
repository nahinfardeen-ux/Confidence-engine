import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Constants
DATA_PATH = "Sentiment Analysis for Financial News/all-data.csv"
VECTORIZER_PATH = "models/vectorizer.pkl"
MODEL_PATH = "models/text_model.pkl"

def train_text_model():
    print("Loading data...")
    # Load dataset. Assuming no header, as seen in the file preview.
    # Col 0 is Label, Col 1 is Text.
    # Using 'latin-1' encoding which is common for this dataset if utf-8 fails, but let's try defaults or handle errors.
    try:
        df = pd.read_csv(DATA_PATH, header=None, encoding='utf-8', names=['label', 'text'])
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, header=None, encoding='latin-1', names=['label', 'text'])

    print(f"Original dataset shape: {df.shape}")
    
    # Filter for positive and negative only
    df = df[df['label'].isin(['positive', 'negative'])]
    print(f"Filtered dataset shape: {df.shape}")

    # Map labels: positive -> 1, negative -> 0
    df['label_num'] = df['label'].map({'positive': 1, 'negative': 0})

    X = df['text']
    y = df['label_num']

    # Vectorization
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Train Model
    print("Training Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Save Vectorizer and Model
    print("Saving model and vectorizer...")
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Saved to {VECTORIZER_PATH} and {MODEL_PATH}")

if __name__ == "__main__":
    train_text_model()
