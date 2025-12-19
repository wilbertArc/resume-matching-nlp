import os
import re
import io
import fitz
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform

_SBERT_MODEL = None

# Default paths (same as notebook)
PATH_RAW_FOLDERS = "archive(1)/data/data"


def clean_text(text):
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def extract_text_from_pdf_bytes(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception:
        return ""


def load_data_from_folders(base_path=PATH_RAW_FOLDERS):
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Folder '{base_path}' not found.")
    data = []
    for root, dirs, files in os.walk(base_path):
        category = os.path.basename(root)
        if root == base_path:
            continue
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        if not pdf_files:
            continue
        for file in pdf_files:
            file_path = os.path.join(root, file)
            try:
                with fitz.open(file_path) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                data.append({
                    "Category": category,
                    "Filename": file,
                    "Cleaned_Text": clean_text(text)
                })
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
    return pd.DataFrame(data)


def _ensure_sbert(model_name="all-mpnet-base-v2"):
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            raise ImportError("Please install sentence-transformers to use embeddings: pip install -U sentence-transformers")
        _SBERT_MODEL = SentenceTransformer(model_name)
    return _SBERT_MODEL


def embed_texts(texts, batch_size=64, model_name="all-mpnet-base-v2"):
    model = _ensure_sbert(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)


def train_model(df, tfidf_max_features=20000, n_iter=20, random_state=42):
    X = df['Cleaned_Text'].astype('U')
    y = df['Category']

    tfidf_vectorizer = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(X.tolist())

    emb = embed_texts(X.tolist())
    emb_sparse = sparse.csr_matrix(emb)
    X_combined = sparse.hstack([emb_sparse, tfidf_matrix]).tocsr()

    base_clf = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='saga', n_jobs=-1, class_weight='balanced')
    param_dist = {'C': loguniform(1e-4, 1e4)}
    search = RandomizedSearchCV(base_clf, param_distributions=param_dist, n_iter=n_iter, cv=5, scoring='accuracy', n_jobs=-1, random_state=random_state)
    search.fit(X_combined, y)
    clf = search.best_estimator_
    return {
        'clf': clf,
        'tfidf': tfidf_vectorizer,
        'search': search
    }


def predict_topk(texts, clf, tfidf_vectorizer, top_k=3):
    texts_clean = [clean_text(t) for t in texts]
    tfidf = tfidf_vectorizer.transform(texts_clean)
    emb = embed_texts(texts_clean)
    emb_sparse = sparse.csr_matrix(emb)
    X_combined = sparse.hstack([emb_sparse, tfidf]).tocsr()

    probs = clf.predict_proba(X_combined)
    classes = clf.classes_
    results = []
    for i, p in enumerate(probs):
        idxs = np.argsort(p)[::-1][:top_k]
        results.append([(classes[j], float(p[j])) for j in idxs])
    return results


def save_artifacts(path_dir, clf, tfidf_vectorizer):
    os.makedirs(path_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(path_dir, 'clf.joblib'))
    joblib.dump(tfidf_vectorizer, os.path.join(path_dir, 'tfidf.joblib'))


def load_artifacts(path_dir):
    clf_path = os.path.join(path_dir, 'clf.joblib')
    tfidf_path = os.path.join(path_dir, 'tfidf.joblib')
    if not os.path.exists(clf_path) or not os.path.exists(tfidf_path):
        raise FileNotFoundError('Model artifacts not found in ' + path_dir)
    clf = joblib.load(clf_path)
    tfidf = joblib.load(tfidf_path)
    return clf, tfidf
