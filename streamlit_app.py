import streamlit as st
import pandas as pd
import os

from resume_pipeline import (
    extract_text_from_pdf_bytes,
    clean_text,
    load_artifacts,
    predict_topk,
)

st.set_page_config(page_title="Resume Classifier (inference)", layout="wide")
st.title("Resume Classifier — Inference Only")

# Sidebar settings
with st.sidebar:
    st.header("Model / Settings")
    artifacts_dir = st.text_input('Artifacts directory', value='model_artifacts')
    top_k = st.number_input('Top K predictions', min_value=1, max_value=10, value=3)
    st.markdown('---')
    st.write('Place your trained artifacts (clf.joblib, tfidf.joblib) into the artifacts folder.')

# Auto-load artifacts at startup
if 'model_loaded' not in st.session_state:
    try:
        clf, tfidf = load_artifacts(artifacts_dir)
        st.session_state['clf'] = clf
        st.session_state['tfidf'] = tfidf
        st.session_state.model_loaded = True
        st.success(f'Model loaded from {artifacts_dir}')
    except Exception:
        st.session_state.model_loaded = False

st.header('Upload a single resume (PDF)')
uploaded_file = st.file_uploader("Upload a PDF", type='pdf')

if uploaded_file is not None:
    raw = uploaded_file.read()
    text = extract_text_from_pdf_bytes(raw)
    cleaned = clean_text(text)

    st.subheader('Cleaned text')
    st.text_area('', cleaned, height=300)

    if not st.session_state.get('model_loaded'):
        st.error('Model artifacts not loaded. Place clf.joblib and tfidf.joblib in the artifacts folder and refresh.')
    else:
        if st.button('Predict'):
            try:
                clf = st.session_state['clf']
                tfidf = st.session_state['tfidf']
                preds = predict_topk([cleaned], clf, tfidf, top_k=top_k)
                rows = [{'rank': i+1, 'category': p[0], 'score': p[1]} for i, p in enumerate(preds[0])]
                dfp = pd.DataFrame(rows)
                st.table(dfp)
                csv = dfp.to_csv(index=False).encode('utf-8')
                st.download_button('Download predictions CSV', data=csv, file_name='predictions.csv')
            except Exception as e:
                st.error(f'Prediction failed: {e}')

st.markdown('---')
st.header('Notes')
st.write('- Train the model in your notebook, then run `save_artifacts("model_artifacts", clf, tfidf_vectorizer)` to create `clf.joblib` and `tfidf.joblib`.')
st.write('- Put the `model_artifacts` folder beside this app or set the full path in the sidebar.')
