# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# PAGE CONFIG
st.set_page_config(page_title="Instant Resume Matcher", layout="wide")
DATA_FILE = "processed_data.pkl"

# --- FAST LOADER ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    with open(DATA_FILE, 'rb') as f:
        return pickle.load(f)

# --- MAIN APP ---
st.title("⚡ Instant Resume Matcher (Enterprise)")

data = load_data()

if data is None:
    st.error(f"❌ Data file '{DATA_FILE}' not found. Please run 'python preprocess.py' first!")
    st.stop()

# Unpack data
df_resumes = data["resumes"]
resume_vectors = data["resume_vectors"]
df_jobs = data["jobs"]
job_vectors = data["job_vectors"]

# SIDEBAR STATS
st.sidebar.header("Data Status")
st.sidebar.success(f"📂 Loaded {len(df_resumes)} Resumes")
st.sidebar.info(f"📋 Loaded {len(df_jobs)} Jobs")

# Check Categories
categories = df_resumes['Category'].unique()
st.sidebar.markdown("### Categories Found:")
st.sidebar.write(categories)

# INTERFACE
tab1, tab2 = st.tabs(["🔍 Match Existing Jobs", "✍️ Paste Custom Job"])

with tab1:
    st.markdown("### Select a Job Role")
    
    # Searchable Dropdown
    job_titles = df_jobs['Job_Title'].unique()
    selected_job_title = st.selectbox("Search Job Title:", job_titles)
    
    # Filter sliders
    top_k = st.slider("Show Top N Candidates", 3, 20, 5)
    
    if st.button("Find Best Matches"):
        # Find the index of the selected job
        job_idx = df_jobs[df_jobs['Job_Title'] == selected_job_title].index[0]
        
        # INSTANT MATH: Compare pre-computed vectors
        target_vector = job_vectors[job_idx].reshape(1, -1)
        
        # Calculate similarity
        scores = cosine_similarity(target_vector, resume_vectors)[0]
        
        # Get Top K indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        st.subheader(f"Results for: {selected_job_title}")
        
        for idx in top_indices:
            score = scores[idx]
            resume = df_resumes.iloc[idx]
            
            # Color code the score
            if score > 0.60: color = "green"
            elif score > 0.45: color = "orange"
            else: color = "red"
            
            with st.container():
                c1, c2, c3 = st.columns([1, 2, 5])
                c1.markdown(f":{color}[**{score*100:.1f}%**]")
                c2.write(f"**{resume['Category']}**")
                c2.caption(resume['Filename'])
                with c3:
                    with st.expander("📄 View Snippet"):
                        st.write(resume['Text'][:800] + "...")
                st.divider()

with tab2:
    st.write("Paste a new job description to find matches instantly.")
    custom_jd = st.text_area("Job Description", height=200)
    
    if st.button("Match Custom Job"):
        if custom_jd:
            # Load model just for this one query
            model = SentenceTransformer('all-MiniLM-L6-v2')
            custom_vector = model.encode([custom_jd])
            
            # Compare
            scores = cosine_similarity(custom_vector, resume_vectors)[0]
            top_indices = np.argsort(scores)[::-1][:10]
            
            st.write("### Top Candidates")
            for idx in top_indices:
                score = scores[idx]
                resume = df_resumes.iloc[idx]
                st.markdown(f"**{score*100:.1f}%** | {resume['Category']} | {resume['Filename']}")