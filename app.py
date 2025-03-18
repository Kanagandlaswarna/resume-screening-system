import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)  # ✅ Directly pass uploaded file
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text


def extract_text_from_docx(docx_file):
    return docx2txt.process(docx_file)  # ✅ Directly pass uploaded file


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

def calculate_similarity(job_desc, resumes):
    vectorizer = TfidfVectorizer()
    all_texts = [job_desc] + resumes
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    job_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(job_vector, resume_vectors)[0]
    return scores

def main():
    st.title("AI-Powered Resume Screening and Ranking System")
    job_desc = st.text_area("Enter Job Description:")
    uploaded_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    
    if st.button("Analyze Resumes"):
        if job_desc and uploaded_files:
            resumes = []
            resume_names = []
            for uploaded_file in uploaded_files:
                resume_names.append(uploaded_file.name)  # Store file name
                if uploaded_file.type == "application/pdf":
                   text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_text_from_docx(uploaded_file)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue

                resumes.append(preprocess_text(text))  # Preprocess and store text

            
            job_desc = preprocess_text(job_desc)
            scores = calculate_similarity(job_desc, resumes)
            ranked_resumes = sorted(zip(resume_names, scores), key=lambda x: x[1], reverse=True)
            
            results_df = pd.DataFrame(ranked_resumes, columns=["Resume Name", "Match Score"])
            results_df["Match Score"] = results_df["Match Score"].round(4)
            st.dataframe(results_df)
        else:
            st.warning("Please provide both Job Description and Resumes!")

if __name__ == "__main__":
    main()
