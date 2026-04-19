import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# 1. Page Configuration
st.set_page_config(page_title="HireXpert AI", page_icon="🤖", layout="wide")

# 2. PDF Text Extraction Function
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        return ""

# 3. Universal Job Database (Banks, Schools, Tech, etc.)
alternative_jobs = {
    "Bank Officer": "Banking operations, KYC, credit analysis, loan processing, and financial accounting.",
    "School Teacher": "Lesson planning, classroom management, student evaluation, and subject expertise.",
    "Customer Support": "Communication skills, problem-solving, ticketing systems, and client satisfaction.",
    "Sales Executive": "Lead generation, negotiation, marketing, and relationship management.",
    "Data Analyst": "Excel, SQL, data visualization, reporting, and analytical thinking.",
    "Administrative Assistant": "Scheduling, office management, documentation, and coordination."
}

# 4. Sidebar Branding
with st.sidebar:
    st.title("Settings")
    threshold = st.slider("Success Threshold (%)", 0, 100, 45)
    st.info("Resumes scoring below this will get a recommendation.")

# 5. Main UI
st.title("🤖 HireXpert: Universal AI Resume Screener")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Details")
    jd = st.text_area("Paste the Job Description (Bank, School, or Any Corp):", height=250)

with col2:
    st.subheader("Candidate Files")
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

# 6. Analysis Logic
if st.button("🚀 Analyze & Rank"):
    if jd and uploaded_files:
        with st.spinner('AI is calculating match scores...'):
            resume_texts = [extract_text_from_pdf(f) for f in uploaded_files]
            
            for i, text in enumerate(resume_texts):
                filename = uploaded_files[i].name
                st.write(f"### Results for: {filename}")
                
                # Calculate Main Score
                vectorizer = TfidfVectorizer(stop_words='english')
                matrix = vectorizer.fit_transform([jd, text])
                main_score = cosine_similarity(matrix[0:1], matrix[1:])[0][0]
                main_percent = round(main_score * 100, 2)
                
                # Visual Feedback
                if main_percent >= threshold:
                    st.success(f"**MATCH FOUND:** {main_percent}% Match Score")
                    st.progress(main_score)
                else:
                    st.error(f"**LOW MATCH:** {main_percent}% - Candidate may not be eligible for this specific role.")
                    
                    # Recommendation Logic
                    best_alt_job = ""
                    best_alt_score = 0
                    
                    for job_title, job_desc in alternative_jobs.items():
                        alt_matrix = vectorizer.fit_transform([job_desc, text])
                        alt_score = cosine_similarity(alt_matrix[0:1], alt_matrix[1:])[0][0]
                        if alt_score > best_alt_score:
                            best_alt_score = alt_score
                            best_alt_job = job_title
                    
                    if best_alt_score > 0.1: # Only suggest if there's some match
                        st.info(f"💡 **Recommendation:** Based on their skills, this candidate is a better fit for a **{best_alt_job}** role.")
                    else:
                        st.warning("💡 **Recommendation:** No close matches found in our standard database.")
                
                st.divider()
    else:
        st.warning("Please provide a job description and at least one resume to begin.")

# 7. Footer
st.caption("HireXpert School Project - Powered by NLP & Scikit-Learn")
