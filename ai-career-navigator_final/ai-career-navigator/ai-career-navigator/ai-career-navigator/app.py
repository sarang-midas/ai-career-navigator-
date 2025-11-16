
import streamlit as st
from career_advisor import get_career_paths, get_learning_plan, analyze_skill_gaps, load_role_skills
from resume_helper import improve_resume
from interview_bot import run_mock_interview

st.set_page_config(page_title="AI Career Navigator", layout="wide")
st.title("🚀 AI-Powered Personal Career Navigator")
st.caption(": career mapping • skill gaps • learning plans • resume • mock interview")

# Sidebar: user profile
with st.sidebar:
    st.header("Your Profile")
    name = st.text_input("Name", value="")
    education = st.selectbox("Education Level", ["B.Tech", "M.Tech", "B.Sc", "M.Sc", "Diploma", "Other"])
    experience = st.selectbox("Experience", ["Fresher", "0-1 years", "1-2 years", "2+ years"])
    skills = st.text_area("Your Skills (comma separated)", placeholder="Python, SQL, Machine Learning, Excel")
    interests = st.text_area("Your Interests", placeholder="Data Science, Computer Vision, Web Development")
    target_role = st.text_input("Target Role (optional)", value="Data Analyst")

role_skills = load_role_skills()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Career Paths", "Skill Gaps + Plan", "Learning Plan", "Resume Helper", "Mock Interview"
])

with tab1:
    st.subheader("📌 Recommended Career Paths")
    if st.button("Find Career Paths"):
        with st.spinner("Thinking..."):
            output = get_career_paths(skills, interests, education, experience)
        st.markdown(output)

with tab2:
    st.subheader("🧭 Skill Gap Analysis")
    st.caption("Compares your skills with common requirements for your selected role.")
    if st.button("Analyze Skill Gaps"):
        with st.spinner("Analyzing..."):
            results = analyze_skill_gaps(skills, target_role, role_skills)
        st.json(results)

with tab3:
    st.subheader("📚 Learning Plan")
    duration = st.selectbox("Plan Duration", ["90 days", "6 months", "1 year"], index=0)
    if st.button("Generate Learning Plan"):
        with st.spinner("Generating plan..."):
            plan = get_learning_plan(skills, interests, duration, target_role)
        st.markdown(plan)

with tab4:
    st.subheader("📝 Resume Improvement (ATS-friendly)")
    uploaded_file = st.file_uploader("Upload your resume (.txt or .pdf text pasted)", type=["txt"])
    resume_text = ""
    if uploaded_file is not None:
        resume_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.success("Resume loaded!")
    resume_text = st.text_area("Or paste your resume text here", value=resume_text, height=200)
    if st.button("Improve My Resume"):
        if not resume_text.strip():
            st.warning("Please upload or paste your resume text first.")
        else:
            with st.spinner("Reviewing resume..."):
                tips = improve_resume(resume_text, target_role)
            st.markdown(tips)

with tab5:
    st.subheader("💼 Mock Interview Coach")
    st.caption("Practice role-specific HR + technical questions and receive feedback.")
    run_mock_interview(target_role or "Data Analyst")
