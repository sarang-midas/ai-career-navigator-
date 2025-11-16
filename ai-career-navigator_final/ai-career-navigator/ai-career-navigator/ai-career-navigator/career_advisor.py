
import os
import pandas as pd
import streamlit as st

# Try to import Groq first. If Groq is not installed or no API key is set,
# we'll try OpenAI next. If neither is available, fall back to a local reply.
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Read credentials from environment variables. Users should set these locally.
_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_MODEL = "llama-3.1-8b-instant"

# Initialize clients depending on availability and credentials
groq_client = None
openai_client = None

if Groq is not None and _GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=_GROQ_API_KEY)
    except Exception:
        groq_client = None

if OpenAI is not None:
    try:
        # OpenAI client will pick up environment variables like OPENAI_API_KEY
        openai_client = OpenAI()
    except Exception:
        openai_client = None


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "skills_dataset.csv")

def load_role_skills():
    try:
        df = pd.read_csv(DATA_PATH)
        # Normalize columns
        df['role'] = df['role'].str.strip().str.lower()
        df['skills'] = df['skills'].fillna('').astype(str)
        return df
    except Exception as e:
        return pd.DataFrame(columns=["role","skills"])

def _chat(prompt: str, model: str = None) -> str:
    """Call the Groq client if available, otherwise return a helpful fallback.

    The fallback provides a short, deterministic response so the UI remains usable
    even without external model access.
    """
    model = model or _MODEL
    # Prefer groq_client, then openai_client, then fallback
    if groq_client is not None:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":"You are an expert AI career mentor for students."},
                      {"role":"user","content":prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content

    if openai_client is not None:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are an expert AI career mentor for students."},
                      {"role":"user","content":prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content

    # Fallback: produce a concise, safe markdown answer derived from the prompt.
    # This keeps the app functional offline or without credentials.
    header = "**(Fallback) Career Advisor — overview**\n\n"
    body = (
        "I don't have access to Groq or OpenAI (no API keys or packages installed). "
        "Here's a short example output based on the provided profile prompt.\n\n"
    )
    example = (
        "- Suggested roles: Data Analyst, Business Analyst, ML Engineer\n"
        "- Why fit: Matches skills and interests; entry-level friendly\n"
        "- Must-have skills: Python, SQL, Data Visualization\n"
        "- Starter projects: Sales dashboard, EDA on public dataset, simple ML model\n"
        "- Compensation (INR): 3.0L - 7.0L (entry-level, approximate)\n"
    )
    return header + body + example

def get_career_paths(skills: str, interests: str, education: str, experience: str) -> str:
    prompt = f"""
    Profile:
    - Education: {education}
    - Experience: {experience}
    - Skills: {skills}
    - Interests: {interests}

    Task: Suggest 4-6 high-demand career paths suitable for the profile in India, with for each:
    - What the role does (1-2 lines)
    - Why it's a fit for this profile
    - 3 must-have skills
    - Typical starter projects
    - Entry-level compensation range (INR, realistic)
    Format as markdown with headings and bullet points.
    """
    return _chat(prompt)

def analyze_skill_gaps(user_skills_csv: str, target_role: str, role_skills_df: pd.DataFrame):
    user = {s.strip().lower() for s in user_skills_csv.split(",") if s.strip()}
    role = (target_role or "data analyst").strip().lower()
    # find matching role rows and collect skills
    rows = role_skills_df[role_skills_df['role'] == role]
    required = set()
    for _, r in rows.iterrows():
        required.update({x.strip().lower() for x in str(r['skills']).split(",") if x.strip()})
    # If dataset has nothing, provide a minimal default
    if not required:
        required = {"python","sql","statistics","excel","data visualization","etl","power bi","tableau"}
    gaps = sorted(list(required - user))
    matches = sorted(list(required & user))
    coverage = 0 if not required else int((len(matches) / len(required)) * 100)
    return {
        "target_role": role,
        "have_skills": matches,
        "missing_skills": gaps,
        "coverage_percent": coverage
    }

def get_learning_plan(skills: str, interests: str, duration: str, target_role: str) -> str:
    prompt = f"""
    Create a step-by-step learning plan to become a strong {target_role or 'Data Analyst'} in {duration}.
    Student's current skills: {skills}
    Interests: {interests}
    Structure:
    - Phases with timelines (Week 1-2, etc.)
    - Exact outcomes and checkpoints
    - Free resources (YouTube, docs, datasets, practice sites)
    - 3 portfolio projects with acceptance criteria and how to present results
    Keep it concise, actionable, and India-centric.
    """
    return _chat(prompt)
