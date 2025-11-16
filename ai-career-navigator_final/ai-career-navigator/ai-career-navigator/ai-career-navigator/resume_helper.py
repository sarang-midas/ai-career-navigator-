
from openai import OpenAI

client = OpenAI()

def improve_resume(resume_text: str, target_role: str = "Data Analyst") -> str:
    prompt = f"""
    You are an ATS resume expert.
    Improve the following resume for the role: {target_role}.
    1) Give an ATS optimization checklist.
    2) Rewrite the summary, skills, and 2 sample bullet points with strong verbs + metrics.
    3) Suggest keywords and a clean one-page layout.
    Resume text:
    ---
    {resume_text}
    ---
    Format output in markdown.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"Be precise, practical, and recruiter-friendly."},
                  {"role":"user","content":prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content
