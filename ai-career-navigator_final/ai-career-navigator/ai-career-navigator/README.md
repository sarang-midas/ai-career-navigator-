# Resume Analyzer + Job Matcher (Pro)

**Streamlit** app with attractive UI and robust features:
- Upload & analyze resumes (PDF/DOCX/TXT)
- Skill detection and gaps
- TF‑IDF job matching (built‑in jobs or pasted JD)
- Polished charts and metrics
- One‑click PDF reports
- History (SQLite) with score distribution
- Admin module to manage skills & jobs

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Optional env: `ADMIN_PASS=yourpass`

Uploads stored in `uploads/`. History in `history.sqlite3`.
