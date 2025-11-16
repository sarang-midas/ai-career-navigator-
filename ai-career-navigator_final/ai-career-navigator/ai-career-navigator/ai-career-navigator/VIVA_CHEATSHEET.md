
# Viva Cheat Sheet

**What problem do you solve?**  
Students lack clear path to job readiness. We give tailored role choices, show missing skills, give a plan, fix resume, and practice interviews.

**Why Streamlit?**  
Rapid UI, low code, easy to deploy for demo.

**Why OpenAI (vs training my own model)?**  
Time and compute constraints. Using reliable APIs ensures quality. Future versions can replace with open-source models.

**Data used?**  
A curated CSV mapping roles to typical skills. Can be expanded based on research.

**How skill gaps computed?**  
Tokenize user skills and compare against role's required skills set → coverage % and missing list.

**Security/Privacy?**  
API key stored as environment variable. No PII stored in this demo.

**Limitations?**  
- Dependent on LLM for text quality
- CSV coverage is limited
- No persistent user accounts in this version

**Future Enhancements?**  
- Database + auth
- PDF resume parsing
- Real job scraping & ranking
- Export PDF of plans
