
# AI-Powered Personal Career Navigator

A turnkey final-year project that helps students:
- Discover career paths
- Analyze skill gaps
- Generate learning plans
- Improve resumes (ATS-friendly)
- Practice mock interviews

## Quick Start

1. **Install Python 3.10+**  
2. **Clone or unzip this project**  
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
4. **Set your OpenAI API key**  
   - Windows (PowerShell):  
     ```powershell
     setx OPENAI_API_KEY "sk-..."
     ```
   - macOS/Linux (bash):  
     ```bash
     export OPENAI_API_KEY="sk-..."
     ```
5. **Run**  
   ```bash
   streamlit run app.py
   ```

## Deploy (Streamlit Community Cloud)
1. Push this folder to a **public GitHub repo**.
2. Go to Streamlit Community Cloud and deploy the repo.
3. Add `OPENAI_API_KEY` as a secret in the app settings.

## Tech Stack
- **UI**: Streamlit
- **AI**: OpenAI Chat Completions (gpt-4o-mini)
- **Data**: Simple CSV-based skill map
- **Lang**: Python 3

## Notes
- This project uses prebuilt LLM APIs (no heavy ML training). Perfect for a final-year showcase and real users.
- Replace placeholder data in `data/skills_dataset.csv` with your own research to boost quality.
