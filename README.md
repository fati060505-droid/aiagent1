# AI Research Agent

A simple research agent built with **CrewAI + Groq + DuckDuckGo + Streamlit**.
Enter a topic, and the agent searches the web and writes a structured report.

## Run locally
1. Create a virtual environment and activate it
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and put your Groq API key inside
4. `streamlit run app.py`

## Deploy on Streamlit Community Cloud
Add this in App settings -> Secrets:
```
GROQ_API_KEY = "your_groq_api_key_here"
```
Use Python 3.12 in Advanced settings.
