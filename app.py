import os

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from ddgs import DDGS

# Reads GROQ_API_KEY from a local .env file (for local use)
load_dotenv()

MODEL_NAME = "groq/openai/gpt-oss-120b"


def get_api_key():
    """Get the key from Streamlit Secrets (cloud) or .env (local)."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass  # no secrets file locally - that's fine
    return os.getenv("GROQ_API_KEY")


@tool("DuckDuckGo Web Search")
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo. Input is a search query string.
    Returns titles, links and short summaries of the top results."""
    try:
        results = DDGS().text(query, max_results=6)
    except Exception as e:
        return f"Search failed: {e}. Try a different query."
    if not results:
        return "No results found. Try a different query."
    lines = []
    for r in results:
        lines.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSummary: {r.get('body')}\n")
    return "\n".join(lines)


def run_research(topic: str, api_key: str) -> str:
    llm = LLM(model=MODEL_NAME, api_key=api_key, temperature=0.3)

    researcher = Agent(
        role="Research Analyst",
        goal=f"Research the topic '{topic}' using web search and write an accurate report.",
        backstory="You are a careful analyst who only uses facts found in search results "
                  "and always lists the sources you used.",
        tools=[web_search],
        llm=llm,
        max_iter=8,
        verbose=False,
    )

    task = Task(
        description=(
            f"Research this topic: {topic}\n\n"
            "Use the web search tool a few times with different queries. "
            "Then write a research report in Markdown."
        ),
        expected_output=(
            "A Markdown report with these sections: Title, Introduction, Key Findings "
            "(bullet points), Detailed Discussion, Conclusion, Sources (list of URLs used)."
        ),
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return result.raw


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Research Agent", page_icon="🔎")
st.title("🔎 AI Research Agent")
st.write("Enter a topic. The agent will search the web and write a research report.")

topic = st.text_input("Research topic", placeholder="e.g. Benefits of solar energy in Pakistan")

if st.button("Generate Research Report"):
    api_key = get_api_key()
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file (local) or Streamlit Secrets (cloud).")
    elif not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        # CrewAI/LiteLLM also look for this environment variable
        os.environ["GROQ_API_KEY"] = api_key
        try:
            with st.spinner("Researching... this can take 1-2 minutes."):
                report = run_research(topic.strip(), api_key)
            st.markdown(report)
            st.download_button("Download report (.md)", report, file_name="research_report.md")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
