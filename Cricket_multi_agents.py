
import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIResponses
from agno.tools.duckduckgo import DuckDuckGoTools


# --------------------------------------------------
# 1. Load API key
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Check your .env file."
    )


# --------------------------------------------------
# 2. Configure OpenAI model
# --------------------------------------------------

model = OpenAIResponses(
    id="gpt-6-luna",
    api_key=api_key
)


# --------------------------------------------------
# 3. Create Live Cricket Match Agent
# --------------------------------------------------

match_agent = Agent(
    name="Live Match Agent",
    role="Cricket live score specialist",
    model=model,
    tools=[DuckDuckGoTools()],
    instructions=[
        "Search for the latest available cricket match scores.",
        "Identify the teams, current score, wickets and overs.",
        "Summarize top performers and match situation.",
        "Mention the source and time of the information.",
        "Never invent live scores."
    ],
    markdown=True
)


# --------------------------------------------------
# 4. Create Player Statistics Agent
# --------------------------------------------------

player_agent = Agent(
    name="Player Stats Agent",
    role="Cricket player statistics specialist",
    model=model,
    tools=[DuckDuckGoTools()],
    instructions=[
        "Find recent cricket player statistics.",
        "Find batting and bowling statistics for the last 5 matches.",
        "Include runs, average, strike rate and wickets where available.",
        "Use markdown tables.",
        "Mention reliable sources."
    ],
    markdown=True
)


# --------------------------------------------------
# 5. Create Cricket News Agent
# --------------------------------------------------

news_agent = Agent(
    name="Cricket News Agent",
    role="Cricket news specialist",
    model=model,
    tools=[DuckDuckGoTools()],
    instructions=[
        "Find the latest cricket news.",
        "Highlight upcoming matches, injuries and tournament updates.",
        "Provide news headlines and their sources.",
        "Use recent and reliable information."
    ],
    markdown=True
)


# --------------------------------------------------
# 6. Create Multi-Agent Cricket Team
# --------------------------------------------------

cricket_team = Team(
    name="Cricket Analysis Team",
    model=model,
    members=[
        match_agent,
        player_agent,
        news_agent
    ],
    instructions=[
        "You are the leader of a cricket analysis team.",
        "Delegate live score research to the Live Match Agent.",
        "Delegate player statistics to the Player Stats Agent.",
        "Delegate cricket news research to the Cricket News Agent.",
        "Combine their findings into one comprehensive report.",
        "Use markdown headings and tables.",
        "Clearly mention when live information is unavailable."
    ],
    markdown=True
)


# --------------------------------------------------
# 7. Execute Cricket Analysis
# --------------------------------------------------

if __name__ == "__main__":

    cricket_team.print_response(
        """
        Find the latest available India vs New Zealand cricket
        match score.

        Also provide:
        1. Recent statistics of Abhishek Sharma.
        2. Latest cricket news.
        3. Upcoming cricket matches.

        Include sources and dates for the information.
        """,
        stream=True
    )
