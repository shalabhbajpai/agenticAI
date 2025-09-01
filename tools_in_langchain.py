#from langchain_community.tools import DuckDuckGoSearchRun
from duckduckgo_search import DDGS

search_tool = DDGS()

keywords = "top news in India today"

ddgs_results_gen = search_tool.text(
    keywords, 
    region="wt-wt",  # "wt-wt" for worldwide region
    safesearch="Off", # "Off", "Moderate", or "Strict"
    timelimit=None,   # "d" (day), "w" (week), "m" (month), "y" (year)
    max_results=5     # Maximum number of results to retrieve
)

#result = search_tool.invoke("top news in India today?")

print(f"Search results for '{keywords}':")
for i, r in enumerate(ddgs_results_gen):
    print(f"{i+1}. Title: {r.get('title')}")
    print(f"   Snippet: {r.get('snippet')}")
    print(f"   URL: {r.get('href')}\n")