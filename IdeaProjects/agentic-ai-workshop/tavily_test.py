import os
from tavily import TavilyClient

tavily = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"]
)

results = tavily.search(
    query="latest artificial intelligence developments",
    max_results=3
)

for result in results["results"]:
    print(result["title"])
    print(result["url"])
    print()