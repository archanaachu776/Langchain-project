# To install: pip install tavily-python
from tavily import TavilyClient
client = TavilyClient("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")
response = client.search(
    query="best places to visit in winter"
)
print(response)