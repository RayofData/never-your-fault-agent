from strands import Agent
from strands.models.ollama import OllamaModel
from strands_tools import http_request


model = OllamaModel(
    host="http://localhost:11434",
    model_id="phi4-mini:3.8b",
    temperature=0,
)

agent = Agent(
    model=model,
    tools=[http_request],
)


result = agent(
    """
    You must use the http_request tool before answering.

    Call http_request with:
    - method: GET
    - url: https://httpbin.org/get

    After the tool returns, report the URL found in the response.
    Do not answer from memory and do not claim that web access is unavailable.
    """
)

print(result)