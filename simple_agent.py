from strands import Agent
from strands.models.ollama import OllamaModel


model = OllamaModel(
    host="http://localhost:11434",
    model_id="phi4-mini:3.8b",
)

agent = Agent(model=model)

agent("Explain League of Legends in two sentences.")