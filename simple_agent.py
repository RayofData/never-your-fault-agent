from strands import Agent
from strands.models.ollama import OllamaModel


model = OllamaModel(
    host="http://localhost:11434",
    model_id="qwen3.5:4b",
)

agent = Agent(model=model)

agent("Explain League of Legends in two sentences.")