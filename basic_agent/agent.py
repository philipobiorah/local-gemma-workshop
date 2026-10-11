import os
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

# Model configuration
model_tag = os.environ.get("GEMMA_MODEL", "gemma4:e2b")
ollama_host = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434")

local_model = LiteLlm(
    model=f"ollama_chat/{model_tag}",
    api_base=ollama_host,
    temperature=0.2,
    num_ctx=8192,
)

# Root agent definition
root_agent = Agent(
    model=local_model,
    name="root_agent",
    description="A minimalist offline assistant using Gemma 4.",
    instruction="You are a helpful and concise local assistant."
)