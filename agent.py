import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool('get_topic', description='Return the name of the topic', return_direct=False)
def get_topic(topic: str):
    response = "India"
    return response

agent = create_agent(
        model = "google_genai:gemini-3.5-flash-lite",
        tools = [get_topic],
        system_prompt = "You are a helpful model who writes a comprehensive essay on the given topic"
)

response = agent.invoke({'messages': [{'role': 'user', 'content': 'Write an essay on topic India'}]})

print(response)
print(response["messages"][-1].content)
