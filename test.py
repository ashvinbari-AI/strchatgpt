from dotenv import load_dotenv
from google import genai
import os

# Load .env
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

print("API Key Found:", api_key is not None)
print("API Key:", api_key[:10] + "..." if api_key else "None")

# Create client
client = genai.Client(api_key=api_key)

# List available models
for model in client.models.list():
    print(model.name) 

from dotenv import load_dotenv
import os

load_dotenv()

print("Tracing:", os.getenv("LANGSMITH_TRACING"))
print("Project:", os.getenv("LANGSMITH_PROJECT"))
print("API Key Exists:", os.getenv("LANGSMITH_API_KEY") is not None) 


