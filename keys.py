import os

# Load environment variables from .env file
with open(".env", "r") as file:
    for line in file:
        if line.strip() and not line.startswith("#"):
            key, value = line.strip().split("=")
            os.environ[key] = value

# Access environment variables
openAI_API = os.getenv("OPENAI_API_KEY")
pinecone_API = os.getenv("PINECONE_API_KEY")
pinecone_ENV = os.getenv("PINECONE_API_ENV")
flask_secret_key = os.getenv("FLASK_SECRET_KEY")
