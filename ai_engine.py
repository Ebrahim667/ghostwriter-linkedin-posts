from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load GROQ_API_KEY from the .env file into the environment
load_dotenv()
# Single shared LLM instance used across the whole project — llama-3.3-70b via Groq for fast inference
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")


if __name__ == "__main__":
    response = llm.invoke("how can i escape the matrix?")
    print(response.content)