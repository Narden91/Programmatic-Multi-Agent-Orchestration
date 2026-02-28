from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

try:
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )
    
    response = llm.invoke("Say hello!")
    print("✅ API Key funziona!")
    print(f"Risposta: {response.content}")
    
except Exception as e:
    print(f"❌ Errore: {str(e)}")