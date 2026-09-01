import os

from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
   api_key=os.getenv("GEMINI_API_KEY")
)

def generate_embedding(text):
   response = client.models.embed_content(
       model="gemini-embedding-001",
       contents=text
   )

   return response.embeddings[0].values
