import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
   api_key=os.getenv("GEMINI_API_KEY")
)

def generate_answer(question, context):

   prompt = f"""
You are an AI Resume Assistant.

Answer the user's question using only the information
provided in the resume context.

Resume Context:
{context}

Question:
{question}

If the answer is not available in the resume context,
say that the information is not available in the resume.
"""

   response = client.models.generate_content(
       model="gemini-3.5-flash",
       contents=prompt
   )

   return response.text
