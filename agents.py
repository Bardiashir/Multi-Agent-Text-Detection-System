import os
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
from groq import Groq
from perplexity import calculate_perplexity

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def evaluator_a(text: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at detecting AI-generated text. Analyze the given text and determine if it is AI-generated or human-written. Reply with either HUMAN or AI followed by a brief explanation."
            },
            {
                "role": "user",
                "content": f"Is this text AI generated or Human written?\n\n{text}"
            }
        ],
        temperature=0.2,
        max_tokens=250
    )
    return response.choices[0].message.content


def evaluator_b(text: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system", "content": "You are an expert at detecting AI-generated text. Analyze the given text and determine if it is AI-generated or human-written. Reply with either HUMAN or AI followed by a brief explanation."
            },

            {
                "role": "user", "content": f"Is this text AI generated or Human written?\n\n{text}"
            }
        ],
        temperature=0.2,
        max_tokens=250
    )
    return response.choices[0].message.content


def evaluator_c(text: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system", "content": "You are an expert at detecting AI-generated text. Analyze the given text and determine if it is AI-generated or human-written. Reply with either HUMAN or AI followed by a brief explanation."
            },

            {
                "role": "user", "content": f"Is this text AI generated or Human written?\n\n{text}"
            }
        ],
        temperature=0.2,
        max_tokens=250
    )
    return response.choices[0].message.content


def report_generator(text :str , eval_a: str, eval_b: str, eval_c: str) -> str:
    perp_score = calculate_perplexity(text)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert AI text detection analyst. You will receive opinions from 3 independent AI evaluators and a perplexity score. Weigh all the evidence and output a final verdict of either HUMAN or AI, followed by a clear explanation of your reasoning."
            },
            {
                "role": "user",
                
                "content": f"Based on the following evidence, what is your final verdict?\n\nEvaluator A (OpenAI): {eval_a}\n\nEvaluator B (Groq/Gemma): {eval_b}\n\nEvaluator C (Groq/Llama): {eval_c}\n\nPerplexity Score: {perp_score} (low score = likely AI, high score = likely Human)\n\nProvide your final verdict and explanation."
            }
        ],
        temperature=0.2,
        max_tokens=400
    )
    return response.choices[0].message.content + "\n\nTERMINATE"




###

if __name__ == "__main__":
    test_text = "The advancements in artificial intelligence have led to significant improvements in natural language processing capabilities."
    
    a = evaluator_a(test_text)
    b = evaluator_b(test_text)
    c = evaluator_c(test_text)
    score = calculate_perplexity(test_text)
    
    print("=== FINAL VERDICT ===")
    print(report_generator(a, b, c, score))