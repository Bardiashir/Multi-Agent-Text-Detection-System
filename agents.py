import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from perplexity import calculate_perplexity
from config import OPENAI_MODEL, GROQ_MODEL_B, GROQ_MODEL_C, TEMPERATURE, MAX_TOKEN_EVALUATORS, MAX_TOKEN_REPORT_GENERATOR

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def evaluator_a(text: str) -> str:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
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
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKEN_EVALUATORS
    )
    return response.choices[0].message.content


def evaluator_b(text: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL_B,
        messages=[
            {
                "role": "system", "content": "You are an expert at detecting AI-generated text. Analyze the given text and determine if it is AI-generated or human-written. Reply with either HUMAN or AI followed by a brief explanation."
            },

            {
                "role": "user", "content": f"Is this text AI generated or Human written?\n\n{text}"
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKEN_EVALUATORS
    )
    return response.choices[0].message.content


def evaluator_c(text: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL_C,
        messages=[
            {
                "role": "system", "content": "You are an expert at detecting AI-generated text. Analyze the given text and determine if it is AI-generated or human-written. Reply with either HUMAN or AI followed by a brief explanation."
            },

            {
                "role": "user", "content": f"Is this text AI generated or Human written?\n\n{text}"
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKEN_EVALUATORS
    )
    return response.choices[0].message.content


def report_generator(text: str, eval_a: str, eval_b: str, eval_c: str) -> str:
    perp_score = calculate_perplexity(text)
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
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
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKEN_REPORT_GENERATOR
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    test_text = "The advancements in artificial intelligence have led to significant improvements in natural language processing capabilities."

    a = evaluator_a(test_text)
    b = evaluator_b(test_text)
    c = evaluator_c(test_text)
    score = calculate_perplexity(test_text)

    print("=== FINAL VERDICT ===")
    print(report_generator(a, b, c, score))
