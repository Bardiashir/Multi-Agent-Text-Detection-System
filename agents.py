import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from perplexity import calculate_perplexity
from config import OPENAI_MODEL, GROQ_MODEL_B, GROQ_MODEL_C, TEMPERATURE, MAX_TOKEN_EVALUATORS, MAX_TOKEN_REPORT_GENERATOR, EVALUATOR_SYSTEM_PROMPT, EVALUATOR_USER_PROMPT, REPORT_GENERATOR_SYSTEM_PROMPT, REPORT_GENERATOR_USER_PROMPT

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def evaluator_a(text: str) -> str:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": EVALUATOR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": EVALUATOR_USER_PROMPT.format(text=text)
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
                "role": "system", "content": EVALUATOR_SYSTEM_PROMPT
            },

            {
                "role": "user", "content": EVALUATOR_USER_PROMPT.format(text=text)
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
                "role": "system", "content": EVALUATOR_SYSTEM_PROMPT
            },
            {
                "role": "user", "content": EVALUATOR_USER_PROMPT.format(text=text)
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
                "content": REPORT_GENERATOR_SYSTEM_PROMPT
            },
            {
                "role": "user",

                "content": REPORT_GENERATOR_USER_PROMPT.format(
                    eval_a=eval_a,
                    eval_b=eval_b,
                    eval_c=eval_c,
                    perp_score=perp_score
                )
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
