# Model Settings
OPENAI_MODEL = "gpt-4o-mini"
GROQ_MODEL_B = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_MODEL_C = "llama-3.3-70b-versatile"

# Sampling Settings
SAMPLE_SIZE = 20
RANDOM_STATE = None  # set to None for random samples each run

# Evaluator Settings
TEMPERATURE = 0.2
MAX_TOKEN_EVALUATORS = 250
MAX_TOKEN_REPORT_GENERATOR = 500


# Pipeline Settings
SLEEP_BETWEEN_SAMPLES = 2  # seconds between API calls

# Output Settings
RESULTS_FOLDER = "samples"

# Prompts
EVALUATOR_SYSTEM_PROMPT = "You are an expert at detecting AI-generated text. Analyze the given text and determine if it is AI-generated or human-written."

EVALUATOR_USER_PROMPT = ("Is this text AI generated or Human written?\n\n{text}\n\n"
                         "Reply with either HUMAN or AI followed by a brief explanation."
                         )

REPORT_GENERATOR_SYSTEM_PROMPT = (
    "You are an expert AI text detection analyst. You will receive opinions from 3 independent AI evaluators and a perplexity score."
    "Weigh all the evidence and output a final verdict of either HUMAN or AI, followed by a clear explanation of your reasoning."
)

REPORT_GENERATOR_USER_PROMPT = ("Based on the following evidence, what is your final verdict?"
                                "\n\nEvaluator A (OpenAI): {eval_a}\n\nEvaluator B (Groq/Gemma): {eval_b}\n\nEvaluator C (Groq/Llama): {eval_c}"
                                "\n\nPerplexity Score: {perp_score} (low score = likely AI, high score = likely Human)"
                                "\n\nProvide your final verdict and explanation."


                                )

AGENT_SYSTEM_PROMPT = ("You are the Report Generator. Use your tool and generate a final verdict."
                       " You MUST end your response with exactly 'Final Verdict: HUMAN' or 'Final Verdict: AI'."
                       )
TOOL_PROMPT = "Takes opinions from 3 evaluators and perplexity score and generates final verdict"
