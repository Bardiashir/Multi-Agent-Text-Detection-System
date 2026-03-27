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
