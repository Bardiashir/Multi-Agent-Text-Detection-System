import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from agents import evaluator_a, evaluator_b, evaluator_c, report_generator
from data import load_sample
from datetime import datetime
import logging
from config import OPENAI_MODEL, SAMPLE_SIZE, SLEEP_BETWEEN_SAMPLES

logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)

load_dotenv()


model_client = OpenAIChatCompletionClient(
    api_key=os.getenv("OPENAI_API_KEY"), model=OPENAI_MODEL)


tool_report_generator = FunctionTool(
    report_generator, "Takes opinions from 3 evaluators and perplexity score and generates final verdict")


async def main(text: str):
    verdict_a = evaluator_a(text)
    verdict_b = evaluator_b(text)
    verdict_c = evaluator_c(text)
    task = (
        f"Original text: {text}\n\n"
        f"Evaluator A verdict: {verdict_a}\n"
        f"Evaluator B verdict: {verdict_b}\n"
        f"Evaluator C verdict: {verdict_c}\n\n"
        "Now call your tool and generate the final report."
    )
    agent_report_generator = AssistantAgent(model_client=model_client, name="ReportGenerator", tools=[
        tool_report_generator], system_message=("You are the Report Generator. Use your tool and generate a final verdict. You MUST end your response with exactly 'Final Verdict: HUMAN' or 'Final Verdict: AI'."))

    result = await agent_report_generator.run(task=task)
    last_message = result.messages[-1].content
    last_upper = last_message.upper()
    if "FINAL VERDICT: HUMAN" in last_upper:
        return 1
    elif "FINAL VERDICT: AI" in last_upper:
        return 0
    else:
        return -1


async def run_all(samples):
    pred_labels = []
    for i, row in samples.iterrows():
        print(f"Running sample {i+1}/{SAMPLE_SIZE}...")
        pred = await main(row["text"])
        pred_labels.append(pred)
        await asyncio.sleep(SLEEP_BETWEEN_SAMPLES)
    return pred_labels

samples = load_sample(n=SAMPLE_SIZE)
pred_labels = asyncio.run(run_all(samples))


samples["predicted_labels"] = pred_labels

valid = samples[samples["predicted_labels"] != -1]
accuracy = accuracy_score(valid["label"], valid["predicted_labels"])
print(f"Accuracy is {accuracy * 100:.2f}%")
print(classification_report(
    valid["label"], valid["predicted_labels"], target_names=["AI", "HUMAN"]))
print(f"Valid samples: {len(valid)}/{SAMPLE_SIZE}")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

csv_path = f"samples/run_{timestamp}.csv"
samples.to_csv(csv_path, index=False)

summary_path = f"samples/run_{timestamp}_summary.txt"
with open(summary_path, "w") as f:
    f.write(f"Run: {timestamp}\n")
    f.write(f"Valid samples: {len(valid)}/20\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write(classification_report(
        valid["label"], valid["predicted_labels"], target_names=["AI", "HUMAN"]))

print(f"Results saved to {csv_path}")
