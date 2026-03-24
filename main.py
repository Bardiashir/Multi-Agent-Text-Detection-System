import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from agents import evaluator_a, evaluator_b, evaluator_c, report_generator

load_dotenv()


model_client = OpenAIChatCompletionClient(
    api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")


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
        "Now call your tool and generate the final report. End with TERMINATE."
    )
    agent_report_generator = AssistantAgent(model_client=model_client, name="ReportGenerator", tools=[
        tool_report_generator], system_message="You are the Report Generator. Use your tool with the 3 evaluator verdicts and generate a final verdict of HUMAN or AI. End your response with TERMINATE.")

    termination = TextMentionTermination("TERMINATE")
    
    result = await agent_report_generator.run(task=task)
    return result.messages

res = asyncio.run(main(
    "I'm so sorry but i wont be able to attend the class today, since i have a court going on"))
#####
for message in res:
    if message.type in ("TextMessage", "ToolCallSummaryMessage"):
        print(f"\n{'='*50}")
        print(f"Agent:   {message.source}")
        print(f"Message: {message.content}")
