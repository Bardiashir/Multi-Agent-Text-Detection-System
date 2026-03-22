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


tool_evaluator_a = FunctionTool(
    evaluator_a, description="Evaluates text using OpenAI GPT-4o-mini and returns HUMAN or AI with explanation")
tool_evaluator_b = FunctionTool(
    evaluator_b, description="Evaluates text using OpenAI GPT-4o-mini and returns HUMAN or AI with explanation")
tool_evaluator_c = FunctionTool(
    evaluator_c, description="Evaluates text using OpenAI GPT-4o-mini and returns HUMAN or AI with explanation")
tool_report_generator = FunctionTool(
    report_generator, "Takes opinions from 3 evaluators and perplexity score and generates final verdict")


async def main(text: str):
    agent_a = AssistantAgent(model_client=model_client, name="EvaluatorA", tools=[
                             tool_evaluator_a], system_message="You are Evaluator A. Use your tool to evaluate the given text and report your findings.")
    agent_b = AssistantAgent(model_client=model_client, name="EvaluatorB", tools=[
                             tool_evaluator_b], system_message="You are Evaluator B. Use your tool to evaluate the given text and report your findings.")
    agent_c = AssistantAgent(model_client=model_client, name="EvaluatorC", tools=[
                             tool_evaluator_c], system_message="You are Evaluator C. Use your tool to evaluate the given text and report your findings.")
    agent_report_generator = AssistantAgent(model_client=model_client, name="ReportGenerator", tools=[
        tool_report_generator], system_message="You are Report Generator.\n\nWhen it is your turn, you MUST complete ALL steps below:\n\nSTEP 1: Call calculate_perplexity on the original text be aware that after that your turn is not over you have to complete all the steps.\n\nSTEP 2: In the SAME turn, call report_generator with the 3 evaluator verdicts and the perplexity score from STEP 1.\n\nSTEP 3: Output the final verdict and make sure to end your message with TERMINATE.\n\nDo NOT finish your turn until all 3 steps are done.")

    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat(termination_condition=termination, participants=[        #change it to something else
                               agent_a, agent_b, agent_c, agent_report_generator], max_turns=8)
    result = await team.run(task=text)
    return result.messages

res = asyncio.run(main(
    "is shaq online today?"))
#####
for message in res:
    if message.type in ("TextMessage", "ToolCallSummaryMessage"):
        print(f"\n{'='*50}")
        print(f"Agent:   {message.source}")
        print(f"Message: {message.content}")
