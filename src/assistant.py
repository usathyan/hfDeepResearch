import argparse
import os
import threading
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from huggingface_hub import login
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from tqdm import tqdm

from smolagents import (
    CodeAgent,
    OpenAIServerModel,
    Model,
    ToolCallingAgent,
)

MANAGED_AGENT_PROMPT = """You are a helpful agent named '{name}'.
You have been submitted this task by your manager.
---
Task:
{task}
---
You're helping your manager solve a wider task: so do not just provide a one-line answer immediately.
First, you MUST use the search_agent to gather information about the task.
Only after you have gathered sufficient information from the search_agent, you can provide the final answer.

Your final_answer WILL HAVE to contain these parts:
### 1. Task outcome (short version):
### 2. Task outcome (extremely detailed version):
### 3. Additional context (if relevant):

Put all these in your final_answer tool, everything that you do not pass as an argument to final_answer will be lost.
And even if your task resolution is not successful, please return as much context as possible, so that your manager can act upon this feedback.
"""

MODEL="gpt-3.5-turbo-1106"

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
    "biopython",
    "matplotlib",
    "seaborn",
    "plotly",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    return parser.parse_args()

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

def create_agent_hierarchy(model: Model):
    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    WEB_TOOLS = [
        SearchInformationTool(browser),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )

    manager_agent = ToolCallingAgent(
        model=model,
        tools=[visualizer, ti_tool],
        max_steps=12,
        verbosity_level=2,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )
    return manager_agent

def answer_single_question(question: str):
    model = OpenAIServerModel(
        model_id=MODEL,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ["SMOL_KEY"],
    )
    document_inspection_tool = TextInspectorTool(model, 100000)

    agent = create_agent_hierarchy(model)

    augmented_question = (
        """You have one question to answer. It is paramount that you provide a correct answer.
    Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
    Run verification steps if that's needed, you must make sure you find the correct answer!
    Here is the task:
    """
        + question
    )

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        final_result = agent.run(task=augmented_question)
        output = str(final_result)
        agent_memory = agent.write_memory_to_messages(summary_mode=True)

        # final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)
        from scripts.reformulator import prepare_response
        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)

        intermediate_steps = [str(step) for step in agent.memory.steps]

        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps]) else False
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        print("Error on ", augmented_question, e)
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example = {
        "agent_name": model.model_id,
        "question": question,
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
    }
    print(f"Answer: {output}")


def main():
    args = parse_args()
    print(f"Starting run with arguments: {args}")
    answer_single_question(args.question)


if __name__ == "__main__":
    main()
