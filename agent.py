import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool
from langchain_ollama import ChatOllama

WORKSPACE = Path(__file__).parent / "workspace"
DATA_DIR = WORKSPACE / "data"
OUTPUT_DIR = WORKSPACE / "outputs"
for d in (DATA_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are a data analysis agent.

Environment:
- Your working directory contains `data/` (input files) and `outputs/` (everything you produce).
- Inside Python scripts, use RELATIVE paths: `data/<file>.csv` and `outputs/<name>.png`.
- Use the `execute` tool to run scripts (python is available with pandas, matplotlib, seaborn).

Workflow:
1. Use write_todos to plan the analysis steps.
2. Inspect the data first (columns, dtypes, missing values, head).
3. Write a Python script to `analysis.py`, run it, fix errors if any.
4. Save plots as PNG in `outputs/` (matplotlib backend must be Agg; call plt.savefig, never plt.show).
5. Finish with a concise written report: key metrics, insights, and the plot file names.
Never invent numbers: report only what the script printed."""


def make_slack_tool():
    """Optional Slack tool. Token stays outside the agent's shell environment."""
    from slack_sdk import WebClient

    client = WebClient(token=os.environ["SLACK_USER_TOKEN"])
    channel = os.environ["SLACK_CHANNEL_ID"]

    @tool(parse_docstring=True)
    def slack_send_message(text: str, file_path: str | None = None) -> str:
        """Send a message to Slack, optionally with an attached file.

        Args:
            text: Text content of the message.
            file_path: Path of an attachment relative to the working directory (e.g. outputs/plot.png).
        """
        if not file_path:
            client.chat_postMessage(channel=channel, text=text)
        else:
            full = (WORKSPACE / file_path.lstrip("/")).resolve()
            if WORKSPACE.resolve() not in full.parents or not full.exists():
                return f"Error: file not found in workspace: {file_path}"
            client.files_upload_v2(channel=channel, file=str(full), initial_comment=text)
        return "Message sent."

    return slack_send_message


def build_agent(model: str, base_url: str, temperature: float, num_ctx: int,
                checkpointer, use_slack: bool = False):
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,  # Ollama's default context is small; agents need 16k+
    )
    tools = [make_slack_tool()] if use_slack else []
    # NOTE: LocalShellBackend runs commands directly on your machine (no sandbox).
    backend = LocalShellBackend(root_dir=WORKSPACE, virtual_mode=True, timeout=120, inherit_env=True)
    return create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        checkpointer=checkpointer,
        middleware=[TodoListMiddleware()],
    )
