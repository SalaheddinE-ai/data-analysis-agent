# Data Analysis Agent (Deep Agents + Ollama + Streamlit)

## Setup
```bash
# 1. Ollama + a tool-calling model
ollama pull qwen3:8b          # or qwen2.5:7b / llama3.1:8b

# 2. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run
python make_sample_data.py    # optional sample CSV
streamlit run app.py
```

## Optional Slack
```bash
export SLACK_USER_TOKEN="xoxp-..."
export SLACK_CHANNEL_ID="C0123456ABC"
```
Then tick "Send results to Slack" in the sidebar.

## Notes
- Local backend = no sandbox: the agent executes shell commands on your machine inside `workspace/`.
  Run it in a container/VM for anything beyond experiments.
- Small local models struggle with long agent loops: use >=8B params, keep `num_ctx` >= 16k,
  temperature 0. If it loops, raise the model size (e.g. qwen3:14b).
- Outputs (scripts, plots) land in `workspace/outputs/`.
