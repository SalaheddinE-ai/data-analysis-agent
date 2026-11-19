"""Streamlit UI for the local data analysis deep agent."""
import os
import time
import uuid

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from agent import DATA_DIR, OUTPUT_DIR, build_agent

st.set_page_config(page_title="Data Analysis Agent", page_icon="📊", layout="wide")
st.title("📊 Data Analysis Agent")
st.caption("Deep Agents + local Ollama model. Everything runs on your machine.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama model", "qwen3:8b", help="Must support tool calling (qwen3, qwen2.5, llama3.1...)")
    base_url = st.text_input("Ollama URL", "http://localhost:11434")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    num_ctx = st.select_slider("Context window", [8192, 16384, 32768, 65536], value=16384)
    use_slack = st.checkbox("Send results to Slack", value=False,
                            help="Needs SLACK_USER_TOKEN and SLACK_CHANNEL_ID env vars")
    st.divider()
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded:
        (DATA_DIR / uploaded.name).write_bytes(uploaded.getvalue())
        st.session_state["csv_name"] = uploaded.name
        st.success(f"Saved to data/{uploaded.name}")
    if st.button("🔄 New conversation"):
        for k in ("agent", "agent_cfg", "thread_id", "history"):
            st.session_state.pop(k, None)
        st.rerun()

# ---------------- State ----------------
st.session_state.setdefault("history", [])
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("checkpointer", InMemorySaver())

cfg = (model, base_url, temperature, num_ctx, use_slack)
if st.session_state.get("agent_cfg") != cfg:
    if use_slack and not (os.getenv("SLACK_USER_TOKEN") and os.getenv("SLACK_CHANNEL_ID")):
        st.sidebar.error("Set SLACK_USER_TOKEN and SLACK_CHANNEL_ID first.")
        st.stop()
    st.session_state["agent"] = build_agent(model, base_url, temperature, num_ctx,
                                            st.session_state["checkpointer"], use_slack)
    st.session_state["agent_cfg"] = cfg

if "csv_name" in st.session_state:
    import pandas as pd
    with st.expander(f"Preview: {st.session_state['csv_name']}", expanded=False):
        st.dataframe(pd.read_csv(DATA_DIR / st.session_state["csv_name"]).head(50))


def render_plots(paths):
    for p in paths:
        st.image(str(p), caption=p.name, use_container_width=True)


def png_snapshot():
    return {p: p.stat().st_mtime for p in OUTPUT_DIR.glob("*.png")}


# ---------------- History ----------------
for m in st.session_state["history"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_plots([OUTPUT_DIR / n for n in m.get("plots", []) if (OUTPUT_DIR / n).exists()])

# ---------------- Chat ----------------
default = "Analyze the uploaded CSV, compute key metrics, and generate a beautiful plot."
prompt = st.chat_input(default)
if prompt:
    if "csv_name" not in st.session_state:
        st.warning("Upload a CSV first.")
        st.stop()
    full_prompt = f"{prompt}\n\n(The file to analyze is: data/{st.session_state['csv_name']})"
    st.session_state["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    before = png_snapshot()
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}, "recursion_limit": 100}

    with st.chat_message("assistant"):
        status = st.status("Agent working...", expanded=True)
        todo_box = st.empty()
        final_text, seen = "", 0
        try:
            for state in st.session_state["agent"].stream(
                {"messages": [{"role": "user", "content": full_prompt}]},
                config, stream_mode="values",
            ):
                msgs = state["messages"]
                # skip messages already rendered (history from earlier turns on first snapshot)
                if seen == 0:
                    seen = len(msgs)
                    continue
                for msg in msgs[seen:]:
                    if msg.type == "ai":
                        if msg.content and isinstance(msg.content, str):
                            status.markdown(f"💬 {msg.content}")
                            final_text = msg.content
                        for tc in msg.tool_calls:
                            args = tc["args"]
                            if tc["name"] == "write_file":
                                status.markdown(f"📝 **write_file** `{args.get('file_path')}`")
                                status.code(str(args.get("content", ""))[:3000], language="python")
                            elif tc["name"] == "execute":
                                status.markdown("⚙️ **execute**")
                                status.code(args.get("command", ""), language="bash")
                            else:
                                status.markdown(f"🔧 **{tc['name']}** `{str(args)[:200]}`")
                    elif msg.type == "tool":
                        status.markdown(f"↩️ _{msg.name}_ result:")
                        status.code(str(msg.content)[:1500])
                seen = len(msgs)
                todos = state.get("todos")
                if todos:
                    icons = {"completed": "✅", "in_progress": "⏳", "pending": "⬜"}
                    todo_box.markdown("**Plan**\n\n" + "\n".join(
                        f"- {icons.get(t['status'], '⬜')} {t['content']}" for t in todos))
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Error", state="error")
            final_text = f"❌ {type(e).__name__}: {e}"

        st.markdown(final_text)
        after = png_snapshot()
        new_plots = sorted(p for p, t in after.items() if before.get(p) != t)
        render_plots(new_plots)

    st.session_state["history"].append(
        {"role": "assistant", "content": final_text, "plots": [p.name for p in new_plots]})
