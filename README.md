import asyncio
from pathlib import Path

import streamlit as st

from config import settings
from core.orchestrator import MultiAgentOrchestrator

st.set_page_config(page_title="Multi-Agent Orchestration Platform", layout="wide")
st.title("Multi-Agent Orchestration Platform")

with st.sidebar:
    st.header("Credentials")
    hf_token = st.text_input("HF Token", settings.HF_TOKEN, type="password")
    github_token = st.text_input("GitHub Token", settings.GITHUB_TOKEN, type="password")
    openrouter_key = st.text_input("OpenRouter API Key", settings.OPENROUTER_API_KEY, type="password")
    e2b_key = st.text_input("E2B API Key", settings.E2B_API_KEY, type="password")

    st.header("Task Selector")
    task_type = st.radio("Task", ["AUTO", "CODE", "PICTURE", "VIDEO"], index=0, horizontal=True)

    st.header("Model Picks")
    fast_model = st.selectbox("Fast", ["openai/gpt-oss-20b:free", "meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-chat-v3.1"])
    deep_model = st.selectbox("Deep", ["deepseek/deepseek-v3.1", "deepseek/deepseek-r1", "meta-llama/llama-3.3-70b-instruct"])
    code_model = st.selectbox("Code", ["mistralai/codestral-latest", "qwen/qwen3-coder-480b-instruct", "openai/gpt-oss-20b:free"])

    st.caption("Token reduction layers: prompt cache, history compaction, output caps, log summarization.")

user_goal = st.text_area("Goal", height=180, placeholder="Write a Python script to monitor CPU and save a PNG chart.")

if st.button("Run orchestration"):
    orchestrator = MultiAgentOrchestrator(
        hf_token=hf_token,
        github_token=github_token,
        openrouter_key=openrouter_key,
        e2b_key=e2b_key,
        generator_model=deep_model,
        critic_model=code_model,
        arbiter_model=fast_model,
    )

    with st.spinner("Running the task and reducing prompt/token overhead..."):
        result = asyncio.run(orchestrator.run(user_goal, task_type=task_type))

    if result.modality == "CODE":
        st.subheader("Final Python")
        st.code(result.code, language="python")
        if result.execution_logs:
            st.text_area("Execution Logs", result.execution_logs, height=200)
        if result.comparison_results:
            st.subheader("Fast / Deep / Code compare")
            for item in result.comparison_results:
                st.markdown(f"### {item['model']}")
                st.code(item["output"], language="python")
        with st.expander("Agent logs"):
            for line in result.logs:
                st.write(line)

    elif result.modality == "IMAGE":
        st.subheader("Generated Image")
        if result.image_path and Path(result.image_path).exists():
            st.image(result.image_path, caption="Generated image")
            with open(result.image_path, "rb") as f:
                st.download_button("Download image", f.read(), file_name="generated_image.png", mime="image/png")
        st.write(result.final_output)

    elif result.modality == "VIDEO":
        st.subheader("Generated Video")
        if result.video_path and Path(result.video_path).exists():
            st.video(result.video_path)
            with open(result.video_path, "rb") as f:
                st.download_button("Download video", f.read(), file_name="generated_video.mp4", mime="video/mp4")
        st.write(result.final_output)

    with st.expander("All logs"):
        for line in result.logs:
            st.write(line)
