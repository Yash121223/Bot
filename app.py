import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

@st.cache_resource
def load_model():
    model_path = "yash2913/fine-tuned-healix-mistral" 

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
        use_auth_token=os.environ["HF_TOKEN"]
    )
    return tokenizer, model

tokenizer, model = load_model()

def chat_with_mistral(prompt, history):
    system_prompt = "[INST] You are a kind, supportive mental health assistant. Offer empathy, encouragement, and self-care suggestions. Be warm, calm, and non-judgmental. Do not diagnose or give medical advice. Always recommend talking to a licensed professional when needed. [/INST]\n"
    full_prompt = system_prompt + "".join([f"User: {msg[0]}\nAssistant: {msg[1]}\n" for msg in history]) + f"User: {prompt}\nAssistant:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()

st.set_page_config(page_title="Healix Mental Health Chatbot", layout="centered")
st.title("🧠 Healix – Your Mental Health Companion")
st.markdown("I'm here to listen and support you.")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input.strip():
    with st.spinner("Thinking..."):
        bot_reply = chat_with_mistral(user_input, st.session_state.history)
        st.session_state.history.append((user_input, bot_reply))

if st.session_state.history:
    for user_msg, bot_msg in reversed(st.session_state.history):
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**Healix:** {bot_msg}")