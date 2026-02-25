

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS    = 500
MAX_HISTORY_TURNS = 5

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()


def respond(user_message: str, history: list):
    if not user_message.strip():
        yield history
        return

    messages = []
    for msg in history[-MAX_HISTORY_TURNS * 2:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    thread = Thread(target=model.generate, kwargs=dict(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        streamer=streamer,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    ))
    thread.start()

    new_history   = list(history) + [{"role": "user", "content": user_message}]
    partial_reply = ""

    for token in streamer:
        partial_reply += token
        yield new_history + [{"role": "assistant", "content": partial_reply + "▌"}]

    thread.join()

    yield new_history + [{"role": "assistant", "content": partial_reply.strip()}]

# Gradio UI

CSS = """
#chatbot { height: 520px; }
#title   { text-align:center; font-size:1.8rem; font-weight:700; margin-bottom:.2rem; }
#sub     { text-align:center; color:#666; margin-bottom:1rem; }
#warn    { text-align:center; color:#e67e22; font-size:0.9rem; margin-bottom:1rem; }
"""

with gr.Blocks(title="Chubs") as demo:

    gr.Markdown("# Chubs", elem_id="title")
    gr.Markdown(f"Powered by **{MODEL_NAME}**", elem_id="sub")

    chatbot = gr.Chatbot(elem_id="chatbot", label="Conversation", height=500)

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type your message and press Enter …",
            show_label=False,
            scale=9,
        )
        send_btn = gr.Button("Send ➤", scale=1, variant="primary")

    clear_btn = gr.Button("🗑️ Clear conversation", variant="secondary")


    state = gr.State([])

    def submit(msg, hist):
        for partial in respond(msg, hist):
            yield partial, partial, ""

    user_input.submit(submit, [user_input, state], [chatbot, state, user_input])
    send_btn.click(submit,    [user_input, state], [chatbot, state, user_input])
    clear_btn.click(lambda: ([], []), outputs=[chatbot, state])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CSS,
    )