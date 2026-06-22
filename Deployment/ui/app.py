import os
import gradio as gr
from PIL import Image
from client import call_qcm

API_URL = os.environ.get("API_URL", "http://localhost:8000")

def answer_qcm(image, question, ca, cb, cc, cd):
    if image is None:
        return "⚠️ Veuillez uploader une image."
    choices = []
    for letter, text in zip("abcd", [ca, cb, cc, cd]):
        if text.strip():
            choices.append(f"{letter}) {text.strip()}")
    if not choices:
        return "⚠️ Veuillez renseigner au moins un choix."
    try:
        data = call_qcm(API_URL, question, choices, image)
        return (
            f"### Réponse : **{data['answer'].upper()}**\n\n"
            f"Sortie brute : `{data['raw_output']}`\n\n"
            f"⏱ {data['generation_time_ms']} ms"
        )
    except Exception as e:
        return f"⚠️ Erreur API : {e}"

with gr.Blocks(title="VLM — QCM") as demo:
    gr.Markdown("# 🧠 VLM — Répondre à un QCM avec image")
    with gr.Row():
        img_input = gr.Image(type="pil", label="Image")
        with gr.Column():
            question = gr.Textbox(label="Question", value="What is in this image?")
            ca = gr.Textbox(label="Choix A")
            cb = gr.Textbox(label="Choix B")
            cc = gr.Textbox(label="Choix C")
            cd = gr.Textbox(label="Choix D (optionnel)")
    btn = gr.Button("Répondre", variant="primary")
    output = gr.Markdown()
    btn.click(answer_qcm, inputs=[img_input, question, ca, cb, cc, cd], outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7860)