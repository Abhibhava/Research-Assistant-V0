import gradio as gr
from rag import ask_question, create_vectorstore, extract_text_from_pdf

PDF_PATH = "C:\\Users\\manan\\OneDrive\\Desktop\\research-assistant\\data\\NIPS-2017-attention-is-all-you-need-Paper.pdf"

def process_pdf(file):
    text = extract_text_from_pdf(file.name)  # .name gives file path
    create_vectorstore(text)
    return "✅ PDF processed and vector DB created successfully!"

def answer_question(query):
    return ask_question(query)

with gr.Blocks() as demo:
    gr.Markdown("## 🔍 RAG PDF Assistant")

    with gr.Tab("📁 Upload PDF"):
        file_input = gr.File(label="Upload a PDF")
        upload_output = gr.Textbox()
        upload_button = gr.Button("Process PDF")
        upload_button.click(fn=process_pdf, inputs=file_input, outputs=upload_output)

    with gr.Tab("💬 Ask a Question"):
        question_input = gr.Textbox(label="Ask your question here")
        answer_output = gr.Textbox(label="Answer")
        ask_button = gr.Button("Get Answer")
        ask_button.click(fn=answer_question, inputs=question_input, outputs=answer_output)

demo.launch(share=True)
