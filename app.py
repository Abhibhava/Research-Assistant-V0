import gradio as gr
from rag import ask_question, create_vectorstore, extract_text_from_pdf

PDF_PATH = "C:\\Users\\manan\\OneDrive\\Desktop\\research-assistant\\data\\NIPS-2017-attention-is-all-you-need-Paper.pdf"

def process_pdf(PDF_PATH):
    text = extract_text_from_pdf(PDF_PATH)
    create_vectorstore(text)
    return "PDF processed and vector database created successfully!"
