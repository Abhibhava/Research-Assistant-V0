import fitz

PDF_PATH = "C:\\Users\\manan\\OneDrive\\Desktop\\research-assistant\\data\\NIPS-2017-attention-is-all-you-need-Paper.pdf"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

text = extract_text_from_pdf(PDF_PATH)
print(text)
