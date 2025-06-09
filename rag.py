import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#loading the data
PDF_PATH = "C:\\Users\\manan\\OneDrive\\Desktop\\research-assistant\\data\\NIPS-2017-attention-is-all-you-need-Paper.pdf"
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

text = extract_text_from_pdf(PDF_PATH)
print(text)


#splitting the data into chunks
splitter_func = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = splitter_func.split_text(text)
docs = splitter_func.create_documents([text]) # Document(...) is just LangChain's way of attaching metadata to each chunk — which is required by vector stores like FAISS

print(docs)


#convert to embeddings and store in vector database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
print("VectorStore\n")
print(vectorstore)