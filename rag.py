import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline


#loading the data
PDF_PATH = "C:\\Users\\manan\\OneDrive\\Desktop\\research-assistant\\data\\NIPS-2017-attention-is-all-you-need-Paper.pdf"
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

#text = extract_text_from_pdf(PDF_PATH)



#splitting the data into chunks
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
def create_vectorstore(text):
    splitter_func = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = splitter_func.split_text(text)
    docs = splitter_func.create_documents([text]) # Document(...) is just LangChain's way of attaching metadata to each chunk — which is required by vector stores like FAISS
    #convert to embeddings and store in vector database
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectorstore/")
    return vectorstore


#now to build a model pipeline and load the stored vector database
def ask_question(question):
    stored_info = FAISS.load_local("vectorstore/", embeddings=embeddings,allow_dangerous_deserialization=True)
    generator = hf_pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=150) 
    llm = HuggingFacePipeline(pipeline=generator)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=stored_info.as_retriever())
    return qa.run(question)

# choice = 'y'

# while(choice == 'y'):
#     question = input("Ask a question: \n")
#     answer = qa.run(question)
#     print(answer)

#     print("Do you want to continue\n")
#     choice = input("y/n\n")
#     if(choice == 'n'): 
#         break