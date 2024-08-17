import streamlit as st
from dotenv import load_dotenv
import  fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA


load_dotenv()

def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    print(text) 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=20)
    chunks = text_splitter.create_documents([text])

    return chunks

def extract_text_from_pdf(pdf_file):

    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=20)
    chunks = text_splitter.create_documents([text])

    return chunks

def generate_summary(chunks):
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128,temperature=0.6,stop_sequences=["End of Report"])

    template = ''' Summarize the following pathology laboratory blood test report.Provide a concise answer based on the available data.
    Use the following report.{text}
    Provide a concise answer:
    '''
    final = PromptTemplate(template=template,input_variables=['text'])

    chain = load_summarize_chain(llm, chain_type="map_reduce",combine_prompt=final,verbose=False)
    summary = chain.invoke(chunks)
    return summary['output_text']

def vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")

    # Create FAISS vector store 
    vector_store = FAISS.from_documents(chunks,embedding_model)
    return vector_store

def create_chain(vs,question):
# Create a retrieval chain with the vector store and a HuggingFace endpoint
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128,temperature=0.6,stop_sequences=['Answer'])

    retriever = VectorStoreRetriever(vectorstore=vs)
    
    chain = RetrievalQA.from_chain_type(llm,retriever=retriever)
    answer = chain({"query":question})
    return answer['result']


    
st.set_page_config(page_title="Report Summary",page_icon='👨‍⚕️')
st.title("Medical Report Diagnosis 👨‍⚕️")
    
uploaded_file = st.file_uploader("Add Your Report Here :",type=["jpg","jpeg","pdf"])

if uploaded_file is not None :

    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension == "pdf":
        if "chunks" not in st.session_state:
            st.session_state.chunks = extract_text_from_pdf(uploaded_file)
        if "vec_store" not in st.session_state:
            st.session_state.vec_store = vector_store(st.session_state.chunks)

    elif file_extension in ["jpg", "jpeg"]:
        image = Image.open(uploaded_file)
        if "chunks" not in st.session_state:
            st.session_state.chunks = extract_text_from_image(image)
        if "vec_store" not in st.session_state:
            st.session_state.vec_store = vector_store(st.session_state.chunks)


    with st.sidebar:
        st.header("Summary : ")
        if st.button("Summarize Report"):
            st.session_state.summary = generate_summary(st.session_state.chunks)
            st.write(st.session_state.summary)
        if "summary" in st.session_state:
            st.write(st.session_state.summary)    


    question = st.chat_input("Ask About Report ")
    if question is not None and question!="":
        st.chat_message("user").write(question)
        answer = create_chain(st.session_state.vec_str,question)
        st.chat_message("assistant").write(answer)


    