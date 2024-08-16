import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from io import StringIO
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.ctransformers import CTransformers
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_huggingface.llms import HuggingFacePipeline
import torch
from transformers import pipeline




def extract_text_from_image(image):
    #image_path = image
    img = Image.open(image)
    text = pytesseract.image_to_string(img)
    return text


def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


def get_text_chunks(text):
     text_splitter = CharacterTextSplitter(
          separator="\n",
          chunk_size=500,
          chunk_overlap=50,
          length_function=len
     )
     chunks = text_splitter.split_text(text)
     return chunks


#def get_vector_store(text_chunks):
#    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
#    return vectorstore


  #def get_conversation_chain(vectorstore): 
  #  llm = CTransformers(model='model\llama-2-7b-chat.Q8_0.gguf',
  #                      model_type='llama')
  #  memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
  #  conversation_chain = ConversationalRetrievalChain.from_llm(
  #      llm=llm,
  #      retriever=vectorstore.as_retriever(),
  #      memory=memory
  #  )
  #  return conversation_chain


def report_gen(text):
    generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
                         trust_remote_code=True, return_full_text=True)
      
    ## Prompt Template
    template = """ 
    Generate a summary of following medical report data : {text}
    """

    prompt = PromptTemplate(input_variables=["text"],template=template)
    
    hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

    llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)

    ## Generate Response from llama model
    #response = LLMChain(llm=hf_pipeline,prompt=prompt)

    response = print(llm_chain.predict(text=text).lstrip())

    return response




def main():
    load_dotenv() 

    st.set_page_config(page_title="Medical Report Diagnosis")

    if "text" not in st.session_state:
        st.session_state.text = None
    

    st.title("Medical Report Diagnosis !!!")

    uploaded_file = st.file_uploader("Choose a file",type=["jpg","jpeg","pdf"])

    if uploaded_file is not None:
        file_type = uploaded_file.type
        if file_type == "application/pdf":
                st.session_state.text = extract_text_from_pdf(uploaded_file)
                if st.button ("Extracted Text:"):
                    st.write(st.session_state.text)
        elif file_type == "image/jpeg":
            st.session_state.text = extract_text_from_image(uploaded_file)
            if st.button ("Extracted Text:"):
                st.write(st.session_state.text)
        else:
            st.error("Unsupported file type")


        #if st.button("Get Info"):
        #    with st.spinner("Processing"):
        #        # create text chunks
        #        st.session_state.text_chunks = get_text_chunks(text)

        #        # create vector store
        #        #st.session_state.vector_store = get_vector_store(text_chunks)
        #        
        #        # create conversation chain
        #        #st.session_state.conversation = get_conversation_chain(vector_store)      
            
        if st.button("Generate Report Details "):
            with st.spinner('Generating'):
                response = report_gen(st.session_state.text)
                st.write(response)
                


if __name__ == '__main__':
    main()


