import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,HumanMessagePromptTemplate,SystemMessagePromptTemplate
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple


load_dotenv()

     
def get_vectorstore(url):
    # get the url text
    loader = WebBaseLoader(url)
    documents = loader.load()

    # split document to chunks
    text_splitter = RecursiveCharacterTextSplitter()
    doc_chunks = text_splitter.split_documents(documents)
    
    # create vector store
    embedding_model = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")

    # Create FAISS vector store
    vector_store = FAISS.from_documents(doc_chunks,embedding_model)

    return vector_store

def get_context_retriever_chain(vector_store):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(repo_id=model_id, max_length=128,temperature =0.5)

    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("history"),
        ("user","{input}"),
        ("user","Given the above conversation, generate a search query to look up in to get information relevant to the conversation.")    
    ])

    retriever_chain = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt)
    return retriever_chain

def get_conversation_rag_chain(retrieved_chain):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(repo_id=model_id, max_length=128,temperature =0.5)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer the user's question based on below context and keep the answer short and concise and just answer what has been asked nothing else.:\n\n{context}"),
        MessagesPlaceholder("chat"),
        ("human","{input}")
    ])

    stuff_document_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)

    return create_retrieval_chain(retrieved_chain,stuff_document_chain)

def get_response(user_query):

    # create conversation chain
    retrieved_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversation_rag_chain(retrieved_chain)

    # generating response 
    response = conversation_rag_chain.invoke({
        "chat": st.session_state.chat_history,
        "input" : user_query
    })

    # fetching the answer part
    return response.get('answer')
   

# app config
st.set_page_config(page_title="Website Chat",page_icon="")
st.title("CHAT WITH WEBSITES")


# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url=="":
    st.info("Please input a Website URL")

else:
    # session-state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content="Hello how can I help you ?")
    ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore(website_url)


    # user input
    user_input = st.chat_input("Type your message here...")
    if user_input is not None and user_input!="":
        response = get_response(user_input)
        with st.sidebar:
            st.write(HumanMessage(content=user_input))
            st.write(AIMessage(content=response))
        
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)









