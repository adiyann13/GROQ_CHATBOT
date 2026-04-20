import streamlit as st
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

###set up streamlit
st.title("Converstaional QA BOT with history")
st.write('üpload pdf files here')

api_key = st.text_input('ënter your api key here',type='password')

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if api_key:
    llm_mod = ChatGroq(groq_api_key = api_key , model_name='qwen/qwen3-32b')

    session_id = st.text_input('session id:',value = 'session_1')

    if 'store' not in st.session_state:
        st.session_state['store'] = {}
    
    uploaded_files = st.file_uploader('çhoose you pdf file',type='pdf', accept_multiple_files=True)
    ##processing uploaded files
    if uploaded_files :
        documents=[]
        for fils in uploaded_files:
            temp_file = f"./temp.pdf"
            with open(temp_file, 'wb') as file:
                file.write(fils.read())
                file_name = fils.name
            
            loader = PyPDFLoader(temp_file)
            docs = loader.load()
            documents.extend(docs)
        

        text_splitters = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap=60)
        splits= text_splitters.split_documents(documents)
        vectordb = Chroma.from_documents(documents =splits, embedding = embedder)
        retriever = vectordb.as_retriever()
    

        contextual_system_prompt = (
            "Given a chat history and the latest user questions"
            "which might reference context in history"
            "formulate  a standalone question which can be understood"
            "without the chat history do not answer the questions"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", contextual_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', "{input}"),
        ]
        )

        history_retriever = create_history_aware_retriever(llm_mod, retriever, prompt)
        ##

        system_prompt = """ You are a powerful ai bot who can answer to the user questions use the retrived context provided to you
        to answer the questions. If you do not know the answer just say you dont know the answer do not makeup answers
        Context : {context}"""


        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ('human', "{input}"),
        ])

        qa_chain =create_stuff_documents_chain(llm_mod, qa_prompt)
        rag_chain = create_retrieval_chain(history_retriever, qa_chain)

        def get_session_hist(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

    
        covo_rag_chain = RunnableWithMessageHistory(rag_chain, get_session_hist,
                                                    input_messages_key='input', history_messages_key='chat_history',
                                                    output_messages_key='answer')
        
        user_input = st.text_input('enter your questions')
        if user_input:
            sess_histry = get_session_hist(session_id)
            response = covo_rag_chain.invoke(
                {'input': user_input},
                config={
                    'configurable':{'session_id':session_id}
                },

            )

            st.write(st.session_state.store)
            st.success(f'Assistant: {response["answer"]}')
    else:
        st.warning('please enter groq api key')
    
    
    




