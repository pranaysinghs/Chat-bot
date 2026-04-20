import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            # PyPDF2 can only extract text, not images. 
            # If the page is an image, this returns ""
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits the large text into smaller chunks for the AI to process."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, 
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks, api_key):
    """Converts text chunks into embeddings and stores them in a FAISS vector database."""
    # Use HuggingFace's free local embedding model to avoid Google rate limits!
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore, api_key):
    """Creates the conversational chain that links the vector database with the AI model."""
    # Initialize the Google Gemini Chat model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def handle_userinput(user_question):
    """Handles the user's question, gets the response from the chain, and updates the chat UI."""
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    
    # Get API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            pass
    
    with st.sidebar:
        st.subheader("Configuration")
        
        if not api_key or api_key == "your_api_key_here":
            user_api_key = st.text_input("Google API Key", type="password", help="Get your free API key from aistudio.google.com")
            if user_api_key:
                api_key = user_api_key
        else:
            st.success("API Key is securely loaded from server!")
            
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=['pdf'])
        
        if st.button("Process"):
            if not api_key or api_key == "your_api_key_here":
                st.error("Cannot process documents without a Google API Key.")
                return
                
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
                
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                
                # SAFETY CHECK: Make sure the PDF actually had readable text!
                if not raw_text.strip():
                    st.error("Could not extract any text from the uploaded PDFs! They might be scanned images or empty documents. Please try a different PDF that contains readable text.")
                    return

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks, api_key)
                st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
                st.success("Processing complete! You can now ask questions.")

    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please upload and process your PDFs first.")
        else:
            handle_userinput(user_question)

if __name__ == '__main__':
    main()
