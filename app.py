import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        # Create a PDF reader object
        pdf_reader = PdfReader(pdf)
        # Loop through each page and extract text
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits the large text into smaller chunks for the AI to process."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, # Overlap helps keep context between chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, api_key):
    """Converts text chunks into embeddings and stores them in a FAISS vector database."""
    # Use OpenAI Embeddings to convert text to vectors
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    # Create the FAISS database from the chunks
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, api_key):
    """Creates the conversational chain that links the vector database with the AI model."""
    # Initialize the OpenAI Chat model
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    
    # Create memory to remember the conversation history
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    # Create the chain that will retrieve relevant text and generate an answer
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    """Handles the user's question, gets the response from the chain, and updates the chat UI."""
    # Send the question to the conversational chain
    response = st.session_state.conversation({'question': user_question})
    
    # Update the chat history in the session state
    st.session_state.chat_history = response['chat_history']

    # Display the conversation
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # User message
            with st.chat_message("user"):
                st.write(message.content)
        else:
            # AI message
            with st.chat_message("assistant"):
                st.write(message.content)

def main():
    # Set the page configuration for the Streamlit app
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # App title
    st.header("Chat with multiple PDFs :books:")
    
    # Get API Key from environment variable first
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If not found, try to get it from Streamlit Secrets (for cloud deployment)
    if not api_key or api_key == "your_api_key_here":
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
    
    # Sidebar configuration
    with st.sidebar:
        st.subheader("Configuration")
        
        # Only show the input box if we don't have a permanent key
        if not api_key or api_key == "your_api_key_here":
            user_api_key = st.text_input("OpenAI API Key", type="password", help="Get your API key from platform.openai.com")
            if user_api_key:
                api_key = user_api_key
        else:
            st.success("API Key is securely loaded from server!")
            
        st.subheader("Your documents")
        # File uploader allows multiple PDF files
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=['pdf'])
        
        # Button to trigger processing
        if st.button("Process"):
            if not api_key or api_key == "your_api_key_here":
                st.error("Cannot process documents without an OpenAI API Key.")
                return
                
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
                
            with st.spinner("Processing..."):
                # 1. Get the raw text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                # 2. Split the text into smaller chunks
                text_chunks = get_text_chunks(raw_text)

                # 3. Create vector database to store the chunks
                vectorstore = get_vectorstore(text_chunks, api_key)

                # 4. Create the conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
                
                st.success("Processing complete! You can now ask questions.")

    # Chat input at the bottom of the screen
    user_question = st.chat_input("Ask a question about your documents:")
    
    # If the user enters a question and the chain is initialized
    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please upload and process your PDFs first.")
        else:
            handle_userinput(user_question)

if __name__ == '__main__':
    main()
