import streamlit as st
import json
import os
import time
import tempfile
import shutil
from fpdf import FPDF
from dotenv import load_dotenv
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index import ServiceContext, set_global_service_context, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import GradientEmbedding
from llama_index.llms import GradientBaseModelLLM


# Load environment variables
# Function to load environment variables from .env file
def load_env_variables():
    load_dotenv()

load_env_variables()

# Function to initialize and return the Cassandra session
def init_cassandra():
    # Retrieve credentials from environment variables
    client_id = os.getenv('CASSANDRA_CLIENT_ID')
    client_secret = os.getenv('CASSANDRA_CLIENT_SECRET')

    auth_provider = PlainTextAuthProvider(client_id, client_secret)
    cloud_config = {
        'secure_connect_bundle': 'secure-connect-end-to-end-pdf-chat-using-llamaindex-llama-2.zip'
    }
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    return cluster.connect()

# Function to initialize LlamaIndex service context
def init_llama_index():
    # Set environment variables
    os.environ['GRADIENT_ACCESS_TOKEN'] = os.getenv('GRADIENT_ACCESS_TOKEN')
    os.environ['GRADIENT_WORKSPACE_ID'] = os.getenv('GRADIENT_WORKSPACE_ID')

    # Initialize LlamaIndex components
    llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)
    embed_model = GradientEmbedding(
        gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
        gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
        gradient_model_slug="bge-large",
    )
    service_context = ServiceContext.from_defaults(
        chunk_size=256, llm=llm, embed_model=embed_model
    )
    set_global_service_context(service_context)
    return service_context

# Function to process uploaded PDF and query
def process_pdf_and_query(uploaded_file, user_question, session, service_context):
    # Save the uploaded file to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Assuming SimpleDirectoryReader can process a directory of PDF files
        documents = SimpleDirectoryReader(temp_dir).load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(user_question)

    return response
# Main Streamlit app
def main():
    # UI Styling
    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">PDF Chatting App using Llama2 & Llamaindex📚</p>', unsafe_allow_html=True)

    # Initialize response variable
    response = ""
    
    # Initialize Cassandra session and LlamaIndex service context
    session = init_cassandra()
    service_context = init_llama_index()

    # UI for uploading PDF
    st.subheader("Upload a PDF Document 📄")
    uploaded_file = st.file_uploader("", type=["pdf"])

    # UI for user question input
    st.subheader("Ask a Question About the PDF❓")
    user_question = st.text_input("", placeholder="Type your question here...")

    # Submit button with progress bar
    if st.button("Submit 🚀", key="submit") and uploaded_file is not None and user_question:
        with st.spinner('Processing...⏳'):
            my_bar = st.progress(0)
            for percent_complete in range(100):
                # Simulate a delay for processing (this is a placeholder, adjust the delay as needed)
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)

            # Process the PDF and the query
            response = process_pdf_and_query(uploaded_file, user_question, session, service_context)

            # Display the response
            st.subheader("Response 📜")
            st.text_area("", value=response, height=200, key="response")

# Run the main function
if __name__ == "__main__":
    main()
