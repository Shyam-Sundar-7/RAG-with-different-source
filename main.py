import streamlit as st
from helper import aws,file_to_chunks,azure_data_download,main
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
st.title("RAG with different source")
import time
# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Which data source you are using?",
    ("Azure Blob Storage", "Local storage","AWS S3 Bucket"),
    index=None,
    placeholder="Select contact method...",
)

if add_selectbox == "Azure Blob Storage":
    st.sidebar.write("You selected Azure Blob Storage.")
    AZURE_CONNECTION_STRING = st.sidebar.text_input("Azure Connection String Input",type="password")
    CONTAINER_NAME = st.sidebar.text_input("Azure Container Name")
elif add_selectbox == "Local storage":
    st.sidebar.write("You selected Local storage.")
    LOCAL_PATH = st.sidebar.selectbox(
    "Local Path",
    ("Local_data","NIL"),
    index=None,
    placeholder="Select contact method...",
)
elif add_selectbox == "AWS S3 Bucket":
    aws_access_key = st.sidebar.text_input("AWS Access Key",type="password")
    aws_secret_access_key = st.sidebar.text_input("AWS SECRET ACCESS KEY",type="password")
    bucket_name= st.sidebar.text_input("AWS BUCKET NAME")
    object_name= st.sidebar.text_input("AWS OBJECT NAME")
else:
    st.sidebar.write("You selected nothing.")

if st.sidebar.button("Injest"):
    if add_selectbox == "Azure Blob Storage" and AZURE_CONNECTION_STRING and CONTAINER_NAME:
        # Download PDF from Azure Blob Storage
        with st.sidebar:
            try:
                with st.spinner("Azure connection is creating....."):
                    azure_data_download(AZURE_CONNECTION_STRING=AZURE_CONNECTION_STRING, CONTAINER_NAME=CONTAINER_NAME)
                st.session_state.pages = file_to_chunks("Azure_data")
                with st.spinner("Chroma VectorDatabse is creating....."):
                    st.session_state.db = Chroma.from_documents(st.session_state.pages, OpenAIEmbeddings(), persist_directory="Azure_Chroma_db")
                st.success("Chroma VectorDatabse is created successfully in the Azure_Chroma_db")
            except Exception as e:
                st.error(f"Error connecting with Azure: {str(e)}")
    elif add_selectbox == "Local storage" and LOCAL_PATH == "Local_data":
        with st.sidebar:
            with st.spinner("Local Folder documents to chunks are in the process.........."):
                st.session_state.pages = file_to_chunks("Local_data")
            with st.spinner("Chroma VectorDatabse is creating....."):
                st.session_state.db = Chroma.from_documents(st.session_state.pages, OpenAIEmbeddings(), persist_directory="Local_Chroma_db")
            st.success("Chroma VectorDatabse is created successfully in the Local_Chroma_db folder")
    elif add_selectbox == "AWS S3 Bucket":
        st.write("AWS S3 Bucket")
        st.write(aws_access_key)
        st.write(aws_secret_access_key)
        st.write(bucket_name)
        st.write(object_name)
    st.session_state.injest = True



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "injest" not in st.session_state:
    st.session_state.injest = False
if "pages" not in st.session_state:
    st.session_state.pages = []
if "db" not in st.session_state:
    st.session_state.db = []  

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Lets have a chat with our document") and st.session_state.injest:
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    answer=main(prompt,chunks=st.session_state.pages,db=st.session_state.db)
    response = f"{answer}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})