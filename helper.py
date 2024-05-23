from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import boto3
import os
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel
import numpy as np
from dotenv  import load_dotenv
load_dotenv()
import shutil

def load_file(file_name):
    loader=[]
    # print(file_name.split(".")[-1])
    if file_name.split('.')[-1] == "pptx":
        loader = UnstructuredPowerPointLoader(file_name).load()
    elif file_name.split('.')[-1] == "pdf":
        loader = PyPDFLoader(file_name).load()    
    elif file_name.split('.')[-1] == "docx":
        loader = Docx2txtLoader(file_name).load()
    elif file_name.split('.')[-1] == "html":
        loader = UnstructuredHTMLLoader(file_name).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300)
    pages = text_splitter.split_documents(loader)
    return pages


def file_to_chunks(folder):
    pages=[]
    for file_name in os.listdir(f"{folder}"):
        pages.extend(load_file(f"{folder}\\{file_name}"))
    if folder != "Local_data":
        shutil.rmtree(f"{folder}")
    return pages

def azure_data_download(AZURE_CONNECTION_STRING,CONTAINER_NAME):
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    if not os.path.exists("Azure_data"):
        os.mkdir("Azure_data")
    for file_name in container_client.list_blobs():
        blob_client = container_client.get_blob_client(file_name)
        with open(f"Azure_data\\{file_name.name}", "wb") as file:
            data = blob_client.download_blob().readall()
            file.write(data)


def aws(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME,object_name):
        # Create an S3 client
    s3 = boto3.client('s3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    # List objects in the bucket
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)

    if not os.path.exists("S3_data"):
        os.mkdir("S3_data")

    # Download files in the 'data' object
    for i in response.get('Contents',[]):
        if i['Key'].split('/')[-1] != "" and i['Key'].split('/')[0] == object_name:
            # print(i['Key'])
            file_path = os.path.join("S3_data", i['Key'].split('/')[-1])
            # print(file_path)
            s3.download_file(BUCKET_NAME, i['Key'], file_path)


def generate_queries(query):
        
    # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate Four 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)


    generate_querie = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
        | (lambda x: [query] + x)
    )
    return generate_querie 

def _get(a):
    dd=[]
    for s in a:
        dd.extend(s)
    return dd

def get_unique_documents(doc_list):
    seen_content = set()
    unique_documents = []
    
    for doc in doc_list:
        content = doc.page_content
        if content not in seen_content:
            seen_content.add(content)
            unique_documents.append(doc)
    
    del seen_content
    
    return unique_documents

def keyword_extractor():
    prompt="""
    You are an AI language model assistant. Your task is to help the user retrieve keywords from their query. 

    Please provide me with the keywords you would like to extract from your query. 

    Keywords: {keywords}
    """
    prompt_perspectives=ChatPromptTemplate.from_template(prompt)
    generate_querie = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() )
    return generate_querie

def main(Query,chunks,db):
    
    faiss_retriever=db.as_retriever(search_kwargs={'k': 10})

    Bm25_retriever = BM25Retriever.from_documents(chunks)
    Bm25_retriever.k = 10

    map_chain=generate_queries | faiss_retriever.map() | _get | get_unique_documents
    key_chain=keyword_extractor() | Bm25_retriever | get_unique_documents

    ensemble_retriever = EnsembleRetriever(
    retrievers=[map_chain, key_chain], weights=[0.5, 0.5]
    )

    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=4)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    final_prompt="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:"""

    final_prompt_perspectives=ChatPromptTemplate.from_template(final_prompt)

    llm_chain= ({"context": itemgetter("query") | compression_retriever,
            "question":itemgetter("query")}
            | 
            RunnableParallel({
                "response":  final_prompt_perspectives | ChatOpenAI(temperature=0) | StrOutputParser() ,
                "context": itemgetter("context")
            })
            )
    
    return llm_chain.invoke({"query":Query})