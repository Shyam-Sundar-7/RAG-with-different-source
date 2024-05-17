
# RAG Implementation with Multiple Storage Options

This project implements a Retrieval-Augmented Generation (RAG) model that interacts with various storage options including local storage, Azure Blob Storage, and AWS S3 buckets. The application is designed to process and convert data into chroma vector stores after chunking the data retrieved based on user credentials. This implementation is encapsulated in a user-friendly Streamlit interface.

## Features

- **Multi-Source Data Handling**: Seamlessly integrates with local storage, Azure Blob Storage, and AWS S3 to fetch data.
- **Data Processing**: Converts data into chunks and subsequently into a chroma vector store.
- **RAG Model**: Utilizes a multi-query RAG model to generate summaries by retrieving relevant documents based on the input query.
- **Streamlit Interface**: Provides a simple and interactive UI to manage the workflow and display results.

## Storage Configuration

### Azure Blob Storage
- Requires Azure Blob connection string and container name.

### AWS S3 Bucket
- Requires AWS Access Key, AWS Secret Key, AWS Bucket Name, and Object Name.

### Local Storage
- Uses a local folder located at the root of the project.

## RAG Model Workflow

1. **Query Input**: Accepts an input query.
2. **Question Generation**: Uses an LLM model to generate different questions while preserving the semantic meaning of the input query.
3. **Keyword Extraction and Document Retrieval**: Extracts keywords and retrieves documents using a BM25 retriever.
4. **Document Ensemble**: Passes documents through an ensemble retriever to fetch relevant documents.
5. **Re-ranking**: Uses a cross-encoder reranker to select the top 4 documents.
6. **Summary Generation**: The top 4 documents are used as context for the LLM to generate a comprehensive summary.
## Installation and Setup

To replicate and run this project, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shyamsundar009/RAG-with-different-source
   cd RAG-with-different-source
   ```

2. **Install Required Libraries**
    
    Follow the steps to create a virtual environment and install the required libraries.
   ```bash
    python -m venv myenv
    
    myenv\Scripts\activate
    
    pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:

    Insert OpenAI Key

    Insert your OpenAI API key in the appropriate configuration file or environment variable. Use the .env_template file and paste the key and rename it as .env

    ```bash
    OPENAI_API_KEY="sk-#####################"
    ```
4. **Run the Streamlit Application**
   ```bash
   streamlit run main.py
   ```

5. **Access the Application**
   - Open your web browser and navigate to `http://localhost:8501` to view the application.


# Demo

## For Local_data:
[(https://github-production-user-asset-6210df.s3.amazonaws.com/167984593/331570907-0b794eb9-57a5-45fa-af7c-448d92e21a81.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240517%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240517T110610Z&X-Amz-Expires=300&X-Amz-Signature=4cf15634fe21b52a3b25847cac4a64f904940fea97b44806d05cdf90332e09dd&X-Amz-SignedHeaders=host&actor_id=167984593&key_id=0&repo_id=801965576)]

## For Azure_data:
[(https://github-production-user-asset-6210df.s3.amazonaws.com/167984593/331570893-2de560ce-b986-4038-8563-f8a1b10b9ddc.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240517%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240517T110559Z&X-Amz-Expires=300&X-Amz-Signature=8bb72c7bd1698c01b479d3367d7191d3420548104423308ab78c9500f29edf85&X-Amz-SignedHeaders=host&actor_id=167984593&key_id=0&repo_id=801965576)]
