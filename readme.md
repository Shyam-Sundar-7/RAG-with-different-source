
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