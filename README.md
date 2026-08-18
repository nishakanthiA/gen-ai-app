# OmniRAG

OmniRAG is a Streamlit-based Retrieval-Augmented Generation (RAG) application that allows users to interact with a
Chroma vector store for document-based question answering. The app connects to a persistent Chroma database, retrieves
relevant document chunks, and uses OpenAI's GPT model to generate concise answers. Additionally, it includes features
for document summarization and image description.

## Pages Overview

### 1. Document Store Page

- Allows users to upload documents.
- Save the uploaded documents to a Chroma database for later retrieval.
- Displays the total number of records stored in the Chroma database.

### 2. Chat Page

- Provides a summary table of all documents, including their file names and chunk counts.
- Enables users to ask questions about the documents stored in the Chroma database.
- Displays the conversation history between the user and the assistant.
- Provides concise answers to user queries, along with optional source context for reference.
- Allows users to clear the chat history.

### 1. Document Summery Page

- Allows users to upload documents.
- Allows users to select a summary style.
- Generates a concise summary by processing the document's content.

### 2. Image Describer Page

- Allows users to upload an image.
- Generate a detailed description.
- Enables users to ask questions about the Image.
- Provides concise answers to user queries about the Image.
- Displays the conversation history between the user and the assistant.
- Allows users to clear the chat history.

## How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gen-ai-app.git
   cd gen-ai-app

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Set up your environment variables:
    - Create a `.env` file in the root directory of the project.
    - Add your OpenAI API key to the `.env` file:
      ```
      OPENAI_API_KEY=your_openai_api_key
      ```
4. Run the Streamlit app:
    ```bash
    streamlit run main.py
    ```
5. Open your web browser and navigate to `http://localhost:8501` to access the application.

## Features

- **Document Store**: Allows users to upload PDF or TXT files to create a knowledge base stored in a Chroma
  database. This knowledge base is used for a Retrieval-Augmented Generation (RAG) application to enable document-based
  question answering.
- **Interactive Q&A**: Retrieval-Augmented Generation (RAG) application that allows users to interact with a Chroma
  vector store for document-based question answering. It provides an interactive chat interface.
- **Document Summary**: Summarizing uploaded PDF or TXT documents. It allows users to upload a document, select a
  summary style, and generates a concise summary by processing the document's content through a text-splitting and
  prompt-based language model pipeline.
- **Image Description**: Generating image descriptions. It allows users to upload an image, generate a detailed
  description, and interact with the assistant through a chat interface for further inquiries about the image.

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for the list of dependencies.
