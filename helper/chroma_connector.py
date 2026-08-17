import hashlib

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from helper.constants import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH, HUGGINGFACE_EMBEDDING_MODEL


def get_db(
    persist_directory: str = CHROMA_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model: str = HUGGINGFACE_EMBEDDING_MODEL
) -> Chroma:
    """
    Initializes and returns a Chroma database instance.
    """
    try:
        # Initialize the embedding model
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        # Initialize your Chroma instance
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        return db
    except Exception as e:
        print(f"An error occurred while initializing the Chroma database: {e}")
        return None


def generate_content_chunk_id(doc: Document) -> str:
    """
    Generates a deterministic ID based on the file source and actual chunk content.
    Format: hash(file_source + chunk_page_content)
    """
    source = doc.metadata.get("source", "unknown")
    content = doc.page_content.strip()

    unique_str = f"{source}::{content}"
    return hashlib.sha256(unique_str.encode("utf-8")).hexdigest()


def save_document(documents, file_name, persist_directory: str, collection_name: str):
    """
    Reads a PDF file, splits it into chunks, and stores embeddings into Chroma DB.
    """

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunked_docs = text_splitter.split_documents(documents)

    # Initialize Chroma DB
    db = get_db(persist_directory, collection_name)

    current_file_chunk_ids = []
    new_chunks = []

    for chunk in chunked_docs:
        # Update source to use file name
        chunk.metadata["source"] = file_name
        # Generate a deterministic ID for the chunk based on its content and source
        chunk_id = generate_content_chunk_id(chunk)
        current_file_chunk_ids.append(chunk_id)
        new_chunks.append(chunk)

        # Clean up stale chunks from previous runs of this file
        # Query Chroma for all IDs belonging to this file source
        existing_records = db.get(
            where={"source": file_name},
            include=[]
        )
        existing_ids = set(existing_records.get("ids", []))

        # Determine stale IDs (chunks that existed previously but no longer match new content)
        new_id_set = set(current_file_chunk_ids)
        stale_ids = list(existing_ids - new_id_set)

        if stale_ids:
            print(f"Removing {len(stale_ids)} stale chunk(s) from previous version...")
            db.delete(ids=stale_ids)

        # Upsert active chunks
        if new_chunks:
            # Batch upsert
            batch_size = 500
            for i in range(0, len(new_chunks), batch_size):
                batch_chunks = new_chunks[i: i + batch_size]
                batch_ids = current_file_chunk_ids[i: i + batch_size]

                db.add_documents(
                    documents=batch_chunks,
                    ids=batch_ids
                )
            print(f"Upserted {len(new_chunks)} active chunk(s) cleanly.")

    print("Successfully ingested and saved data!")


def chroma_custom_search(persist_directory: str, collection_name: str, query: str):
    """
    Reads from Chroma DB for a given query and saves the results to a .txt file.
    """

    try:
        db = get_db(persist_directory, collection_name)
        # Perform a similarity search for the query
        documents = db.similarity_search(query, k=3)

        # Save documents to a .txt file
        with open("chroma.txt", "w", encoding="utf-8") as file:
            for doc in documents:
                file.write(doc.page_content + "\n\n")  # Write content with spacing

        print("Successfully retrieved documents from Chroma DB.")
        return documents

    except Exception as e:
        print(f"An error occurred while reading from Chroma DB: {e}")
        return None


def clear_chromadb(persist_directory: str, collection_name: str):
    """
    Clears all data from a specified Chroma DB collection.
    """
    db = get_db(persist_directory, collection_name)
    db.reset_collection()


def read_all_from_chroma(persist_directory: str, collection_name: str) -> pd.DataFrame:
    """
    Reads all data from Chroma DB for a given collection.
    """

    try:
        db = get_db(persist_directory, collection_name)
        data = db.get()
        df = pd.DataFrame()
        # Format into a clean dataframe
        if data["ids"]:
            df = pd.DataFrame(
                {
                    "ID": data["ids"],
                    "Document Content": data["documents"],
                    "Metadata": [str(m) for m in data["metadatas"]]
                }
            )

        return df

    except Exception as e:
        print(f"An error occurred while reading from Chroma DB: {e}")
        return None


def get_collection_summery(persist_directory: str, collection_name: str) -> dict:
    """
    Returns a summary of the Chroma DB collection, including total chunks and document counts.
    """
    total_chunks = 0
    document_list = []
    df = read_all_from_chroma(persist_directory, collection_name)

    if not df.empty:
        total_chunks = len(df)
        df["file_name"] = df["Metadata"].apply(lambda meta: eval(meta).get("source", "unknown"))
        document_count = df.groupby("file_name").size().reset_index(name="chunk_count")
        document_list = document_count.to_dict(orient="records")

    return {
        "collection_name": collection_name,
        "chunk_count": total_chunks,
        "document_count": len(document_list),
        "documents": document_list
    }
