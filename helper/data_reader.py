import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_document(file_path, file_ext):
    try:
        if file_ext.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext.lower() == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return loader.load()

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)