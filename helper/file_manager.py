import base64
import os
import tempfile

from streamlit.runtime.uploaded_file_manager import UploadedFile


def encode_image(image_path: str) -> str:
    """
    Encodes an image file to a base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_temp_file_path(uploaded_file: UploadedFile) -> tuple[str, str, str]:
    """
    Saves an uploaded file to a temporary location and returns the file path, file name, and extension.
    """
    file_name = uploaded_file.name
    file_ext = os.path.splitext(file_name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        file_path = temp_file.name

    return file_path, file_name, file_ext


def format_docs(docs: list) -> str:
    """
    Combines retrieved document contents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)
