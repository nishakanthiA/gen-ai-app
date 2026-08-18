import streamlit as st

from helper.chroma_connector import read_all_from_chroma, save_document
from helper.constants import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH
from helper.data_reader import load_document
from helper.file_manager import get_temp_file_path

# --- Page Configuration ---
st.set_page_config(page_title="Document Uploader", page_icon="📝", layout="centered")
st.title("📝 Document Uploader")
st.caption("Upload a PDF or TXT file to generate to save in Chroma DB")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded_file is not None:
    # Display File Details
    if st.button("Save file", type="primary"):
        with st.spinner("inserting into DB..."):
            try:
                # 1. Load document
                tmp_path, file_name, file_ext = get_temp_file_path(uploaded_file)
                docs = load_document(tmp_path, file_ext)

                # 2. Save Document
                save_document(docs, file_name, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)
                st.success("Done!")

                # 3. Show db records
                df = read_all_from_chroma(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)
                st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
