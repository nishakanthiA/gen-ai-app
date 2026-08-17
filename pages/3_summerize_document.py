import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from helper.constants import GPT_LLM, OPENAI_API_KEY
from helper.data_reader import load_document
from helper.file_manager import get_temp_file_path


def summarize_document(docs, api_key: str, summary_type: str = "Bullet Points"):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    full_text = "\n\n".join([doc.page_content for doc in split_docs])

    # Configure prompt style
    if summary_type == "Executive Summary":
        instructions = "Provide a high-level executive summary in 2-3 concise paragraphs "
    elif summary_type == "Detailed Overview":
        instructions = "Provide a detailed section-by-section breakdown of key topics covered "
    else:  # Bullet Points
        instructions = "Provide a key takeaways list with bullet points summarizing the core ideas "

    # Custom prompt
    prompt = PromptTemplate.from_template(
        """{instructions} from the following document.  
        If the document is empty, respond with 'No content to summarize.
        Restrict to 200 words maximum.'
        Document:
        {context}

        Summary:"""
    )

    llm = ChatOpenAI(model=GPT_LLM, temperature=0, api_key=api_key)

    # Build pipe chain
    chain = prompt | llm | StrOutputParser()

    # Run chain passing joined page content
    summary = chain.invoke({"context": full_text, "instructions": instructions})

    return summary


# --- Page Configuration ---
st.set_page_config(page_title="Document Summerizer", page_icon="📝", layout="centered")
st.title("📝 Document Summarizer")
st.caption("Upload a PDF or TXT file to generate a fast summary using LangChain & Gemini")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

summary_style = st.selectbox(
    "Summary Style",
    ["Bullet Points", "Executive Summary", "Detailed Overview"]
)

if uploaded_file is not None:
    # Display File Details
    st.info(f"**File uploaded:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

    if st.button("Generate Summary", type="primary"):
        with st.spinner("Reading document and generating summary..."):
            try:
                # 1. Load document
                tmp_path, _, file_ext = get_temp_file_path(uploaded_file)
                docs = load_document(tmp_path, file_ext)

                # 2. Summarize
                summary = summarize_document(docs, OPENAI_API_KEY, summary_type=summary_style)

                # 3. Output
                st.subheader("Summary Result")
                st.markdown(summary)
                st.success("Done!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
