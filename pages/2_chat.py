import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from helper.chroma_connector import get_collection_summery, get_db
from helper.constants import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH, GPT_LLM, OPENAI_API_KEY
from helper.file_manager import format_docs


@st.cache_resource
def load_qa_chain(persist_dir: str, collection_name: str, api_key: str):
    # Connect to persistent Chroma store
    db = get_db(persist_dir, collection_name)

    # Configure retriever
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Initialize LLM
    llm = ChatOpenAI(model=GPT_LLM, temperature=0, api_key=api_key)

    # Define system prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )

    # Create a chat prompt template with system and human messages
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    # Construct the RAG chain using LCEL pipe syntax
    rag_chain = prompt | llm | StrOutputParser()

    # `RunnableParallel` pipeline for a conversational Retrieval-Augmented Generation (RAG) chain.
    # `RunnableParallel` allows running multiple tasks in parallel. Here, it processes "docs" and "question" inputs
    conversational_rag_chain = RunnableParallel(
        {
            "docs": RunnablePassthrough() | retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(
        context=lambda x: format_docs(x["docs"])
    ).assign(
        answer=rag_chain
    )

    return conversational_rag_chain


def clear_chat_history():
    st.session_state.messages = []


# --- Page Configuration ---
st.set_page_config(page_title="Doc Q&A Assistant", page_icon="📚")
st.title("📚 Ask Questions from Document Store")
st.success(f"Connected to Chroma DB (`{CHROMA_COLLECTION_NAME}`)")
st.button("Clear Chat History", on_click=clear_chat_history)

# Show summery of the document store
summery = get_collection_summery(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)

st.caption(f"Total entries: {summery['chunk_count']}")
st.caption(f"Total documents: {summery['document_count']}")

st.table(
    [{
        "Document": doc['file_name'],
        "Chunks": doc['chunk_count']
    }
        for doc in summery['documents']]
)

# Initialize the chat history if it does not already exist in the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input & Generation
if user_query := st.chat_input("Ask a question about your saved documents..."):
    # Display user prompt in chat
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching Chroma DB and generating answer..."):
            # Invoke the RAG chain with the user query and extract the answer
            rag_chain = load_qa_chain(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, OPENAI_API_KEY)
            response = rag_chain.invoke(user_query)
            answer = response["answer"]
            sources = response.get("docs", [])

            # Display main answer
            st.markdown(answer)

            # Display referenced sources in an expandable widget
            if sources:
                with st.expander("View Source Context"):
                    for idx, doc in enumerate(sources):
                        source_file = doc.metadata.get("source", "Unknown file")
                        page_num = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Source {idx + 1}:** `{source_file}` (Page {page_num})")
                        st.caption(doc.page_content)

    # Save assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
