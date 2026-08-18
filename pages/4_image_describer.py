import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from helper.constants import GPT_LLM, OPENAI_API_KEY
from helper.file_manager import encode_image, get_temp_file_path


def generate_image_description(image_path: str, prompt_text: str = None) -> str:
    if not prompt_text:
        prompt_text = "Describe the image in detail. Max 100 words."

    # Convert image to base64
    base64_image = encode_image(image_path)

    # Define the prompt template with image placeholder directly
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                [
                    {"type": "text", "text": "{prompt}"},
                    {"type": "image_url", "image_url": {"url": "{image_url}"}},
                ],
            )
        ]
    )

    # Define model and parser
    model = ChatOpenAI(model=GPT_LLM, max_tokens=300, api_key=OPENAI_API_KEY)
    parser = StrOutputParser()

    # Pure LCEL Chain: Prompt -> Model -> Parser
    vision_chain = prompt | model | parser

    # Invoke chain directly
    result = vision_chain.invoke(
        {
            "prompt": prompt_text,
            "image_url": f"data:image/jpeg;base64,{base64_image}",
        }
    )

    return result


def clear_history():
    st.session_state.image_chat_history = []
    st.session_state.image_description = ""


# --- Page Configuration ---
st.set_page_config(page_title="Image Describer", page_icon="📷", layout="centered")
st.title("📷  Image Describer")
st.caption("Upload a png, jpg or jpeg file to generate a summary using LangChain & GPT")

# Initialize session state variables if they do not already exist
if "image_chat_history" not in st.session_state:
    st.session_state.image_chat_history = []

if "image_description" not in st.session_state:
    st.session_state.image_description = ""

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], on_change=clear_history)

if uploaded_image is not None:

    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    # Get the temporary file path for the uploaded image
    file_path, _, _ = get_temp_file_path(uploaded_image)

    if st.button("Describe Image", type="primary"):
        with st.spinner("generating summary..."):
            try:
                # Generate a description for the uploaded image
                description = generate_image_description(file_path)
                st.session_state.image_description = description
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display the generated image description
    st.markdown(st.session_state.image_description)

    # Display the chat history for the image
    for message in st.session_state.image_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input for questions about the image
    if user_input := st.chat_input("Ask any question about the image:"):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.image_chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    # Generate a response based on the uploaded image and user input
                    response = generate_image_description(file_path, user_input)
                    st.markdown(response)
                    st.session_state.image_chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    st.button("Clear Chat History", on_click=lambda: st.session_state.image_chat_history.clear())
