
import streamlit as st
from google import genai

from backend import (
    EMBEDDING_MODEL_NAME,
    SUMMARY_FEATURE_MESSAGE,
    process_pdf,
    answer_pdf_question,
)

from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InsightPilot",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

# Cache the embedding model so Streamlit does not reload it
# every time the user clicks a button.
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


embedding_model = load_embedding_model()


# Gemini API key is stored privately in Streamlit secrets.
# We will configure this during deployment.
@st.cache_resource
def load_gemini_client():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )


# ============================================================
# SESSION STATE
# ============================================================

# Streamlit reruns the script whenever the user interacts with
# the interface. Session state keeps the processed PDF in memory.

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "answer" not in st.session_state:
    st.session_state.answer = None

if "source_pages" not in st.session_state:
    st.session_state.source_pages = []


# ============================================================
# HEADER
# ============================================================

st.title("InsightPilot")
st.caption("AI-Powered Document Intelligence")

st.markdown(
    "Upload a document and ask questions using natural language."
)

st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf"],
    help="PDF support is available in the current version."
)


# ============================================================
# PROCESS UPLOADED PDF
# ============================================================

if uploaded_file is not None:

    # Only process the PDF when a new file is uploaded.
    # This prevents unnecessary embedding on every Streamlit rerun.
    file_identifier = (
        uploaded_file.name,
        uploaded_file.size
    )

    if st.session_state.processed_file != file_identifier:

        with st.spinner(
            "Reading and indexing your document..."
        ):

            try:

                file_bytes = uploaded_file.getvalue()

                result = process_pdf(
                    file_bytes,
                    embedding_model
                )

                st.session_state.pdf_chunks = result["chunks"]
                st.session_state.faiss_index = result["index"]
                st.session_state.processed_file = file_identifier

                # Clear the previous document's answer.
                st.session_state.answer = None
                st.session_state.source_pages = []

            except Exception as error:

                st.error(
                    f"Could not process the PDF: {error}"
                )

    if st.session_state.processed_file == file_identifier:

        st.success(
            f"{uploaded_file.name} is ready."
        )


# ============================================================
# QUESTION + ANSWER
# ============================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


with left_column:

    st.subheader("Ask Your Document")

    question = st.text_area(
        "What would you like to know?",
        placeholder=(
            "e.g. What are the main challenges "
            "discussed in this document?"
        ),
        height=150
    )

    ask_button = st.button(
        "Ask InsightPilot",
        type="primary",
        use_container_width=True
    )

    summary_button = st.button(
        "Summarize Document",
        use_container_width=True
    )


# ============================================================
# HANDLE QUESTION
# ============================================================

if ask_button:

    if uploaded_file is None:

        st.warning(
            "Please upload a PDF first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif st.session_state.faiss_index is None:

        st.warning(
            "The document is still being processed."
        )

    else:

        with st.spinner(
            "Searching the document..."
        ):

            try:

                client = load_gemini_client()

                result = answer_pdf_question(
                    question=question,
                    chunks=st.session_state.pdf_chunks,
                    index=st.session_state.faiss_index,
                    embedding_model=embedding_model,
                    client=client
                )

                st.session_state.answer = result["answer"]
                st.session_state.source_pages = (
                    result["source_pages"]
                )

            except Exception as error:

                st.error(
                    f"InsightPilot could not generate an answer: {error}"
                )


# ============================================================
# ANSWER DISPLAY
# ============================================================

with right_column:

    st.subheader("Answer")

    if st.session_state.answer:

        st.markdown(
            st.session_state.answer
        )

        if st.session_state.source_pages:

            pages = ", ".join(
                str(page)
                for page
                in st.session_state.source_pages
            )

            st.caption(
                f"Sources: Page {pages}"
            )

    else:

        st.info(
            "Upload a document and ask a question "
            "to see your answer here."
        )


# ============================================================
# SUMMARY PLACEHOLDER
# ============================================================

if summary_button:

    if uploaded_file is None:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        st.info(
            SUMMARY_FEATURE_MESSAGE
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "InsightPilot • RAG-powered document intelligence | Developed by Angela Alminanza"
)
