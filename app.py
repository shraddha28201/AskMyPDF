import streamlit as st
from google import genai

import rag

st.title("AskMyPDF")
st.caption("Built by Shraddha Nimbalkar | github.com/shraddha28201")

api_key = rag.get_key() or st.sidebar.text_input("Gemini API key", type="password")
pdf_file = st.file_uploader("Upload a PDF", type="pdf")


@st.cache_data
def get_index(pdf_bytes):
    return rag.build_index(pdf_bytes)


if pdf_file and api_key:
    chunks, embs = get_index(pdf_file.read())
    client = genai.Client(api_key=api_key)
    question = st.text_input("Ask a question about your PDF")
    if question:
        with st.spinner("Thinking..."):
            answer, top = rag.ask(question, chunks, embs, client)
        st.write(answer)
        with st.expander("Sources"):
            for i in top:
                st.caption(f"Page {chunks[i]['page']}")
                st.write(chunks[i]["text"][:400])
else:
    st.info("Upload a PDF to start.")