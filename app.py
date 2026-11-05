from hashlib import sha256
import os

import streamlit as st

from rag import (
    MAX_CHUNKS, answer_question, chunk_pages, create_embeddings,
    extract_pdf_pages, find_relevant_chunks,
)


st.set_page_config(page_title="PDF-avustaja", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stHeader"] { display: none; }
    .stApp { background: #f5f7fb; }
    .block-container { max-width: 1480px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1 { color: #17243c; letter-spacing: -0.04em; }
    h2, h3 { color: #17243c; letter-spacing: -0.02em; }
    [data-testid="stForm"] { background: white; border-radius: 16px; }
    [data-testid="stFileUploader"] { border-radius: 12px; }
    .stButton button, .stFormSubmitButton button { border-radius: 10px; }
    .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
        background: #315a96; border-color: #315a96; color: white;
    }
    .stTextArea textarea:disabled {
        color: #334155; -webkit-text-fill-color: #334155; opacity: 1;
    }
    @media (prefers-color-scheme: dark) {
        .stApp { background: #0e1117; }
        h1, h2, h3 { color: #e2e8f0; }
        [data-testid="stForm"] { background: #161b26; }
        .stTextArea textarea:disabled {
            color: #e2e8f0; -webkit-text-fill-color: #e2e8f0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.caption("OMA DOKUMENTTI · SELKEÄT VASTAUKSET · TARKISTETTAVAT LÄHTEET")
st.title("PDF-avustaja")
st.write("Kysy dokumentistasi. Löydä vastaus ja tarkista, mihin se perustuu.")
st.divider()

upload_column, text_column = st.columns([1, 2], gap="large")
pages = None
document = None
embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
llm_model = os.getenv("GEMINI_LLM_MODEL", "gemini-3.1-flash-lite")

with upload_column, st.container(border=True):
    st.caption("01 / DOKUMENTTI")
    st.subheader("Lataa PDF")
    st.write("Valitse yksi tekstiä sisältävä PDF. Enimmäiskoko 10 Mt.")
    st.caption("Skannattuja PDF-tiedostoja ei tueta: tekstintunnistus (OCR) ei ole käytössä.")
    uploaded_pdf = st.file_uploader("Valitse PDF-tiedosto", type=["pdf"])

    if uploaded_pdf is None:
        st.session_state.pop("pdf_document", None)
    else:
        pdf_bytes = uploaded_pdf.getvalue()
        identity = (sha256(pdf_bytes).hexdigest(), embedding_model)
        document = st.session_state.get("pdf_document")
        if document is None or document["identity"] != identity:
            document = {
                "identity": identity, "pages": None, "vectors": None,
                "pending_vectors": [], "matches": [],
            }
            st.session_state["pdf_document"] = document
            try:
                document["pages"] = extract_pdf_pages(pdf_bytes)
                document["chunks"] = chunk_pages(document["pages"])
            except ValueError as error:
                document["error"] = str(error)
        pages = document["pages"]
        if pages is None:
            st.warning(document["error"])
        else:
            st.text(f"Tiedosto: {uploaded_pdf.name}")
            st.metric("Sivuja", len(pages))
            empty_pages = [index for index, text in enumerate(pages, start=1) if not text.strip()]
            if empty_pages:
                st.warning(
                    f"Tekstiä ei löytynyt {len(empty_pages)} sivulta: "
                    + ", ".join(map(str, empty_pages[:20]))
                    + ("…" if len(empty_pages) > 20 else "")
                    + ". Nämä sivut eivät sisälly hakuun. Skannattujen sivujen OCR ei ole käytössä."
                )
            st.caption(
                "Haun valmistelu lähettää PDF:n tekstin Googlelle. "
                "Vastausta varten lähetetään kysymys ja enintään kolme tekstipalaa."
            )
            too_long = len(document["chunks"]) > MAX_CHUNKS
            if too_long:
                st.warning(f"Haku tukee enintään {MAX_CHUNKS} tekstipalaa. Valitse lyhyempi PDF.")
            pending = document.setdefault("pending_vectors", [])
            if pending and document["vectors"] is None:
                st.caption(f"Valmisteltu {len(pending)} / {len(document['chunks'])}. Jatka painikkeella.")
            if st.button(
                "Valmistele haku", disabled=too_long or document["vectors"] is not None,
                type="primary", use_container_width=True,
            ):
                try:
                    progress = st.progress(len(pending) / len(document["chunks"]), text="Valmistellaan hakua…")
                    with st.spinner("Valmistellaan hakua…"):
                        document["vectors"] = create_embeddings(
                            [chunk["text"] for chunk in document["chunks"]],
                            model=embedding_model,
                            completed=pending,
                            on_progress=lambda done, total: progress.progress(
                                done / total, text=f"Valmisteltu {done} / {total} tekstipalaa",
                            ),
                        )
                except ValueError as error:
                    st.error(str(error))
                finally:
                    progress.empty()
            if document["vectors"] is not None:
                st.success("Haku on valmis.")

with text_column, st.container(border=True):
    st.caption("02 / TUTKI DOKUMENTTIA")
    if pages is None:
        st.subheader("Aloita omasta dokumentistasi")
        st.info("Lataa PDF vasemmalta nähdäksesi sen tekstin tässä.")
        st.write("1. Lataa PDF ja valmistele haku.")
        st.write("2. Kirjoita kysymys dokumentin sisällöstä.")
        st.write("3. Lue vastaus ja tarkista alkuperäiset lähteet.")
    else:
        question_tab, preview_tab = st.tabs(["Kysy dokumentista", "Lue PDF:n teksti"])
        with question_tab:
            st.subheader("Mitä haluat tietää?")
            st.caption("Etsi ensin sopivat tekstikohdat ja pyydä sitten lähteisiin perustuva vastaus.")
            st.caption("Voit kysyä myös turkiksi. Vastaukset annetaan suomeksi.")
            with st.form("document_search"):
                question = st.text_input("Kysymys", max_chars=1000, placeholder="Esim. Mitkä ovat dokumentin tärkeimmät tiedot?")
                search = st.form_submit_button(
                    "Etsi tekstikohdat", disabled=document["vectors"] is None,
                    type="primary", use_container_width=True,
                )
            if search and (question.strip() != document.get("question") or not document["matches"]):
                document["matches"] = []
                document.pop("answer", None)
                if not question.strip():
                    st.warning("Kirjoita kysymys ennen hakua.")
                else:
                    try:
                        with st.spinner("Etsitään tekstikohtia…"):
                            question_vector = create_embeddings(
                                [question.strip()], task_type="RETRIEVAL_QUERY", model=embedding_model,
                            )[0]
                            document["matches"] = find_relevant_chunks(
                                document["chunks"], document["vectors"], question_vector,
                            )
                            document["question"] = question.strip()
                    except ValueError as error:
                        st.error(str(error))
            if document["matches"]:
                st.caption(f"Hakutulokset kysymykseen: {document['question']}")
                if document.get("answer_model") != llm_model:
                    document.pop("answer", None)
                if st.button("Vastaa kysymykseen", disabled=document.get("answer") is not None, type="primary"):
                    try:
                        with st.spinner("Luodaan vastausta…"):
                            document["answer"] = answer_question(
                                document["question"], document["matches"],
                            )
                            document["answer_model"] = llm_model
                    except ValueError as error:
                        st.error(str(error))
                if document.get("answer") is not None:
                    st.subheader("Vastaus")
                    with st.container(border=True):
                        st.text(document["answer"]["answer"])
                    if document["answer"]["sources"]:
                        st.caption("LÄHTEET · Alkuperäiset lainaukset dokumentista")
                    for source in document["answer"]["sources"]:
                        with st.expander(f"Lähde [{source['id']}] – sivu {source['page']}", expanded=True):
                            st.text(source["quote"])
                for index, chunk in enumerate(document["matches"], start=1):
                    with st.expander(f"{index}. tekstikohta – sivu {chunk['page']}"):
                        st.text(chunk["text"])

        with preview_tab:
            st.caption(
                "Näytetään valitun sivun koko poimittu teksti. "
                "PDF:n ulkoasu ja palstojen lukujärjestys eivät välttämättä säily."
            )
            page_number = st.selectbox("Valitse sivu", range(1, len(pages) + 1))
            page_text = pages[page_number - 1]
            if page_text.strip():
                st.text_area(
                    "Sivun koko teksti",
                    value=page_text,
                    height=500,
                    disabled=True,
                )
            else:
                st.info("Tältä sivulta ei löytynyt tekstiä.")
