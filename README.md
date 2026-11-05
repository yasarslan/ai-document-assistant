# AI Document Assistant

A local Streamlit app that answers questions about one text-based PDF at a time. Ask in your preferred language, read the answer in Finnish, and check the cited pages and original excerpts.

## What it does

- Extracts and previews PDF text page by page.
- Finds relevant passages using Gemini embeddings and cosine similarity.
- Uses up to three passages to generate a Finnish answer. When the passages do not provide an answer, the app says so.
- Shows original excerpts and page numbers from the PDF. Source verification checks that the excerpts exist in the document; read them to check whether they support the answer.

## Quick start

Use Python 3.12 (tested). Download and extract the repository with **Code → Download ZIP**, or clone it with Git. Open a terminal in the project folder.

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
python -m streamlit run app.py --server.address 127.0.0.1
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed in the terminal. You can upload and preview a PDF without an API key.

## API provider and configuration

Gemini is the example integration included in this repository. The app uses it for embeddings and answers when run as provided. The same retrieval and citation flow can work with other LLM APIs by adapting the answer-generation call in `rag.py`. Embeddings can use a separate provider; use the same embedding model for both PDF passages and questions, and rebuild the index after switching models.

### Quick start with Gemini

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey). Open the local `.env` file and fill in `GEMINI_API_KEY=`. Restart the app after changing its configuration. Each user supplies their own key; the example file contains no credentials. To use a different provider, adapt the API calls and configuration first; changing the key alone does not switch providers.

| Setting | Purpose | Default |
| --- | --- | --- |
| `GEMINI_API_KEY` | Key for Gemini API calls | Empty |
| `GEMINI_EMBEDDING_MODEL` | Embeddings for the PDF and questions | `gemini-embedding-001` |
| `GEMINI_LLM_MODEL` | Finnish answers | `gemini-3.1-flash-lite` |

Keep the actual key in `.env` or an environment variable. Git ignores `.env`, uploaded PDFs, and local files. Commit only `.env.example` with an empty key.

## Using the app

1. Upload a text-based PDF and select **Valmistele haku** (prepare search).
2. In **Kysy dokumentista** (ask the document), enter a question and select **Etsi tekstikohdat** (find passages).
3. Select **Vastaa kysymykseen** (answer the question). Open the cited excerpts to verify the response.
4. Use **Lue PDF:n teksti** (read PDF text) to inspect any extracted page.

Try asking `Mitä dokumentissa sanotaan toimitusajasta?` or `Belgeye göre teslimat süresi nedir?` when your document contains delivery terms.

## Privacy and limits

The app runs locally and does not save uploaded PDFs or their text to disk. Preparing search sends extracted text to Gemini; searching sends the question, and answering sends the question with up to three selected passages. These operations use your Gemini API quota and may incur charges. Only upload documents you are allowed to send to Gemini.

The app supports one PDF up to 10 MB and up to 600 searchable text chunks. Scanned pages require OCR, which this app does not provide. Complex layouts can affect reading order.

## Development

Python · Streamlit · pypdf · Google Gen AI SDK

- `app.py` contains the interface and session state.
- `rag.py` contains extraction, retrieval, Gemini calls, and source verification.
