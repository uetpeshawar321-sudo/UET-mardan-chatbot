import os
import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data/raw"
CHROMA_DIR = "chroma_db"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
PDF_MANIFEST_PATH = "data/pdfs/pdf_manifest.json"


def load_url_lookup():
    """Build a filename -> URL lookup from both manifests, so each chunk
    can be tagged with its real source URL."""
    lookup = {}

    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                lookup[entry["filename"]] = entry["url"]

    if os.path.exists(PDF_MANIFEST_PATH):
        with open(PDF_MANIFEST_PATH, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                lookup[entry["filename"]] = entry["url"]

    return lookup


def main():
    print("Loading documents from", DATA_DIR)
    loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    url_lookup = load_url_lookup()

    # Attach the real source URL to each document's metadata
    for doc in documents:
        filename = os.path.basename(doc.metadata.get("source", ""))
        doc.metadata["source_url"] = url_lookup.get(filename, "Unknown source")
        doc.metadata["filename"] = filename

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model (this may take a minute the first time)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding and storing in ChromaDB...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()

    print(f"\nDone. Stored {len(chunks)} chunks in '{CHROMA_DIR}'.")
    print("Your RAG index is ready to query.")


if __name__ == "__main__":
    main()