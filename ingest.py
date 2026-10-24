
# 0- General Imports
from config import (
    DATA_DIR,
    DB_DIR,
    COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL,
)


# 1- Load Documents Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# 2- Chunking Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 3- Loading Embedding Model
from langchain_huggingface import HuggingFaceEmbeddings


# 4- Store in Vector DB
import shutil

from langchain_chroma import Chroma




# 1- Load Documents 

# Load PDF and TXT files
def load_documents():
    documents = []

    for file_path in sorted(DATA_DIR.iterdir()):
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            loader = PyPDFLoader(str(file_path))

        elif extension == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")

        else:
            continue

        documents.extend(loader.load())

    return documents



# 2- Chunking The Documents

# Split documents into overlapping text chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    return chunks



# 3- Embedding Model

# Load the embedding model
def load_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )

    return embeddings



# 4- Store in Vector DB

# Rebuild the vector database from the text chunks
def build_vector_store(chunks):
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    embeddings = load_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(DB_DIR),
    )

    return vector_store


#-------------------------------------------------------#
# 5 - Linking The all above Functions


# Run the indexing pipeline when this file is executed directly
if __name__ == "__main__":
    if not DATA_DIR.is_dir():
        raise SystemExit("Data folder not found. Create data/ first.")

    documents = load_documents()

    if not documents:
        raise SystemExit("No PDF or TXT documents found in data/.")

    chunks = split_documents(documents)

    if not chunks:
        raise SystemExit("No text chunks found. Check your documents.")

    source_files = {
        document.metadata["source"]
        for document in documents
    }

    print(f"{len(source_files)} files -> {len(chunks)} chunks")
    print("Building vector database...")

    build_vector_store(chunks)

    print(f"Database saved to: {DB_DIR}")