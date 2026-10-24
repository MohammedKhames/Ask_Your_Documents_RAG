# Imports
from pathlib import Path
from dotenv import load_dotenv, find_dotenv







# Load environment variables from .env
load_dotenv(find_dotenv())


# Directory settings
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"


# Vector database settings
COLLECTION = "course_documents"


# Text chunking settings (measured in characters)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# Retrieval settings: number of chunks to return
TOP_K = 4


# Chat model: generates answers through Groq
CHAT_MODEL = "openai/gpt-oss-20b"


# Embedding model: converts document chunks and questions into vectors
# Use the same model for both indexing and searching
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"