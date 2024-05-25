from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

collection_name = "african_recipes_chroma"
embeddings = hf
db = Chroma(embedding_function=embeddings, persist_directory="./african_recipes_chroma_db")

namespace = f"chroma/{collection_name}"
record_manager = SQLRecordManager(
    namespace, db_url="sqlite:///record_manager_index_chroma_cache.sql"
)

record_manager.create_schema()

loader = PyPDFDirectoryLoader("C:/Users/j/.vscode/3D Reconstruction/magic_chef/recipes")
try:
    recipes = loader.load()
    print(f"Loaded {len(recipes)} documents.")
except Exception as e:
    print(f"Error loading documents: {e}")
    recipes = []

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750, chunk_overlap=75)

recipe = text_splitter.split_documents(recipes)

index(recipe, record_manager, db, cleanup="incremental", source_id_key="source")

