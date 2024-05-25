from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
nomic_api_key = os.environ['NOMIC_API_KEY']

collection_name = "camp_recipes_chroma"
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
db = Chroma(embedding_function=embeddings, persist_directory="./recipes_chroma_db")

namespace = f"chroma/{collection_name}"
record_manager = SQLRecordManager(
    namespace, db_url="sqlite:///record_manager_index2_chroma_cache.sql"
)

record_manager.create_schema()

loader = PyPDFDirectoryLoader("C:/Users/j/.vscode/3D Reconstruction/cookbook_folder")
recipes = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)

recipe = text_splitter.split_documents(recipes)

index(recipe, record_manager, db, cleanup="incremental", source_id_key="source")

db.add_documents(recipe)

