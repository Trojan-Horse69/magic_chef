from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

loader = PyPDFDirectoryLoader("C:/Users/j/.vscode/3D Reconstruction/cookbook_folder")
recipes = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)

recipe = text_splitter.split_documents(recipes)

db.add_documents(recipe)
