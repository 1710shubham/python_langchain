import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

curreent_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(curreent_dir,"documents","lord_of_the_rings.txt")
persistent_directory = os.path.join(curreent_dir,"db","chroma_db")

if not os.path.exists(persistent_directory):
    print("persistent directory does not exit.Initializing vector store...")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exit. Please Check the path"
    )

loader = TextLoader(file_path)
documents =loader.load()

text_spliteer = CharacterTextSplitter(chunk_size = 1000,chunk_overlap = 0)
docs =text_spliteer.split_documents(documents)

print("\n--- Document Chunkes Information ---")
print(f"Number of Document chunks:{len(docs)}")
print(f"Sample Chunk:\n{docs[0].page_content}")

print("\n--- Creating embeddings ---")
embeddings = OpenAIEmbeddings(
    model = "text-embeddings-3-small"
)

print("\n--- Finished Creating embeddings ---")

print("\n--- Creating Vector sotre ---")

db = Chroma.from_documents(
    docs,embeddings,persist_directory = persistent_directory)

print("\n--- Finished Creating Vector sotre ---")
