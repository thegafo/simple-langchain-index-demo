# create-index.py

from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
    docs, embeddings, persist_directory="db", collection_name="test")
vectordb.persist()
