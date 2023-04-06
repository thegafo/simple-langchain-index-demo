# query-index.py

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
db = Chroma(collection_name="test", persist_directory="db",
            embedding_function=embeddings)

retriever = db.as_retriever()

query = "What did the speaker say about Putin?"

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever)

print(qa.run(query))
