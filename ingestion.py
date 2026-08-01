import os
os.environ["USER_AGENT"] = "MyAgenticRAGApp/1.0 (Contact: myemail@example.com)"

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings

load_dotenv()



urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

persist_dir = "./.chroma"

if os.path.exists(persist_dir):
    import shutil
    shutil.rmtree(persist_dir)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
    persist_directory=persist_dir,
)

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory=persist_dir,
    embedding_function=embeddings,
).as_retriever()