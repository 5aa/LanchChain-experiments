import logging
import requests
import time
from urllib.parse import urljoin

import chromadb
from chromadb.config import Settings
from bs4 import BeautifulSoup

from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.chroma import Chroma # bugged import workaround


def init_log():
    logger = logging.getLogger(__name__)

    # Set the logging level to INFO (or any other desired level)
    logger.setLevel(logging.INFO)

    # Create a console handler and set its level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add the formatter to the handler
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger


def parse_github_wiki_sidebar_links(url: str):
    # Send an HTTP GET request to the URL and get the HTML content
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Identify the sidebar using the class "wiki-pages-box"
    sidebar = soup.find('nav', class_='wiki-pages-box')

    links = []
    if sidebar:
        logger.info("Sidebar found in the HTML content")
        # Find all links in the sidebar
        for a in sidebar.find_all('a'):
            links.append(urljoin(url, a['href']))
    else:
        logger.info("Sidebar not found in the HTML content.")

    logger.info(f"Found {len(links)} links in the sidebar at {url}.")
    return links


def load_from_github_wiki(url: str):
    links = parse_github_wiki_sidebar_links(url)

    logger.info(f"Loading {len(links)} links from {url}.")
    return UnstructuredURLLoader(
        urls=links,
        show_progress_bar=True
    ).load()


def ingest_docs(url: str):
    # retrieve documents from target url
    docs = load_from_github_wiki(url)

    # chunk documents
    logger.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    doc_chunks = text_splitter.split_documents(docs)

    # initialize embedding model
    logger.info("Initializing embedding model...")
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        cache_folder="./embedding_models",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # initialize vector storage, load documents
    logger.info("Initializing vector storage...")
    start = time.time()
    db = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embedding_model,
        # persist_directory="./chroma",
        client_settings=Settings(anonymized_telemetry=False)
    )
    duration = time.time() - start
    logger.info(f"Vector storage initialized in {duration} seconds.")
    logger.info(f"Contains {len(db.get()['documents'])} documents.")

    query = "Explain the difference between chat and instruct modes."
    retrieved_docs = db.similarity_search(query)
    print(retrieved_docs[0].page_content)

    return


if __name__ == "__main__":
    logger = init_log()

    url = "https://github.com/oobabooga/text-generation-webui/wiki"

    # for link in parse_github_wiki_sidebar_links(url):
    #     print(link)

    ingest_docs(url)