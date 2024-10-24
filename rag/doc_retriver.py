from langchain_community.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from typing import List, Optional, Set
import requests
import os

class DocumentRetriever:
    def __init__(self, embedding_model: str = 'sentence-transformers/all-mpnet-base-v2', persist_directory: Optional[str] = 'chromadb') -> None:
        """
        Initializes the DocumentRetriever class with an embedding model and a directory for persisting the vector store.

        Args:
            embedding_model (str): The HuggingFace embedding model to be used. Defaults to 'sentence-transformers/all-mpnet-base-v2'.
            persist_directory (str, optional): Directory to store the vector database. Defaults to 'chromadb'.
        """
        # Initialize the embedding function
        self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
        self.persist_directory = persist_directory

        # Initialize a Chroma vector store, and optionally persist it if a directory is specified
        self.vectorstore = Chroma(embedding_function=self.embedding_function, persist_directory=self.persist_directory)
        if self.persist_directory:
            self.vectorstore.persist()

    def load_web_document(self, url: str) -> None:
        """
        Loads a document from a web page and adds it to the vector store.

        Args:
            url (str): The URL of the web document to load.
            parse_classes (list, optional): HTML classes to parse. Defaults to typical content classes.
        """
        
        loader = WebBaseLoader(
            web_paths=(url, ),
        )
        documents = loader.load()

        # Automatically add the loaded documents to the vector store
        self.add_documents(documents)

    def load_pdf_from_url(self, pdf_url: str, local_path: str = "temp_pdf_file.pdf") -> None:
        """
        Downloads a PDF from a URL and loads it into the vector store.

        Args:
            pdf_url (str): The URL of the PDF file to download and load.
            local_path (str, optional): The local file path to save the downloaded PDF. Defaults to 'temp_pdf_file.pdf'.
        """
        response = requests.get(pdf_url)
        with open(local_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        
        # Once downloaded, use the load_pdf_document function to process it
        self.load_pdf_document(local_path)

        # Optionally, remove the temporary PDF file after processing
        if os.path.exists(local_path):
            os.remove(local_path)

    def load_pdf_document(self, pdf_path: str) -> None:
        """
        Loads a document from a PDF file and adds it to the vector store.

        Args:
            pdf_path (str): The file path of the PDF document to load.
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Add the loaded documents to the vector store
        self.add_documents(documents)

    def split_documents(self, documents: List[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[dict]:
        """
        Splits documents into smaller chunks for efficient processing.

        Args:
            documents (list): A list of documents to split.
            chunk_size (int, optional): Maximum size of each chunk. Defaults to 1000.
            chunk_overlap (int, optional): Number of characters to overlap between chunks. Defaults to 200.

        Returns:
            list: A list of split document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(documents)
        return all_splits

    def add_documents(self, documents: List[dict]) -> None:
        """
        Adds new documents to the vector store.

        Args:
            documents (list): A list of documents to add to the vector store.
        """
        # Split the documents into smaller parts and add them to the vector store
        all_splits = self.split_documents(documents)
        self.vectorstore.add_documents(documents=all_splits, embeddings=self.embedding_function)

        if self.persist_directory:
            self.vectorstore.persist()

    def delete_documents(self, path: str) -> None:
        """
        Deletes documents from the vector store based on the 'source' metadata.

        Args:
            path (str): The source path of the documents to be deleted.
        """
        self.vectorstore.delete(where={"source": {"$eq": path}})
        if self.persist_directory:
            self.vectorstore.persist()

    def get_retriever(self):
        """
        Returns a retriever instance for searching within the vector store.

        Returns:
            ChromaRetriever: The retriever object for document search.
        """
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
    def get_vectorstore(self) -> Chroma:
        """
        Returns the current vector store.

        Returns:
            Chroma: The vector store instance.
        """
        return self.vectorstore
    
    def get_unique_sources(self) -> Set[str]:
        """
        Retrieves all unique 'source' metadata values from the documents in the vector store.

        Returns:
            set: A set of unique source metadata values.
        """
        unique_sources = set()
        for metadata in self.vectorstore.get()['metadatas']:
            source_value = metadata.get('source')
            if source_value:
                unique_sources.add(source_value)
        return unique_sources
