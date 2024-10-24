from typing import List, Tuple, Dict
from langchain.schema import Document  # Assuming the docs are of this type

class RAGChain:
    def __init__(self, model, retriever) -> None:
        """
        Initializes the RAGChain class with a model and a retriever.

        Args:
            model: The model instance used for generating answers (e.g., a language model like LlamaModel).
            retriever: The retriever instance used for fetching relevant documents based on the user's query.
        """
        self.model = model
        self.retriever = retriever

    def format_docs(self, docs: List[Document]) -> str:
        """
        Formats a list of retrieved documents into a single string.

        Args:
            docs (List[Document]): A list of documents retrieved by the retriever.

        Returns:
            str: The formatted text combining all document contents.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_answer(self, user_message: str, temperature: float = 0.7, max_length: int = 1024) -> Tuple[Dict[str, str], List[Document]]:
        """
        Generates an answer to the user's query by retrieving relevant documents and passing them to the model for context.

        Args:
            user_message (str): The user's question or query.
            temperature (float, optional): Sampling temperature for text generation. Controls randomness. Defaults to 0.7.
            max_length (int, optional): Maximum length of the generated text. Defaults to 1024.

        Returns:
            Tuple[Dict[str, str], List[Document]]: A tuple containing the generated response and the retrieved documents.
        """
        # Retrieve relevant documents using the retriever
        retrieved_docs = self.retriever.invoke(user_message)
        
        # Format the retrieved documents into a string to provide as context
        relevant_content = self.format_docs(retrieved_docs)

        # Create a prompt with the relevant content and user message
        message_format = "Question: {user_message} Context: {relevant_content} Answer:"
        prompt = self.model.create_prompt(
            user_message=user_message,
            message_format=message_format,
            relevant_content=relevant_content,  # Pass the retrieved content as a keyword argument
            system_message=(
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Use three sentences maximum and keep the answer concise."
            )
        )

        # Generate a response using the model and the created prompt
        response = self.model.create(prompt, temperature=temperature, max_length=max_length)

        # Return the response and the retrieved documents
        return response, retrieved_docs
