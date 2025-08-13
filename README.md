How the Chatbot Works 🤖
This application is a Retrieval-Augmented Generation (RAG) medical chatbot built with Python and Flask. It provides a web-based interface for users to ask medical questions. Unlike a simple large language model (LLM), this chatbot first retrieves relevant information from a specific knowledge base before generating a response, which helps ensure the answers are grounded in facts and reduces hallucinations.

The application's backend is powered by a Flask server that performs the following steps:

Data Loading: On startup, the app loads the AfriMed-QA dataset from Hugging Face, a benchmark dataset of medical questions and rationales.

Vector Database: It uses a specialized biomedical embedding model (pubmedbert-base-embeddings) to convert the dataset's questions, answers, and rationales into numerical representations (embeddings). These are then stored in a fast vector database (FAISS) for efficient searching.

User Interaction: When a user submits a question via the web interface, the app uses the same embedding model to create an embedding of the user's query.

Retrieval: The app searches the vector database for the top three most semantically similar documents from the dataset.

Generation: The retrieved documents are combined with the user's question into a single, strict prompt. This prompt is then sent to a generative language model (gpt-neo-125m) which synthesizes an answer based only on the provided context. If the answer isn't in the context, it's instructed to say it cannot find a response.

The frontend  handles user input and displays the chatbot's responses asynchronously.







