# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Google's Gemini model that can answer questions based on PDF documents and YouTube video content.

## Features

- **PDF Document Analysis**: Upload and query information from any PDF document.
- **YouTube Video Analysis**: Provide a YouTube URL to analyze and question its content using the video's transcript.
- **Conversation History**: Track and save all interactions for future reference.
- **Simple User Interface**: Easy-to-use Gradio interface accessible through any web browser.

## How It Works

This application uses a RAG (Retrieval-Augmented Generation) architecture to provide accurate responses:

1. **Document Ingestion**: PDFs or YouTube transcripts are split into chunks and embedded using Gemini's embedding model.
2. **Vector Storage**: Document embeddings are stored in a FAISS vector database for efficient similarity search.
3. **Retrieval**: When a question is asked, the system retrieves the most relevant chunks from the vector store.
4. **Generation**: Gemini 1.5 Pro generates a response based on the retrieved context and the user's question.

## Requirements

- Python 3.8+
- Google Gemini API key
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860).

3. Use the interface to:
   - Upload a PDF file or enter a YouTube URL
   - Process the content
   - Ask questions about the content in the chat tab
   - View conversation history

## Sample Workflow

1. **Upload Content Tab**:
   - Upload a PDF file or paste a YouTube URL
   - Click "Process Content"
   - Wait for the confirmation message

2. **Chat Tab**:
   - Ask questions about the processed content
   - Receive AI-generated answers based on the content

3. **View History Tab**:
   - Click "View Conversation History" to see all past interactions

## File Structure

```
rag-chatbot/
├── app.py              # Main application file
├── .env                # Environment variables (create this file)
├── requirements.txt    # Required Python packages
├── conversation_history.csv  # Auto-generated history file
└── README.md           # This file
```

## Requirements.txt

Create a `requirements.txt` file with the following dependencies:

```
pandas
gradio
python-dotenv
google-generativeai
langchain
langchain-community
langchain-google-genai
faiss-cpu
pypdf
youtube-transcript-api
youtube-search-python
```

## Limitations

- YouTube analysis relies on available transcripts. Videos without transcripts cannot be processed.
- PDF processing may be limited for scanned PDFs without proper text layers.
- Large PDFs or very long videos might require more processing time.

## Future Improvements

- Add support for more document types (DOCX, TXT, etc.)
- Implement memory management for very large documents
- Add authentication layer for multi-user environments
- Create visualization for retrieved document chunks

## License

[MIT License](LICENSE)

## Acknowledgements

- [Google Gemini](https://ai.google.dev/)
- [LangChain](https://python.langchain.com/)
- [Gradio](https://gradio.app/)
- [FAISS](https://github.com/facebookresearch/faiss)
