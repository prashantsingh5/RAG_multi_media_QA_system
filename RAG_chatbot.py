import os
import pandas as pd
import gradio as gr
import re
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import Video

# Load environment variables
load_dotenv()

# Get API key from environment variables
GEMINI_API_KEY = "AIzaSyADtXOCgwP1REFp5gyH6FjIUNH0vxeKjx8"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

class RAGChatbot:
    def __init__(self):
        """Initialize the RAG chatbot that handles both PDFs and YouTube videos."""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
            
        # Initialize embeddings model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY,
            credentials=None  # Explicitly set credentials to None to use API key
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize vector store and qa_chain to None
        self.vector_store = None
        self.qa_chain = None
        self.source_name = None  # To track current source (PDF name or YouTube URL)
        self.source_type = None  # "pdf" or "youtube"
        
    # Rest of your code remains the same
    def save_conversation(self, question, answer):
        """Save the conversation to a CSV file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a DataFrame with the new conversation
        new_conversation = pd.DataFrame({
            'Timestamp': [timestamp],
            'Source_Type': [self.source_type],
            'Source_Name': [self.source_name],
            'Question': [question],
            'Answer': [answer]
        })
        
        # Define the CSV file path
        csv_file = 'conversation_history.csv'
        
        # If file exists, append without header; if not, create with header
        if os.path.exists(csv_file):
            new_conversation.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            new_conversation.to_csv(csv_file, mode='w', header=True, index=False)
            
        return "Conversation saved to history."
    
    def process_pdf(self, pdf_file):
        """Process a PDF file and create a knowledge base."""
        try:
            # Save uploaded file temporarily
            temp_path = "temp_upload.pdf"
            
            # Handle different file upload formats from Gradio
            if hasattr(pdf_file, 'name'):
                # For Gradio's UploadFile object
                pdf_path = pdf_file.name
                # Use the file directly since it's already saved by Gradio
                temp_path = pdf_path
            else:
                # For raw bytes
                with open(temp_path, "wb") as f:
                    f.write(pdf_file)
            
            # Load PDF
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            
            # Split text into chunks
            splits = self.text_splitter.split_documents(pages)
            
            # Create embeddings and vector store
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            
            # Set source info
            self.source_type = "pdf"
            self.source_name = os.path.basename(temp_path)
            
            # Remove temporary file
            os.remove(temp_path)
            
            # Set up PDF-specific prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that answers questions based on the provided PDF content.
                Answer the question using only the context provided. If you're unsure or the answer isn't in 
                the context, say "I don't have enough information to answer that question."
                
                Context: {context}"""),
                ("human", "{input}")
            ])
            
            # Create retrieval chain
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            # Create document chain
            document_chain = create_stuff_documents_chain(self.llm, prompt)
            
            # Create retrieval chain
            self.qa_chain = create_retrieval_chain(retriever, document_chain)
            
            return "PDF processed successfully! You can now ask questions about its content."
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    def extract_video_id(self, youtube_url):
        """Extract the video ID from a YouTube URL."""
        # Match various YouTube URL formats
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        # If it's just the ID
        if len(youtube_url) == 11:
            return youtube_url
        
        return None
    
    def get_video_info(self, video_id):
        """Get video title and description."""
        try:
            video_info = Video.getInfo(video_id)
            return {
                'title': video_info['title'],
                'description': video_info['description']
            }
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {'title': 'Unknown', 'description': ''}
    
    def get_transcript(self, youtube_url):
        """Get the transcript of a YouTube video."""
        video_id = self.extract_video_id(youtube_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([entry['text'] for entry in transcript_list])
            
            # Get video metadata to enrich context
            video_info = self.get_video_info(video_id)
            
            # Combine metadata with transcript
            full_context = f"Title: {video_info['title']}\n\nDescription: {video_info['description']}\n\nTranscript: {transcript_text}"
            
            return full_context
        except Exception as e:
            raise Exception(f"Error fetching transcript: {e}")
    
    def process_youtube(self, youtube_url):
        """Process a YouTube video and create a knowledge base."""
        try:
            # Get the transcript
            transcript = self.get_transcript(youtube_url)
            
            # Split the text into chunks
            chunks = self.text_splitter.split_text(transcript)
            
            # Create vector store from chunks
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
            
            # Set source info
            self.source_type = "youtube"
            self.source_name = youtube_url
            
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create YouTube-specific prompt template
            template = """
            You are an AI assistant that answers questions based on the content of a YouTube video.
            Use only the context provided to answer the question. If you don't have enough information,
            just say "I don't have enough information from the video to answer this question."

            Context from video:
            {context}

            Question: {question}
            
            Answer:
            """
            
            PROMPT = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            return "YouTube video processed successfully! You can now ask questions about its content."
            
        except Exception as e:
            return f"Error processing YouTube video: {str(e)}"
    
    def answer_question(self, question):
        """Answer a question based on the loaded content."""
        if not self.vector_store or not self.qa_chain:
            return "Please upload a PDF or provide a YouTube URL first."
        
        try:
            if self.source_type == "pdf":
                response = self.qa_chain.invoke({"input": question})
                answer = response["answer"]
            else:  # YouTube
                result = self.qa_chain.invoke({"query": question})
                answer = result["result"]
            
            # Save conversation to history
            self.save_conversation(question, answer)
            
            return answer
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_conversation_history(self):
        """Get the conversation history from CSV file."""
        try:
            if os.path.exists('conversation_history.csv'):
                history = pd.read_csv('conversation_history.csv')
                return history.to_string(index=False)
            else:
                return "No conversation history yet."
        except Exception as e:
            return f"Error loading conversation history: {str(e)}"


# Create an instance of the RAG chatbot
chatbot = RAGChatbot()

def process_content(pdf_file, youtube_url):
    """Process either a PDF file or YouTube URL."""
    # Clear previous knowledge base
    chatbot.vector_store = None
    chatbot.qa_chain = None
    
    if pdf_file is not None:
        # Process PDF - Gradio file components return a file object or None
        return chatbot.process_pdf(pdf_file)
    elif youtube_url:
        # Process YouTube video
        return chatbot.process_youtube(youtube_url)
    else:
        return "Please upload a PDF file or provide a YouTube URL."

def chat(message, history):
    """Handle chat interactions."""
    if not chatbot.vector_store or not chatbot.qa_chain:
        return "Please upload a PDF or provide a YouTube URL first."
    
    response = chatbot.answer_question(message)
    return response

def view_history():
    """View conversation history."""
    return chatbot.get_conversation_history()

# Create Gradio Interface
with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("# RAG Chatbot")
    gr.Markdown("Upload a PDF file or provide a YouTube URL to chat with its content.")
    
    with gr.Tab("Upload Content"):
        with gr.Row():
            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            youtube_input = gr.Textbox(label="YouTube URL")
        
        process_btn = gr.Button("Process Content")
        process_output = gr.Textbox(label="Processing Result")
        
        process_btn.click(process_content, inputs=[pdf_upload, youtube_input], outputs=process_output)
    
    with gr.Tab("Chat"):
        chatbot_interface = gr.ChatInterface(chat)
    
    with gr.Tab("View History"):
        view_history_btn = gr.Button("View Conversation History")
        history_output = gr.Textbox(label="Conversation History")
        
        view_history_btn.click(view_history, inputs=[], outputs=history_output)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)