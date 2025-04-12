import os
import re
import pandas as pd
import streamlit as st
from datetime import datetime
import tempfile
import io
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import Video
import docx2txt
import PyPDF2

# Load environment variables
load_dotenv()

# Get API key from environment variables
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Set up Streamlit page configuration
st.set_page_config(
    page_title="RAG System with Summarization",
    page_icon="📄",
    layout="wide"
)

# Initialize session state for storing results
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'source_name' not in st.session_state:
    st.session_state.source_name = None
if 'source_type' not in st.session_state:
    st.session_state.source_type = None
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""
if 'english_summary' not in st.session_state:
    st.session_state.english_summary = ""
if 'hindi_summary' not in st.session_state:
    st.session_state.hindi_summary = ""
if 'youtube_transcript' not in st.session_state:
    st.session_state.youtube_transcript = ""
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system that handles PDFs, URLs, and YouTube videos."""
        if not GEMINI_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
        # Initialize embeddings model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
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
        self.source_name = None  # To track current source (PDF name, YouTube URL, or web URL)
        self.source_type = None  # "pdf", "youtube", or "url"
        
    def save_conversation(self, question, answer):
        """Save the conversation to history."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to conversation history
        conversation = {
            'Timestamp': timestamp,
            'Source_Type': self.source_type,
            'Source_Name': self.source_name,
            'Question': question,
            'Answer': answer
        }
        
        st.session_state.conversation_history.append(conversation)
        
        # Also save to CSV (optional)
        csv_file = 'conversation_history.csv'
        new_conversation = pd.DataFrame([conversation])
        
        if os.path.exists(csv_file):
            new_conversation.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            new_conversation.to_csv(csv_file, mode='w', header=True, index=False)
            
        return "Conversation saved to history."
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            pdf_file.seek(0)  # Reset file pointer
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def process_pdf(self, pdf_file):
        """Process a PDF file and create a knowledge base."""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_file.read())
                temp_path = temp_file.name
                pdf_file.seek(0)  # Reset file pointer
            
            # Extract text for potential summarization
            extracted_text = self.extract_text_from_pdf(pdf_file)
            st.session_state.document_text = extracted_text
            
            # Load PDF
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            
            # Split text into chunks
            splits = self.text_splitter.split_documents(pages)
            
            # Create embeddings and vector store
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            st.session_state.vector_store = self.vector_store
            
            # Set source info
            self.source_type = "pdf"
            st.session_state.source_type = "pdf"
            self.source_name = pdf_file.name
            st.session_state.source_name = pdf_file.name
            
            # Remove temporary file
            os.unlink(temp_path)
            
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
            st.session_state.qa_chain = self.qa_chain
            
            return "PDF processed successfully!"
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    def process_urls(self, urls):
        """Process URLs and create a knowledge base."""
        try:
            # Load URLs
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
            
            # Also store the raw text for potential summarization
            combined_text = "\n\n".join([doc.page_content for doc in data])
            st.session_state.document_text = combined_text
            
            # Split text into chunks
            splits = self.text_splitter.split_documents(data)
            
            # Create embeddings and vector store
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            st.session_state.vector_store = self.vector_store
            
            # Set source info
            self.source_type = "url"
            st.session_state.source_type = "url"
            self.source_name = ", ".join(urls)
            st.session_state.source_name = ", ".join(urls)
            
            # Set up URL-specific prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that answers questions based on the provided web content.
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
            st.session_state.qa_chain = self.qa_chain
            
            return "URLs processed successfully!"
            
        except Exception as e:
            return f"Error processing URLs: {str(e)}"
    
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
            
            return full_context, video_info
        except Exception as e:
            raise Exception(f"Error fetching transcript: {e}")
    
    def process_youtube(self, youtube_url):
        """Process a YouTube video and create a knowledge base."""
        try:
            # Get the transcript
            transcript, video_info = self.get_transcript(youtube_url)
            
            # Store transcript for display
            st.session_state.youtube_transcript = transcript
            st.session_state.document_text = transcript
            
            # Split the text into chunks
            chunks = self.text_splitter.split_text(transcript)
            
            # Create vector store from chunks
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
            st.session_state.vector_store = self.vector_store
            
            # Set source info
            self.source_type = "youtube"
            st.session_state.source_type = "youtube"
            self.source_name = f"{video_info['title']} ({youtube_url})"
            st.session_state.source_name = f"{video_info['title']} ({youtube_url})"
            
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
            st.session_state.qa_chain = self.qa_chain
            
            return "YouTube video processed successfully!"
            
        except Exception as e:
            return f"Error processing YouTube video: {str(e)}"
    
    def generate_summary(self, text, language="English", max_words=500):
        """Generate summary in specified language."""
        if not text:
            return "No text to summarize."
        
        # Create prompts based on language
        if language.lower() == "english":
            prompt = f"""
            Generate a concise summary of the following text in English. 
            The summary should capture the main points and key details.
            Limit the summary to a maximum of {max_words} words.
            
            TEXT: {text}
            
            SUMMARY:
            """
        else:  # Hindi
            prompt = f"""
            Generate a concise summary of the following text in Hindi.
            The summary should capture the main points and key details.
            Limit the summary to a maximum of {max_words} words.
            
            TEXT: {text}
            
            SUMMARY (in Hindi):
            """
        
        try:
            # Split text if it's too long
            chunks = self.text_splitter.split_text(text)
            
            # If the text is too long, just use the first chunk for summary
            if len(chunks) > 1:
                st.warning("Document is long. Summarizing the first section...")
                text_to_summarize = chunks[0]
            else:
                text_to_summarize = text
                
            # Generate summary
            response = self.llm.invoke(prompt.replace("{text}", text_to_summarize))
            return response.content
        except Exception as e:
            st.error(f"Error generating summary: {e}")
            return f"Failed to generate summary: {str(e)}"
        
        
        
    def answer_question(self, question):
        """Answer a question based on the loaded content."""
        # Use session state directly
        if not st.session_state.vector_store or not st.session_state.qa_chain:
            return "Please upload content first (PDF, URL, or YouTube)."
        
        try:
            if st.session_state.source_type in ["pdf", "url"]:
                response = st.session_state.qa_chain.invoke({"input": question})
                answer = response["answer"]
            else:  # YouTube
                result = st.session_state.qa_chain({"query": question})
                answer = result["result"]
            
            # Save conversation to history
            self.save_conversation(question, answer)
            
            return answer
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    # def answer_question(self, question):
    #     """Answer a question based on the loaded content."""
    #     if not self.vector_store or not self.qa_chain:
    #         return "Please upload content first (PDF, URL, or YouTube)."
        
    #     try:
    #         if self.source_type in ["pdf", "url"]:
    #             response = self.qa_chain.invoke({"input": question})
    #             answer = response["answer"]
    #         else:  # YouTube
    #             result = self.qa_chain({"query": question})
    #             answer = result["result"]
            
    #         # Save conversation to history
    #         self.save_conversation(question, answer)
            
    #         return answer
    #     except Exception as e:
    #         return f"Error generating response: {str(e)}"

# Create an instance of the RAG system
rag_system = RAGSystem()

# Main app function
def main():
    if st.session_state.vector_store:
        rag_system.vector_store = st.session_state.vector_store
    if st.session_state.qa_chain:
        rag_system.qa_chain = st.session_state.qa_chain
    if st.session_state.source_name:
        rag_system.source_name = st.session_state.source_name
    if st.session_state.source_type:
        rag_system.source_type = st.session_state.source_type
    st.title("📚 RAG System with Summarization")
    st.markdown("Upload content (PDF, URLs, YouTube) for QA and summarization using Google Gemini")
    
    # Create tabs for different content sources and features
    tab1, tab2, tab3, tab4 = st.tabs(["Upload Content", "Summarize", "QA System", "History"])
    
    # Tab 1: Upload Content
    with tab1:
        st.header("Upload Content")
        
        source_type = st.radio("Select source type:", ["PDF", "URL", "YouTube"])
        
        if source_type == "PDF":
            uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
            
            if uploaded_file is not None and st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    result = rag_system.process_pdf(uploaded_file)
                    st.success(result)
                    
        elif source_type == "URL":
            urls_input = st.text_area("Enter URLs (one per line)")
            
            if urls_input and st.button("Process URLs"):
                urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
                if urls:
                    with st.spinner("Processing URLs..."):
                        result = rag_system.process_urls(urls)
                        st.success(result)
                else:
                    st.error("Please enter at least one valid URL")
                    
        elif source_type == "YouTube":
            youtube_url = st.text_input("Enter YouTube URL")
            
            if youtube_url and st.button("Process YouTube Video"):
                with st.spinner("Processing YouTube video..."):
                    result = rag_system.process_youtube(youtube_url)
                    st.success(result)
                    st.session_state.youtube_processed = True
                    
                    # Display YouTube transcript
                    if st.session_state.youtube_transcript:
                        with st.expander("View Video Transcript"):
                            st.text_area("Transcript", st.session_state.youtube_transcript, height=300)
    
    # Tab 2: Summarize
    with tab2:
        st.header("Generate Summaries")
        
        if st.session_state.document_text:
            st.success(f"Content available for summarization: {st.session_state.source_type} - {st.session_state.source_name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate English Summary"):
                    with st.spinner("Generating English summary..."):
                        st.session_state.english_summary = rag_system.generate_summary(
                            st.session_state.document_text, 
                            language="English"
                        )
                        
                if st.session_state.english_summary:
                    st.subheader("English Summary")
                    st.write(st.session_state.english_summary)
                    st.download_button(
                        label="Download English Summary",
                        data=st.session_state.english_summary,
                        file_name="english_summary.txt",
                        mime="text/plain"
                    )
            
            with col2:
                if st.button("Generate Hindi Summary"):
                    with st.spinner("Generating Hindi summary..."):
                        st.session_state.hindi_summary = rag_system.generate_summary(
                            st.session_state.document_text, 
                            language="Hindi"
                        )
                        
                if st.session_state.hindi_summary:
                    st.subheader("Hindi Summary")
                    st.write(st.session_state.hindi_summary)
                    st.download_button(
                        label="Download Hindi Summary",
                        data=st.session_state.hindi_summary,
                        file_name="hindi_summary.txt",
                        mime="text/plain"
                    )
        else:
            st.info("Please upload content first (PDF, URL, or YouTube) in the 'Upload Content' tab")
    
    # Tab 3: QA System
    with tab3:
        # st.header("Question Answering System")
        # Tab 3: QA System

        st.header("Question Answering System")
        
        # Sync the rag_system with session_state
        rag_system.vector_store = st.session_state.vector_store
        rag_system.qa_chain = st.session_state.qa_chain
        rag_system.source_name = st.session_state.source_name
        rag_system.source_type = st.session_state.source_type
        if st.session_state.vector_store and st.session_state.qa_chain:
            st.success(f"QA system ready for: {st.session_state.source_type} - {st.session_state.source_name}")
            
            query = st.text_input("Ask a question about the content")
            
            if query and st.button("Get Answer"):
                with st.spinner("Generating answer..."):
                    answer = rag_system.answer_question(query)
                    st.write("### Answer")
                    st.write(answer)
        else:
            st.info("Please upload content first (PDF, URL, or YouTube) in the 'Upload Content' tab")
    
    # Tab 4: History
    with tab4:
        st.header("Conversation History")
        
        if st.session_state.conversation_history:
            for i, conv in enumerate(st.session_state.conversation_history):
                with st.expander(f"Conversation {i+1} - {conv['Timestamp']} - {conv['Source_Type']}"):
                    st.write(f"**Source:** {conv['Source_Name']}")
                    st.write(f"**Question:** {conv['Question']}")
                    st.write(f"**Answer:** {conv['Answer']}")
        else:
            st.info("No conversation history yet")

    # App documentation
    with st.expander("📚 Application Documentation"):
        st.markdown("""
        ## RAG System with Summarization Documentation
        
        ### Overview
        This application combines Retrieval-Augmented Generation (RAG) with summarization capabilities for PDFs, websites, and YouTube videos using Google's Generative AI model (Gemini).
        
        ### Features
        - **Multiple Content Sources**:
          - PDF documents
          - Web URLs
          - YouTube videos (with transcript extraction)
        
        - **Summarization**:
          - Generate concise summaries in English
          - Generate concise summaries in Hindi
          - Download summaries as text files
        
        - **Question Answering**:
          - Ask questions about the uploaded content
          - Get AI-generated answers using RAG technology
        
        - **History Tracking**:
          - View conversation history
          - See timestamps and source information
        
        ### How to Use
        
        1. **Upload Content** (Tab 1):
           - Select content type (PDF, URL, or YouTube)
           - Upload or provide link(s)
           - Process the content
        
        2. **Generate Summaries** (Tab 2):
           - Once content is processed, generate summaries in English or Hindi
           - Download summaries for later use
        
        3. **Ask Questions** (Tab 3):
           - Enter questions about the content
           - Receive AI-generated answers based on the content
        
        4. **View History** (Tab 4):
           - Review past conversations
           - See questions and answers organized by source
        
        ### Technical Implementation
        
        - **RAG System**: Combines retrieval-based and generative approaches for accurate answers
        - **Vector Storage**: Uses FAISS for efficient similarity search
        - **LLM Integration**: Powered by Google's Gemini 1.5 Pro model
        - **Text Processing**: Handles chunking and embedding of document content
        """)

# Run the app
if __name__ == "__main__":
    main()