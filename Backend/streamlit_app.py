import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Import project modules
from chunking import chunk_text
from embedding_service import generate_embedding
from vector_store import store_embeddings, search_embeddings
from rag_service import generate_answer

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Resume Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .section-header {
        color: #667eea;
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if "resume_uploaded" not in st.session_state:
    st.session_state.resume_uploaded = False
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None
if "resume_filename" not in st.session_state:
    st.session_state.resume_filename = None

# Header
st.markdown("""
    <div class="main-header">
        <h1>📄 AI Resume Assistant</h1>
        <p>Upload your resume and ask questions using AI-powered search</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.markdown("## 📤 Upload Resume")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload your resume in PDF format"
)

if uploaded_file is not None:
    if st.sidebar.button("📤 Process Resume", use_container_width=True):
        with st.sidebar.status("Processing...", expanded=True) as status:
            try:
                # Read PDF
                st.write("📖 Reading PDF...")
                reader = PdfReader(uploaded_file)
                text = ""
               
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
               
                st.write(f"✅ Extracted text from {len(reader.pages)} pages")
               
                # Chunk text
                st.write("✂️ Chunking text...")
                chunks = chunk_text(text)
                st.write(f"✅ Created {len(chunks)} chunks")
               
                # Generate embeddings
                st.write("🔢 Generating embeddings...")
                embeddings = []
                progress_bar = st.progress(0)
               
                for i, chunk in enumerate(chunks):
                    vector = generate_embedding(chunk)
                    embeddings.append(vector)
                    progress_bar.progress((i + 1) / len(chunks))
               
                st.write(f"✅ Generated {len(embeddings)} embeddings")
               
                # Store embeddings
                st.write("💾 Storing in vector database...")
                store_embeddings(chunks, embeddings)
                st.write("✅ Stored successfully")
               
                # Update session state
                st.session_state.resume_uploaded = True
                st.session_state.upload_status = "success"
                st.session_state.resume_filename = uploaded_file.name
                st.session_state.resume_text = text
                st.session_state.chunks = chunks
               
                status.update(label="✅ Processing Complete", state="complete")
               
            except Exception as e:
                st.session_state.upload_status = "error"
                status.update(label="❌ Processing Failed", state="error")
                st.error(f"Error processing file: {str(e)}")

# Display upload status
if st.session_state.resume_uploaded:
    st.sidebar.markdown("---")
    st.sidebar.success(f"✅ **Resume Uploaded**\n\n📄 {st.session_state.resume_filename}")
    st.sidebar.info(f"📊 Chunks: {len(st.session_state.chunks)}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="section-header">❓ Ask Questions</h2>', unsafe_allow_html=True)
   
    if st.session_state.resume_uploaded:
        st.markdown("""
            <div class="info-box">
                <strong>💡 Tip:</strong> Ask questions about your resume, such as "What are my main skills?" or "What companies have I worked for?"
            </div>
        """, unsafe_allow_html=True)
       
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="E.g., What are my technical skills?",
            help="Type a question about your resume"
        )
       
        # Answer button
        if question:
            if st.button("🔍 Search & Answer", use_container_width=True):
                with st.spinner("🤖 Thinking..."):
                    try:
                        # Generate embedding for question
                        question_embedding = generate_embedding(question)
                       
                        # Search for relevant chunks
                        results = search_embeddings(question_embedding, top_k=3)
                       
                        # Combine context from results
                        context = "\n\n".join(results['documents'][0]) if results['documents'] else ""
                       
                        if not context:
                            st.warning("⚠️ No relevant information found in your resume.")
                        else:
                            # Generate answer
                            answer = generate_answer(question, context)
                           
                            # Display answer
                            st.markdown("""
                                <div class="answer-box">
                                    <h3>✨ Answer</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            st.markdown(answer)
                           
                            # Show source context
                            with st.expander("📚 View Source Context", expanded=False):
                                st.markdown("**Relevant sections from your resume:**")
                                for i, chunk in enumerate(results['documents'][0], 1):
                                    st.markdown(f"**Chunk {i}:**")
                                    st.text(chunk)
                                    st.divider()
                   
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
   
    else:
        st.markdown("""
            <div class="info-box">
                <strong>📌 Getting Started:</strong><br>
                1. Upload a PDF resume using the sidebar<br>
                2. Wait for processing to complete<br>
                3. Ask questions about your resume<br>
                4. Get AI-powered answers instantly
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-header">ℹ️ About</h2>', unsafe_allow_html=True)
   
    st.markdown("""
    ### How it Works
   
    1. **Upload** - Submit your PDF resume
    2. **Process** - Text is chunked and embedded
    3. **Store** - Embeddings saved in vector DB
    4. **Query** - Ask questions about your resume
    5. **Answer** - AI generates contextual responses
   
    ### Tech Stack
    - 📚 **ChromaDB** - Vector database
    - 🔢 **Google Embeddings** - Text embeddings
    - 🤖 **gemini-3.5-flash** - Answer generation
    - ⚡ **Streamlit** - UI framework
    """)
   
    st.markdown("---")
    st.markdown("""
    ### Tips
    - Be specific with questions
    - Use resume keywords
    - Ask follow-up questions
    - Review source context
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>AI Resume Assistant © 2024 | Powered by Google AI</p>",
    unsafe_allow_html=True
)
