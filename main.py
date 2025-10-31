import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import requests
import os
import time
import json
from datetime import datetime
from ingest import ingest_documents
import gdown
import re
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Set page config
st.set_page_config(
    page_title="ajs-docs-bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tesseract installation path for Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    # Try to find Tesseract in common installation paths
    possible_paths = [
        TESSERACT_PATH,
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Users\*\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        r"/usr/bin/tesseract",  # Linux/Mac
        r"/usr/local/bin/tesseract"  # Linux/Mac
    ]
    
    tesseract_found = False
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            break
    
    if not tesseract_found:
        st.sidebar.warning("Tesseract not found. OCR functionality may be limited!\n"
                           "Download from https://github.com/ub-mannheim/tesseract/wiki")
        # Don't stop the app, just show a warning

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f3d7a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        height: 60vh;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 0.75rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #f1f1f1;
        padding: 0.75rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .history-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.5rem;
        cursor: pointer;
        border: 1px solid #e0e0e0;
    }
    .history-item:hover {
        background-color: #f0f0f0;
    }
    .badge {
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .badge-text { background-color: #e3f2fd; color: #1976d2; }
    .badge-image { background-color: #fff8e1; color: #ff8f00; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

# Check if Ollama is running
def check_ollama_service():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        return response.status_code == 200
    except:
        return False

# Check if model is available
def check_ollama_model(model_name="deepseek-r1:1.5b"):
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(model["name"] == model_name for model in models)
    except:
        return False

# Pull model
def pull_ollama_model(model_name="deepseek-r1:1.5b"):
    try:
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model_name},
            timeout=300,
            stream=True
        )
        return response.status_code == 200
    except:
        return False

# Initialize system
def initialize_system():
    try:
        if not os.path.exists("vector_store"):
            if os.path.exists("docs") and any(os.listdir("docs")):
                with st.spinner("Processing documents..."):
                    ingest_documents()
            else:
                return False
        
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vector_store = Chroma(
            persist_directory="vector_store",
            embedding_function=embeddings
        )
        
        prompt_template = """You are a helpful assistant that answers questions based on the provided context.

Context: {context}

Question: {question}

Answer based only on the context. If unsure, say "I don't have enough information."

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        llm = Ollama(
            model="deepseek-r1:1.5b",
            temperature=0.1,
            num_predict=512,
            base_url="http://localhost:11434"
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        st.session_state.vector_store = vector_store
        st.session_state.qa_chain = qa_chain
        st.session_state.system_ready = True
        return True
        
    except Exception as e:
        st.error(f"Error initializing: {e}")
        return False

# Process uploaded files
def process_uploaded_files(uploaded_files):
    docs_path = "docs"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(docs_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    try:
        ingest_documents()
        initialize_system()
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

# Download from Google Drive
def download_from_drive(drive_link):
    try:
        # Extract file ID from Google Drive link
        file_id = None
        if "drive.google.com" in drive_link:
            if "/file/d/" in drive_link:
                file_id = drive_link.split("/file/d/")[1].split("/")[0]
            elif "id=" in drive_link:
                file_id = drive_link.split("id=")[1].split("&")[0]
        
        if file_id:
            output_path = os.path.join("docs", f"drive_file_{file_id}")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
            ingest_documents()
            initialize_system()
            return True
    except Exception as e:
        st.error(f"Drive download failed: {e}")
    return False

# Create new chat
def create_new_chat():
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.chat_history[chat_id] = {
        "title": f"Chat {len(st.session_state.chat_history) + 1}",
        "messages": [],
        "created_at": datetime.now().isoformat()
    }
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    return chat_id

# Load chat history
def load_chat(chat_id):
    if chat_id in st.session_state.chat_history:
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = st.session_state.chat_history[chat_id]["messages"]
    else:
        st.session_state.current_chat_id = create_new_chat()

# Save current chat
def save_current_chat():
    if st.session_state.current_chat_id and st.session_state.messages:
        st.session_state.chat_history[st.session_state.current_chat_id]["messages"] = st.session_state.messages
        st.session_state.chat_history[st.session_state.current_chat_id]["updated_at"] = datetime.now().isoformat()

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>ajs-docs-bot</h2>", unsafe_allow_html=True)
    
    # Chat history section
    st.markdown("### 💬 Chat History")
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        create_new_chat()
    
    for chat_id, chat_data in st.session_state.chat_history.items():
        if st.button(
            f"{chat_data['title']}", 
            key=f"chat_{chat_id}",
            use_container_width=True
        ):
            load_chat(chat_id)
    
    st.divider()
    
    # File upload
    st.markdown("### 📁 Upload Files")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx", "png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files and st.button("Process Files", use_container_width=True):
        with st.spinner("Processing..."):
            if process_uploaded_files(uploaded_files):
                st.success("Files processed!")
                time.sleep(1)
                st.rerun()
    
    # Google Drive integration
    st.markdown("### 🌐 Google Drive")
    drive_link = st.text_input("Paste Google Drive link", label_visibility="collapsed")
    if drive_link and st.button("Download from Drive", use_container_width=True):
        with st.spinner("Downloading..."):
            if download_from_drive(drive_link):
                st.success("File downloaded!")
                time.sleep(1)
                st.rerun()
    
    st.divider()
    
    # System status
    st.markdown("### ⚙️ System Status")
    if check_ollama_service():
        st.success("✅ Ollama: Running")
        if check_ollama_model("deepseek-r1:1.5b"):
            st.success("✅ DeepSeek: Available")
        else:
            st.warning("⚠️ DeepSeek: Not downloaded")
    else:
        st.error("❌ Ollama: Not running")
    
    # Tesseract status
    try:
        pytesseract.get_tesseract_version()
        st.success("✅ Tesseract: Available")
    except:
        st.warning("⚠️ Tesseract: Not available")
    
    if st.session_state.system_ready:
        st.success("✅ System: Ready")
    else:
        st.warning("⚠️ System: Needs initialization")
    
    if st.button("🔄 Initialize System", use_container_width=True):
        with st.spinner("Initializing..."):
            if not check_ollama_service():
                st.error("Start Ollama first: 'ollama serve'")
            elif not check_ollama_model("deepseek-r1:1.5b"):
                if pull_ollama_model("deepseek-r1:1.5b"):
                    st.success("DeepSeek downloaded!")
                else:
                    st.error("Download failed")
            elif initialize_system():
                st.success("System ready!")
            else:
                st.error("Initialization failed")

# Main content
st.markdown("<h1 class='main-header'>ajs-docs-bot</h1>", unsafe_allow_html=True)

# Chat container
chat_container = st.container(height=400)

# Display messages
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    # Initialize if needed
    if not st.session_state.system_ready:
        initialize_system()
    
    # Create new chat if none exists
    if not st.session_state.current_chat_id:
        create_new_chat()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_current_chat()
    
    # Get response
    if st.session_state.system_ready:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain({"query": prompt})
                response = result["result"]
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                save_current_chat()
                st.rerun()
                
            except Exception as e:
                error_msg = "I'm still getting ready. Please try again."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                save_current_chat()
                st.rerun()
    else:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Please initialize the system first using the sidebar button."
        })
        save_current_chat()
        st.rerun()

# Initialize on first load
if not st.session_state.system_ready and check_ollama_service():
    initialize_system()