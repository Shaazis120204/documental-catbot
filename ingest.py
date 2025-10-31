import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile

def extract_text_from_image(image_path):
    """Extract text from images using Tesseract OCR"""
    try:
        # Check if it's a PDF (which might contain images)
        if image_path.lower().endswith('.pdf'):
            # Convert PDF to images and extract text from each page
            images = convert_from_path(image_path)
            extracted_text = ""
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                extracted_text += f"Page {i+1}:\n{text}\n\n"
            return extracted_text
        else:
            # Regular image file
            return pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        print(f"OCR Error with {image_path}: {e}")
        return f"Could not extract text from image: {str(e)}"

def ingest_documents():
    # Use reliable embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    documents = []
    docs_path = "docs"
    
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        print("Created docs folder. Please add documents and run again.")
        return
    
    for file in os.listdir(docs_path):
        file_path = os.path.join(docs_path, file)
        
        try:
            if file.endswith('.pdf'):
                # Try OCR first for PDFs that might be image-based
                try:
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata.update({'source': file, 'type': 'text'})
                    documents.extend(loaded_docs)
                    print(f"Loaded PDF as text: {file}")
                except Exception as pdf_error:
                    # If text extraction fails, try OCR
                    print(f"PDF text extraction failed for {file}, trying OCR: {pdf_error}")
                    pdf_text = extract_text_from_image(file_path)
                    doc = Document(
                        page_content=pdf_text,
                        metadata={"source": file, "type": "pdf_ocr"}
                    )
                    documents.append(doc)
                    print(f"Loaded PDF with OCR: {file}")
                
            elif file.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata.update({'source': file, 'type': 'text'})
                documents.extend(loaded_docs)
                print(f"Loaded TXT: {file}")
                
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata.update({'source': file, 'type': 'text'})
                documents.extend(loaded_docs)
                print(f"Loaded DOCX: {file}")
                
            elif file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Handle images with OCR
                image_text = extract_text_from_image(file_path)
                doc = Document(
                    page_content=image_text,
                    metadata={"source": file, "type": "image_ocr"}
                )
                documents.append(doc)
                print(f"Processed image with OCR: {file}")
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not documents:
        print("No documents found or failed to process documents!")
        return
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_store"
    )
    
    print("Documents ingested successfully!")
    print(f"Vector store created with {len(chunks)} chunks")

if __name__ == "__main__":
    ingest_documents()