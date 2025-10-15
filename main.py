from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import re
from collections import Counter
import math
import base64
import io
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
import uvicorn
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Lifespan events for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading PDFs...")
    await load_pdfs_on_startup()
    yield
    # Shutdown (if needed)
    print("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Mining Safety Chatbot API", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

# Global variables
EMBEDDED_PDFS = {}
vector_store = []

# Pydantic models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class HealthResponse(BaseModel):
    status: str
    pdfs_loaded: int
    chunks_available: int

# Helper functions (same as before)
def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    current_chunk = ''
    
    for sentence in sentences:
        sentence_with_punct = sentence + '.'
        
        if len(current_chunk + sentence_with_punct) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            chunk_sentences = re.split(r'[.!?]+', current_chunk)
            if len(chunk_sentences) >= 2:
                overlap_text = '. '.join(chunk_sentences[-2:]) + '.'
                current_chunk = overlap_text + ' ' + sentence_with_punct
            else:
                current_chunk = sentence_with_punct
        else:
            current_chunk += ' ' + sentence_with_punct
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def create_simple_embedding(text):
    """Create a simple word frequency embedding"""
    clean_text = text.lower()
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    words = re.findall(r'\b\w{2,}\b', clean_text)
    stemmed_words = [re.sub(r'(ing|ed|s|es|tion|ment)$', '', word) for word in words]
    
    word_freq = Counter(stemmed_words)
    return dict(word_freq)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity"""
    all_words = set(list(vec1.keys()) + list(vec2.keys()))
    
    dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in all_words)
    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    return dot_product / (mag1 * mag2)

def extract_text_from_pdf_bytes(pdf_bytes):
    """Extract text from PDF bytes"""
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_embedded_pdfs():
    """Load all embedded PDFs from the EMBEDDED_PDFS dictionary"""
    all_chunks = []
    loaded_files = []
    
    for pdf_name, encoded_pdf in EMBEDDED_PDFS.items():
        if not encoded_pdf or encoded_pdf.strip() == "":
            continue
            
        try:
            pdf_bytes = base64.b64decode(encoded_pdf)
            text = extract_text_from_pdf_bytes(pdf_bytes)
            chunks = chunk_text(text)
            
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'embedding': create_simple_embedding(chunk),
                    'source': pdf_name,
                    'chunk_id': idx
                })
            
            loaded_files.append(pdf_name)
            
        except Exception as e:
            print(f"Error loading {pdf_name}: {e}")
    
    return all_chunks, loaded_files

def retrieve_relevant_chunks(query, top_k=4):
    """Retrieve most relevant chunks"""
    if not vector_store:
        return []
    
    query_embedding = create_simple_embedding(query)
    
    scored_chunks = []
    for chunk in vector_store:
        similarity = cosine_similarity(query_embedding, chunk['embedding'])
        scored_chunks.append({**chunk, 'similarity': similarity})
    
    scored_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    return scored_chunks[:top_k]

def query_openai(prompt, context=""):
    """Query OpenAI GPT model"""
    try:
        system_message = """You are Damodar, a helpful mining safety assistant. Answer questions based on the provided context from mining safety documents. 
        Provide clear, accurate answers based on the context. If the context doesn't contain enough information, you can ans by yourself also."""
        
        user_message = f"""Context from documents:
{context}

Question: {prompt}"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

async def load_pdfs_on_startup():
    """Load PDFs on startup"""
    global vector_store, EMBEDDED_PDFS
    
    pdf_files = [f'Mining{i}.pdf' for i in range(1, 11)]
    
    for filename in pdf_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    EMBEDDED_PDFS[filename] = encoded
                print(f"✓ Loaded {filename}")
            except Exception as e:
                print(f"✗ Failed to load {filename}: {e}")
    
    if EMBEDDED_PDFS:
        chunks, pdf_files = load_embedded_pdfs()
        vector_store = chunks
        print(f"\n✓ Successfully loaded {len(EMBEDDED_PDFS)} PDFs with {len(vector_store)} chunks")
    else:
        print("\n⚠ Warning: No PDFs found!")

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Mining Safety Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Ask a question",
            "GET /health": "Check API health",
            "POST /upload-pdf": "Upload a PDF document"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        pdfs_loaded=len(EMBEDDED_PDFS),
        chunks_available=len(vector_store)
    )

@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask a question to the chatbot"""
    if not vector_store:
        raise HTTPException(
            status_code=503,
            detail="PDFs not loaded yet. Please wait."
        )
    
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        relevant_chunks = retrieve_relevant_chunks(request.question)
        
        if not relevant_chunks:
            return QueryResponse(
                answer="I couldn't find relevant information in the mining PDFs. Please try rephrasing your question.",
                sources=[]
            )
        
        context = "\n\n".join([
            f"[From {chunk['source']}]\n{chunk['text']}"
            for chunk in relevant_chunks
        ])
        
        answer = query_openai(request.question, context)
        sources = list(set([chunk['source'] for chunk in relevant_chunks]))
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a new PDF to the system"""
    global vector_store, EMBEDDED_PDFS
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )
    
    try:
        pdf_bytes = await file.read()
        encoded = base64.b64encode(pdf_bytes).decode('utf-8')
        EMBEDDED_PDFS[file.filename] = encoded
        
        # Reload vector store
        chunks, pdf_files = load_embedded_pdfs()
        vector_store = chunks
        
        return {
            "message": f"Successfully uploaded {file.filename}",
            "total_pdfs": len(EMBEDDED_PDFS),
            "total_chunks": len(vector_store)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading PDF: {str(e)}"
        )

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
