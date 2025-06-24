import streamlit as st
import os
import json
import fitz  # PyMuPDF
import pandas as pd
import requests
from dotenv import load_dotenv
import re
from typing import List, Dict, Optional
import tempfile
from datetime import datetime
import pymongo
from pymongo import MongoClient
import hashlib

# Load environment variables
load_dotenv()

# Get API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'maha_survey_db')

# Page config
st.set_page_config(
    page_title="Maharashtra Economic Survey Search",
    page_icon="📊",
    layout="wide"
)

# Initialize MongoDB connection
@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection"""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DATABASE_NAME]
        
        # Create collections if they don't exist
        if 'documents' not in db.list_collection_names():
            db.create_collection('documents')
            # Create indexes for better search performance
            db.documents.create_index([('filename', 1)])
            db.documents.create_index([('year', 1)])
        
        if 'chunks' not in db.list_collection_names():
            db.create_collection('chunks')
            # Create indexes
            db.chunks.create_index([('document_id', 1)])
            db.chunks.create_index([('year', 1)])
            db.chunks.create_index([('page', 1)])
        
        # Test connection
        db.command('ping')
        return db
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

# Initialize database
db = init_mongodb()

def extract_year_from_filename(filename: str) -> int:
    """Extract year from filename"""
    match = re.search(r'(\d{4})', filename)
    return int(match.group(1)) if match else None

def get_file_hash(content: bytes) -> str:
    """Generate hash of file content"""
    return hashlib.sha256(content).hexdigest()

def check_document_exists(filename: str, file_hash: str) -> bool:
    """Check if document already exists in MongoDB"""
    if db is None:
        return False
    
    existing = db.documents.find_one({
        '$or': [
            {'filename': filename},
            {'file_hash': file_hash}
        ]
    })
    return existing is not None

def pdf_to_txt(pdf_path: str) -> str:
    """Convert PDF to text and return text content"""
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        full_text += f"=== PAGE {page_num + 1} ===\n{text}\n\n"
    
    doc.close()
    return full_text

def pdf_to_chunks(pdf_path: str) -> List[Dict]:
    """Convert PDF to text chunks (1 page = 1 chunk)"""
    doc = fitz.open(pdf_path)
    chunks = []
    filename = os.path.basename(pdf_path)
    year = extract_year_from_filename(filename)
    
    # Create one chunk per page
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        # Clean the text
        text = text.strip()
        
        # Only add non-empty pages
        if text:
            chunks.append({
                'text': text,
                'page': page_num + 1,
                'year': year,
                'filename': filename
            })
    
    doc.close()
    return chunks

def get_embedding_via_api(text: str) -> List[float]:
    """Get embedding using direct API call"""
    try:
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'text-embedding-ada-002',
            'input': text[:8000]  # Limit text length
        }
        
        response = requests.post(
            'https://api.openai.com/v1/embeddings',
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        else:
            st.error(f"OpenAI API error: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        return None

def chat_completion_via_api(messages: List[Dict]) -> str:
    """Get chat completion using direct API call"""
    try:
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': messages,
            'temperature': 0.7
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.text}"
            
    except Exception as e:
        return f"Error generating response: {e}"

def save_to_mongodb(filename: str, chunks: List[Dict], file_hash: str, full_text: str):
    """Save document and chunks to MongoDB"""
    if db is None:
        st.error("MongoDB connection not available")
        return False
    
    try:
        # Save document metadata
        doc_data = {
            'filename': filename,
            'file_hash': file_hash,
            'year': extract_year_from_filename(filename),
            'total_pages': len(chunks),
            'full_text': full_text,
            'processed_at': datetime.now(),
            'chunks_count': len(chunks)
        }
        
        # Insert document
        doc_result = db.documents.insert_one(doc_data)
        document_id = doc_result.inserted_id
        
        # Save chunks
        chunks_to_insert = []
        for chunk in chunks:
            chunk_data = {
                'document_id': document_id,
                'filename': filename,
                'page': chunk['page'],
                'year': chunk['year'],
                'text': chunk['text'],
                'embedding': chunk.get('embedding', [])
            }
            chunks_to_insert.append(chunk_data)
        
        if chunks_to_insert:
            db.chunks.insert_many(chunks_to_insert)
        
        return True
    except Exception as e:
        st.error(f"Error saving to MongoDB: {e}")
        return False

def get_all_documents() -> List[Dict]:
    """Get all processed documents from MongoDB"""
    if db is None:
        return []
    
    try:
        documents = list(db.documents.find({}, {
            'filename': 1, 
            'year': 1, 
            'total_pages': 1, 
            'processed_at': 1,
            '_id': 0
        }))
        return documents
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
        return []

def process_pdf(uploaded_file):
    """Process a single PDF file"""
    # Get file content and hash
    file_content = uploaded_file.getbuffer()
    file_hash = get_file_hash(file_content)
    
    # Check if already processed
    if check_document_exists(uploaded_file.name, file_hash):
        st.info(f"✅ {uploaded_file.name} already processed")
        return
    
    with st.spinner(f"Processing {uploaded_file.name}..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        # Convert to text and extract chunks
        full_text = pdf_to_txt(tmp_path)
        chunks = pdf_to_chunks(tmp_path)
        
        if not chunks:
            st.error(f"No text content found in {uploaded_file.name}")
            os.unlink(tmp_path)
            return
        
        st.info(f"Processing {len(chunks)} pages from {uploaded_file.name}")
        
        # Get embeddings for each page
        progress_bar = st.progress(0)
        embedded_chunks = []
        failed_pages = []
        
        for i, chunk in enumerate(chunks):
            progress_bar.progress((i + 1) / len(chunks))
            st.text(f"Processing page {chunk['page']}...")
            
            embedding = get_embedding_via_api(chunk['text'])
            if embedding:
                chunk['embedding'] = embedding
                embedded_chunks.append(chunk)
            else:
                failed_pages.append(chunk['page'])
            
            # Small delay to avoid rate limits
            if i % 5 == 0 and i > 0:
                import time
                time.sleep(0.5)
        
        # Save to MongoDB
        if save_to_mongodb(uploaded_file.name, embedded_chunks, file_hash, full_text):
            st.success(f"✅ Processed {uploaded_file.name} - {len(embedded_chunks)} pages saved to database")
            
            if failed_pages:
                st.warning(f"⚠️ Failed to process pages: {failed_pages}")
        else:
            st.error(f"Failed to save {uploaded_file.name} to database")
        
        # Clean up
        os.unlink(tmp_path)

def search_documents(query: str, top_k: int = 10, similarity_threshold: float = 0.7) -> List[Dict]:
    """Search across all documents in MongoDB"""
    if db is None:
        st.error("MongoDB connection not available")
        return []
    
    # Get query embedding
    query_embedding = get_embedding_via_api(query)
    if not query_embedding:
        return []
    
    all_results = []
    
    try:
        # Get all chunks with embeddings
        total_chunks = db.chunks.count_documents({})
        
        with st.spinner(f"Searching {total_chunks} pages..."):
            # Fetch all chunks (in production, consider pagination for large datasets)
            chunks = db.chunks.find({'embedding': {'$exists': True, '$ne': []}})
            
            relevant_count = 0
            for chunk in chunks:
                if 'embedding' in chunk and chunk['embedding']:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(query_embedding, chunk['embedding'])
                    
                    # Only include results above threshold
                    if similarity >= similarity_threshold:
                        relevant_count += 1
                        result = {
                            'text': chunk['text'],
                            'page': chunk['page'],
                            'year': chunk['year'],
                            'filename': chunk['filename'],
                            'similarity': similarity
                        }
                        all_results.append(result)
            
            st.info(f"Found {relevant_count} relevant results from {total_chunks} total pages")
    
    except Exception as e:
        st.error(f"Search error: {e}")
        return []
    
    # Sort by similarity and return top results
    all_results.sort(key=lambda x: x['similarity'], reverse=True)
    return all_results[:top_k]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

def generate_response(query: str, results: List[Dict]) -> str:
    """Generate response using OpenAI with better context"""
    # Use more results for better context
    context_results = results[:8] if len(results) >= 8 else results
    
    context = "\n\n".join([
        f"Year {r['year']}, Page {r['page']} (Relevance: {r['similarity']:.2f}):\n{r['text']}"
        for r in context_results
    ])
    
    messages = [
        {"role": "system", "content": """You are a helpful assistant analyzing Maharashtra Economic Survey data. 
        Provide comprehensive answers based on ALL the provided context. 
        Always cite specific years and page numbers when referencing data.
        If multiple years show trends, highlight them."""},
        {"role": "user", "content": f"""Based on the following information from Maharashtra Economic Survey documents,
provide a detailed answer to this query: "{query}"

Context from relevant pages:
{context}

Instructions:
1. Answer the question comprehensively using ALL relevant information from the context
2. Cite specific years and page numbers
3. If data spans multiple years, show the trend
4. Be specific with numbers and facts
5. If the context doesn't contain enough information to fully answer the question, say so

Answer:"""}
    ]
    
    return chat_completion_via_api(messages)

def generate_comparison_table(query: str, results: List[Dict]) -> pd.DataFrame:
    """Generate a comparison table for multi-year data"""
    # Group results by year
    years_data = {}
    for result in results:
        year = result['year']
        if year not in years_data:
            years_data[year] = []
        years_data[year].append(result['text'][:200])
    
    if len(years_data) <= 1:
        return None
    
    # Ask OpenAI to extract relevant metrics
    context = "\n".join([f"Year {year}: {' '.join(texts[:2])}" for year, texts in years_data.items()])
    
    messages = [
        {"role": "system", "content": "You are a data extraction assistant."},
        {"role": "user", "content": f"""Extract the relevant metric for "{query}" from each year's data.
Return as a simple table with columns: Year, Value, Notes

Data:
{context}

Format as CSV with headers."""}
    ]
    
    response = chat_completion_via_api(messages)
    
    try:
        # Parse CSV response
        import io
        df = pd.read_csv(io.StringIO(response))
        return df
    except:
        return None

def main():
    st.title("📊 Maharashtra Economic Survey Search (MongoDB)")
    
    # Check configurations
    if not OPENAI_API_KEY:
        st.error("Please set OPENAI_API_KEY in .env file")
        st.stop()
    
    if db is None:
        st.error("MongoDB connection failed. Please check your MONGODB_URI in .env file")
        st.stop()
    
    # Create temp directory
    os.makedirs("temp", exist_ok=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload PDFs")
        st.info("Name files with year (e.g., Survey_2023.pdf)")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if st.button("Process Files", type="primary"):
            if uploaded_files:
                for file in uploaded_files:
                    process_pdf(file)
            else:
                st.warning("Please upload files first")
        
        st.divider()
        
        # Show processed documents
        st.subheader("📚 Processed Documents")
        documents = get_all_documents()
        if documents:
            for doc in sorted(documents, key=lambda x: x.get('year', 0)):
                st.write(f"• {doc['filename']} ({doc.get('year', 'N/A')}) - {doc.get('total_pages', 0)} pages")
        else:
            st.info("No documents processed yet")
        
        # Test connections
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔧 Test OpenAI"):
                with st.spinner("Testing..."):
                    test_response = chat_completion_via_api([
                        {"role": "user", "content": "Say 'OK' in one word"}
                    ])
                    if "Error" not in test_response:
                        st.success("✅ OpenAI OK")
                    else:
                        st.error(test_response)
        
        with col2:
            if st.button("🔧 Test MongoDB"):
                try:
                    db.command('ping')
                    st.success("✅ MongoDB OK")
                except Exception as e:
                    st.error(f"MongoDB error: {e}")
    
    # Main area
    st.header("🔍 Search Documents")
    
    # Show stats
    col1, col2, col3 = st.columns(3)
    with col1:
        doc_count = db.documents.count_documents({}) if db is not None else 0
        st.metric("Documents", doc_count)
    with col2:
        chunk_count = db.chunks.count_documents({}) if db is not None else 0
        st.metric("Total Pages", chunk_count)
    with col3:
        # Get unique years
        years = db.documents.distinct('year') if db is not None else []
        st.metric("Years Covered", len(years))
    
    # Search interface
    query = st.text_input("Enter your question:", placeholder="e.g., What was the GDP growth rate?")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        search_button = st.button("Search", type="primary", use_container_width=True)
    with col2:
        top_k = st.number_input("Max Results", min_value=5, max_value=20, value=10)
    with col3:
        threshold = st.slider("Relevance", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    with col4:
        output_type = st.selectbox("Output", ["Text", "Table (if multi-year)"])
    
    if search_button and query:
        with st.spinner("Searching..."):
            results = search_documents(query, top_k, threshold)
            
            if results:
                # Check if we should show table
                years = set(r['year'] for r in results)
                
                if output_type == "Table (if multi-year)" and len(years) > 1:
                    st.subheader("📊 Comparison Table")
                    df = generate_comparison_table(query, results)
                    if df is not None:
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("Could not generate table, showing text results instead")
                        output_type = "Text"
                
                if output_type == "Text":
                    # Generate AI response
                    st.subheader("🤖 AI Answer")
                    response = generate_response(query, results)
                    st.write(response)
                
                # Show source pages
                st.subheader("📄 Source Pages")
                for i, result in enumerate(results):
                    with st.expander(f"Year {result['year']} - Page {result['page']} (Relevance: {result['similarity']:.1%})"):
                        st.write(f"**File:** {result['filename']}")
                        st.write(f"**Full Page Text:**")
                        st.text(result['text'])
            else:
                st.warning("No results found. Please upload and process PDF files first.")
    elif search_button:
        st.warning("Please enter a search query")
    
    # Footer
    st.divider()
    st.caption("💡 Tip: Upload PDFs with year in filename (e.g., Survey_2023.pdf) for better organization")

if __name__ == "__main__":
    main()
