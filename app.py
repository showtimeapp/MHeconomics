import streamlit as st
import os
import json
import fitz  # PyMuPDF
import pandas as pd
import requests
from dotenv import load_dotenv
import re
from typing import List, Dict
import tempfile
import pickle

# Load environment variables
load_dotenv()

# Get API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Page config
st.set_page_config(
    page_title="Maharashtra Economic Survey Search",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

def extract_year_from_filename(filename: str) -> int:
    """Extract year from filename"""
    match = re.search(r'(\d{4})', filename)
    return int(match.group(1)) if match else None

def pdf_to_txt(pdf_path: str) -> str:
    """Convert PDF to text file and return text file path"""
    doc = fitz.open(pdf_path)
    txt_path = pdf_path.replace('.pdf', '.txt')
    
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            txt_file.write(f"=== PAGE {page_num + 1} ===\n")
            txt_file.write(text)
            txt_file.write("\n\n")
    
    doc.close()
    return txt_path

def pdf_to_chunks(pdf_path: str) -> List[Dict]:
    """Convert PDF to text chunks (1 page = 1 chunk)"""
    # First convert to text file
    txt_path = pdf_to_txt(pdf_path)
    
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
                'filename': filename,
                'txt_file': os.path.basename(txt_path)
            })
    
    doc.close()
    
    # Save chunk info
    st.info(f"📄 Converted to: {txt_path}")
    st.info(f"📊 Total pages with content: {len(chunks)}")
    
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

def save_processed_data(filename: str, chunks: List[Dict]):
    """Save processed chunks to local file"""
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save as JSON file
    output_path = os.path.join(processed_dir, f"{filename}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'chunks': chunks}, f)
    
    # Update session state
    st.session_state.processed_files.add(filename)
    
    return output_path

def load_processed_data(filename: str) -> Dict:
    """Load processed chunks from local file"""
    filepath = os.path.join("data/processed", f"{filename}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_all_processed_files() -> List[str]:
    """Get list of all processed files"""
    processed_dir = "data/processed"
    if os.path.exists(processed_dir):
        return [f for f in os.listdir(processed_dir) if f.endswith('.json')]
    return []

def process_pdf(uploaded_file):
    """Process a single PDF file"""
    # Check if already processed
    if uploaded_file.name in st.session_state.processed_files:
        st.info(f"✅ {uploaded_file.name} already processed")
        return
    
    # Check if file exists on disk
    existing_data = load_processed_data(uploaded_file.name)
    if existing_data:
        st.session_state.processed_files.add(uploaded_file.name)
        st.info(f"✅ {uploaded_file.name} already processed (loaded from disk)")
        return
    
    with st.spinner(f"Processing {uploaded_file.name}..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        # Extract chunks (1 page = 1 chunk)
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
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = embedding
                embedded_chunks.append(chunk_with_embedding)
            else:
                failed_pages.append(chunk['page'])
            
            # Small delay to avoid rate limits
            if i % 5 == 0 and i > 0:
                import time
                time.sleep(0.5)
        
        # Save locally
        output_path = save_processed_data(uploaded_file.name, embedded_chunks)
        
        if failed_pages:
            st.warning(f"⚠️ Failed to process pages: {failed_pages}")
        
        st.success(f"✅ Processed {uploaded_file.name} - {len(embedded_chunks)} pages embedded successfully")
        
        # Move text file to txt directory
        txt_filename = uploaded_file.name.replace('.pdf', '.txt')
        if os.path.exists(txt_filename):
            import shutil
            shutil.move(txt_filename, os.path.join("data/txt", txt_filename))
        
        # Clean up
        os.unlink(tmp_path)

def search_documents(query: str, top_k: int = 10, similarity_threshold: float = 0.7) -> List[Dict]:
    """Search across all processed documents with improved accuracy"""
    # Get query embedding
    query_embedding = get_embedding_via_api(query)
    if not query_embedding:
        return []
    
    all_results = []
    
    # Load all processed files
    processed_files = get_all_processed_files()
    
    if not processed_files:
        st.warning("No processed documents found")
        return []
    
    with st.spinner(f"Searching {len(processed_files)} documents..."):
        total_chunks = 0
        for filename in processed_files:
            data = load_processed_data(filename.replace('.json', ''))
            
            if data and 'chunks' in data:
                for chunk in data['chunks']:
                    if 'embedding' in chunk:
                        total_chunks += 1
                        # Calculate cosine similarity
                        similarity = cosine_similarity(query_embedding, chunk['embedding'])
                        
                        # Only include results above threshold
                        if similarity >= similarity_threshold:
                            result = {
                                'text': chunk['text'],
                                'page': chunk['page'],
                                'year': chunk['year'],
                                'filename': chunk['filename'],
                                'similarity': similarity
                            }
                            all_results.append(result)
        
        st.info(f"Searched {total_chunks} pages, found {len(all_results)} relevant results")
    
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
5. If you think context is from table the give output in table formate
6. If the context doesn't contain enough information to fully answer the question, say so

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
    st.title("📊 Maharashtra Economic Survey Search (Local Storage)")
    
    # Check if API key is set
    if not OPENAI_API_KEY:
        st.error("Please set OPENAI_API_KEY in .env file")
        st.stop()
    
    # Create directories
    os.makedirs("data/pdfs", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/txt", exist_ok=True)
    
    # Load processed files on startup
    st.session_state.processed_files.update(
        [f.replace('.json', '') for f in get_all_processed_files()]
    )
    
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
        
        # Show processed files
        st.subheader("📚 Processed Files")
        if st.session_state.processed_files:
            for filename in sorted(st.session_state.processed_files):
                year = extract_year_from_filename(filename)
                st.write(f"• {filename} ({year})")
        else:
            st.info("No files processed yet")
        
        # Test OpenAI connection
        st.divider()
        if st.button("🔧 Test OpenAI Connection"):
            with st.spinner("Testing..."):
                test_response = chat_completion_via_api([
                    {"role": "user", "content": "Say 'OK' in one word"}
                ])
                if "Error" not in test_response:
                    st.success("✅ OpenAI Connected")
                else:
                    st.error(test_response)
    
    # Main area
    st.header("🔍 Search Documents")
    
    # Show stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Processed Documents", len(st.session_state.processed_files))
    with col2:
        total_size = sum(
            os.path.getsize(os.path.join("data/processed", f"{f}.json"))
            for f in st.session_state.processed_files
            if os.path.exists(os.path.join("data/processed", f"{f}.json"))
        ) / (1024 * 1024)  # Convert to MB
        st.metric("Storage Used", f"{total_size:.1f} MB")
    
    # Search interface
    query = st.text_input("Enter your question:", placeholder="e.g., What was the GDP growth rate?")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        search_button = st.button("Search", type="primary", use_container_width=True)
    with col2:
        top_k = st.number_input("Max Results", min_value=1, max_value=20, value=10)
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
                
                # Show sources with full page content
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