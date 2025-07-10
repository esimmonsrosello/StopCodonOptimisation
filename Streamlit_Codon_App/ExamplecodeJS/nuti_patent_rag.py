import os
import requests
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dotenv import load_dotenv
from anthropic import Anthropic
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import time
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

# Try to import ollama, make it optional
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Load environment variables
load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    
    def extract_text_from_txt(self, file) -> str:
        """Extract text from TXT file"""
        try:
            return str(file.read(), "utf-8")
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            return ""
    
    def process_document(self, uploaded_file) -> Tuple[str, List[str]]:
        """Process uploaded document and return full text and chunks"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            full_text = self.extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            full_text = self.extract_text_from_docx(uploaded_file)
        elif file_extension == 'txt':
            full_text = self.extract_text_from_txt(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return "", []
        
        if not full_text.strip():
            st.error("No text could be extracted from the document")
            return "", []
        
        chunks = self.text_splitter.split_text(full_text)
        return full_text, chunks

class VectorDatabase:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.metadata = []
        self.db_path = "nuti_vector_db"
        self.current_databases = {}
        
        # Ensure directory exists
        Path(self.db_path).mkdir(exist_ok=True)
        
        # Load existing databases on initialization
        self.load_existing_databases()
    
    def load_existing_databases(self):
        """Load all existing databases into memory for session persistence"""
        try:
            databases = self.list_databases()
            st.session_state.available_databases = databases
        except Exception as e:
            st.session_state.available_databases = []
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        with st.spinner("Creating embeddings..."):
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        return embeddings
    
    def build_index(self, chunks: List[str], document_name: str, custom_db_name: str = None) -> str:
        """Build FAISS index from text chunks and return database ID"""
        if not chunks:
            st.error("No chunks to index")
            return None
        
        embeddings = self.create_embeddings(chunks)
        
        # Create new index
        index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        # Generate database ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_db_name:
            safe_db_name = "".join(c for c in custom_db_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_db_name = safe_db_name.replace(' ', '_')
            db_id = f"db_{safe_db_name}_{timestamp}"
            display_name = custom_db_name
        else:
            safe_doc_name = "".join(c for c in document_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_doc_name = safe_doc_name.replace(' ', '_')
            db_id = f"db_{safe_doc_name}_{timestamp}"
            display_name = document_name
        
        # Save database
        db_path = f"{self.db_path}/{db_id}"
        faiss.write_index(index, f"{db_path}.index")
        
        metadata = [{'document': document_name, 'chunk_id': i} for i in range(len(chunks))]
        
        with open(f"{db_path}.pkl", 'wb') as f:
            pickle.dump({
                'chunks': chunks,
                'metadata': metadata,
                'documents': [document_name],  # Track multiple documents
                'document_name': display_name,  # Display name for the database
                'timestamp': timestamp,
                'db_id': db_id,
                'chunk_count': len(chunks),
                'created_date': timestamp[:8]
            }, f)
        
        # Update session state
        self.load_existing_databases()
        
        st.success(f"Successfully indexed {len(chunks)} chunks from {document_name}")
        return db_id
    
    def append_to_database(self, db_id: str, chunks: List[str], document_name: str) -> bool:
        """Append new chunks to an existing database"""
        try:
            db_path = f"{self.db_path}/{db_id}"
            
            # Load existing data
            with open(f"{db_path}.pkl", 'rb') as f:
                existing_data = pickle.load(f)
            
            # Load existing index
            existing_index = faiss.read_index(f"{db_path}.index")
            
            # Create embeddings for new chunks
            new_embeddings = self.create_embeddings(chunks)
            faiss.normalize_L2(new_embeddings)
            
            # Add new embeddings to existing index
            existing_index.add(new_embeddings.astype('float32'))
            
            # Update metadata
            existing_chunks = existing_data['chunks']
            existing_metadata = existing_data['metadata']
            existing_documents = existing_data.get('documents', [existing_data.get('document_name', '')])
            
            # Add new chunks and metadata
            new_metadata = [{'document': document_name, 'chunk_id': len(existing_chunks) + i} 
                           for i in range(len(chunks))]
            
            updated_chunks = existing_chunks + chunks
            updated_metadata = existing_metadata + new_metadata
            updated_documents = list(set(existing_documents + [document_name]))  # Avoid duplicates
            
            # Save updated database
            faiss.write_index(existing_index, f"{db_path}.index")
            
            with open(f"{db_path}.pkl", 'wb') as f:
                pickle.dump({
                    'chunks': updated_chunks,
                    'metadata': updated_metadata,
                    'documents': updated_documents,
                    'document_name': existing_data['document_name'],
                    'timestamp': existing_data['timestamp'],
                    'db_id': db_id,
                    'chunk_count': len(updated_chunks),
                    'created_date': existing_data.get('created_date', existing_data['timestamp'][:8]),
                    'last_updated': datetime.now().strftime("%Y%m%d_%H%M%S")
                }, f)
            
            # Update session state
            self.load_existing_databases()
            
            st.success(f"Successfully added {len(chunks)} chunks from {document_name} to existing database")
            return True
            
        except Exception as e:
            st.error(f"Error appending to database: {e}")
            return False
    
    def rename_database(self, db_id: str, new_name: str) -> bool:
        """Rename a database"""
        try:
            db_path = f"{self.db_path}/{db_id}"
            
            # Load existing data
            with open(f"{db_path}.pkl", 'rb') as f:
                data = pickle.load(f)
            
            # Update the display name
            data['document_name'] = new_name
            
            # Save updated data
            with open(f"{db_path}.pkl", 'wb') as f:
                pickle.dump(data, f)
            
            # Update session state
            self.load_existing_databases()
            
            st.success(f"Database renamed to '{new_name}'")
            return True
            
        except Exception as e:
            st.error(f"Error renaming database: {e}")
            return False
    
    def load_database(self, db_id: str) -> bool:
        """Load a specific database"""
        try:
            db_path = f"{self.db_path}/{db_id}"
            
            # Load FAISS index
            self.index = faiss.read_index(f"{db_path}.index")
            
            # Load metadata and chunks
            with open(f"{db_path}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
                st.session_state.current_document_name = data.get('document_name', 'Unknown Document')
                st.session_state.current_db_id = db_id
                st.session_state.current_db_documents = data.get('documents', [data.get('document_name', 'Unknown')])
            
            return True
        except Exception as e:
            st.error(f"Error loading database: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks in current database"""
        if not self.index:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(score),
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def list_databases(self) -> List[Dict]:
        """List all available databases with metadata"""
        db_files = []
        for file in Path(self.db_path).glob("*.pkl"):
            try:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    db_files.append({
                        'db_id': data.get('db_id', str(file.stem)),
                        'document_name': data.get('document_name', 'Unknown'),
                        'documents': data.get('documents', [data.get('document_name', 'Unknown')]),
                        'timestamp': data.get('timestamp', ''),
                        'chunk_count': len(data.get('chunks', [])),
                        'created_date': data.get('created_date', data.get('timestamp', '')[:8] if data.get('timestamp') else 'Unknown'),
                        'last_updated': data.get('last_updated', data.get('timestamp', 'Unknown'))
                    })
            except Exception as e:
                continue
        return sorted(db_files, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_database(self, db_id: str) -> bool:
        """Delete a database"""
        try:
            db_path = f"{self.db_path}/{db_id}"
            if os.path.exists(f"{db_path}.index"):
                os.remove(f"{db_path}.index")
            if os.path.exists(f"{db_path}.pkl"):
                os.remove(f"{db_path}.pkl")
            
            # Update session state
            self.load_existing_databases()
            return True
        except Exception as e:
            st.error(f"Error deleting database: {e}")
            return False
    
    def get_database_info(self, db_id: str) -> Dict:
        """Get detailed information about a database"""
        try:
            db_path = f"{self.db_path}/{db_id}"
            with open(f"{db_path}.pkl", 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            return {}

class PatentSearchEngine:
    def __init__(self):
        self.serper_api_key = os.getenv('SERPER_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API')
        self.anthropic = Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None
        
        # Initialize model settings
        if 'ai_model_type' not in st.session_state:
            st.session_state.ai_model_type = 'anthropic'  # Default to anthropic
        if 'ollama_model' not in st.session_state:
            st.session_state.ollama_model = 'llama2'  # Default ollama model
        
    def get_available_ollama_models(self) -> List[str]:
        """Get list of available Ollama models"""
        if not OLLAMA_AVAILABLE:
            return []
        
        try:
            models = ollama.list()
            return [model.model for model in models.models]
        except Exception as e:
            st.error(f"Error fetching Ollama models: {e}")
            return []
    
    def test_ollama_connection(self) -> bool:
        """Test if Ollama is running and accessible"""
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            ollama.list()
            return True
        except Exception:
            return False
    
    def search_patents(self, query: str, num_results: int = 10, excluded_urls: Set[str] = None) -> List[Dict]:
        """Search Google Patents using SERPER API with optional URL exclusions"""
        if not self.serper_api_key:
            st.error("SERPER API key is not configured")
            return []
        
        # For excluded patents, we need to search for more results initially
        # to compensate for the ones we'll filter out
        search_multiplier = 2 if excluded_urls else 1
        search_num = min(num_results * search_multiplier, 100)  # API limit
            
        url = "https://google.serper.dev/search"
        patent_query = f"site:patents.google.com {query}"
        
        payload = {"q": patent_query, "num": search_num}
        headers = {'X-API-KEY': self.serper_api_key, 'Content-Type': 'application/json'}
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 403:
                st.error("API Key is invalid or doesn't have permission")
                return []
            elif response.status_code != 200:
                st.error(f"API Error: {response.status_code}")
                return []
            
            results = response.json().get('organic', [])
            
            # Filter out excluded URLs if provided
            if excluded_urls:
                filtered_results = []
                for result in results:
                    if result.get('link', '') not in excluded_urls:
                        filtered_results.append(result)
                        if len(filtered_results) >= num_results:
                            break
                return filtered_results
            
            return results[:num_results]
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error searching patents: {e}")
            return []
    
    def extract_patent_info(self, search_results: List[Dict]) -> List[Dict]:
        """Extract relevant patent information from search results"""
        patents = []
        for result in search_results:
            patent_info = {
                'title': result.get('title', ''),
                'link': result.get('link', ''),
                'snippet': result.get('snippet', ''),
                'patent_id': self.extract_patent_id(result.get('link', ''))
            }
            patents.append(patent_info)
        return patents
    
    def extract_patent_id(self, url: str) -> str:
        """Extract patent ID from Google Patents URL"""
        try:
            if 'patents.google.com/patent/' in url:
                return url.split('/patent/')[1].split('/')[0].split('?')[0]
            return ""
        except:
            return ""
    
    def generate_ai_response_anthropic(self, query: str, context: str, response_type: str = "patent") -> str:
        """Generate AI response using Anthropic"""
        if not self.anthropic:
            return "Anthropic API is not configured."
        
        if response_type == "patent":
            prompt = f"""
You are a patent research assistant. A user has asked: "{query}"

Based on the following patent search results, provide a comprehensive response:

{context}

Please:
1. Provide a clear answer to the user's question
2. Reference specific patents with IDs and titles
3. Explain key technologies and innovations
4. Include relevant patent links
5. Compare different approaches if applicable
"""
        else:  # document RAG
            prompt = f"""
You are a document analysis assistant. A user has asked: "{query}"

Based on the following document excerpts, provide a comprehensive answer:

{context}

Please:
1. Provide a clear and detailed answer based on the document content
2. Quote relevant sections when appropriate
3. If the answer isn't fully contained in the provided excerpts, mention this
4. Structure your response clearly
5. Be specific and reference the relevant parts of the document
"""

        try:
            message = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error generating AI response: {e}"
    
    def generate_ai_response_ollama(self, query: str, context: str, response_type: str = "patent") -> str:
        """Generate AI response using Ollama"""
        if not OLLAMA_AVAILABLE:
            return "Ollama is not available. Please install ollama-python package."
        
        if response_type == "patent":
            prompt = f"""
You are a patent research assistant. A user has asked: "{query}"

Based on the following patent search results, provide a comprehensive response:

{context}

Please:
1. Provide a clear answer to the user's question
2. Reference specific patents with IDs and titles
3. Explain key technologies and innovations
4. Include relevant patent links
5. Compare different approaches if applicable

Response:
"""
        else:  # document RAG
            prompt = f"""
You are a document analysis assistant. A user has asked: "{query}"

Based on the following document excerpts, provide a comprehensive answer:

{context}

Please:
1. Provide a clear and detailed answer based on the document content
2. Quote relevant sections when appropriate
3. If the answer isn't fully contained in the provided excerpts, mention this
4. Structure your response clearly
5. Be specific and reference the relevant parts of the document

Response:
"""

        try:
            response = ollama.generate(
                model=st.session_state.ollama_model,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'num_predict': 2000,
                }
            )
            return response['response']
        except Exception as e:
            return f"Error generating AI response with Ollama: {e}"
    
    def generate_ai_response(self, query: str, context: str, response_type: str = "patent") -> str:
        """Generate AI response using selected model"""
        if st.session_state.ai_model_type == 'ollama':
            return self.generate_ai_response_ollama(query, context, response_type)
        else:
            return self.generate_ai_response_anthropic(query, context, response_type)

class PatentScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.delay = 2
    
    def scrape_patent_details(self, patent_url: str) -> Dict:
        """Scrape detailed patent information from Google Patents"""
        try:
            time.sleep(self.delay)
            
            response = self.session.get(patent_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            patent_details = {
                'url': patent_url,
                'patent_id': self.extract_patent_id(patent_url),
                'title': self.extract_title(soup),
                'abstract': self.extract_abstract(soup),
                'description': self.extract_description(soup),
                'claims': self.extract_claims(soup),
                'inventors': self.extract_inventors(soup),
                'assignee': self.extract_assignee(soup),
                'filing_date': self.extract_filing_date(soup),
                'publication_date': self.extract_publication_date(soup),
                'classifications': self.extract_classifications(soup),
                'full_text': '',
                'scrape_success': True
            }
            
            patent_details['full_text'] = f"""
Title: {patent_details['title']}

Abstract: {patent_details['abstract']}

Description: {patent_details['description']}

Claims: {patent_details['claims']}

Inventors: {', '.join(patent_details['inventors'])}

Assignee: {patent_details['assignee']}
"""
            
            return patent_details
            
        except Exception as e:
            return {
                'url': patent_url,
                'patent_id': self.extract_patent_id(patent_url),
                'error': str(e),
                'scrape_success': False
            }
    
    def extract_patent_id(self, url: str) -> str:
        try:
            if 'patents.google.com/patent/' in url:
                return url.split('/patent/')[1].split('/')[0].split('?')[0]
            return ""
        except:
            return ""
    
    def extract_title(self, soup: BeautifulSoup) -> str:
        try:
            title_elem = soup.find('span', {'itemprop': 'title'}) or soup.find('h1')
            return title_elem.get_text(strip=True) if title_elem else ""
        except:
            return ""
    
    def extract_abstract(self, soup: BeautifulSoup) -> str:
        try:
            abstract_elem = soup.find('section', {'itemprop': 'abstract'})
            if abstract_elem:
                return abstract_elem.get_text(strip=True)
            
            abstract_elem = soup.find('div', class_='abstract')
            return abstract_elem.get_text(strip=True) if abstract_elem else ""
        except:
            return ""
    
    def extract_description(self, soup: BeautifulSoup) -> str:
        try:
            description_elem = soup.find('section', {'itemprop': 'description'})
            if description_elem:
                for script in description_elem(["script", "style"]):
                    script.decompose()
                return description_elem.get_text(strip=True)
            
            description_elem = soup.find('div', class_='description')
            if description_elem:
                for script in description_elem(["script", "style"]):
                    script.decompose()
                return description_elem.get_text(strip=True)
            
            return ""
        except:
            return ""
    
    def extract_claims(self, soup: BeautifulSoup) -> str:
        try:
            claims_elem = soup.find('section', {'itemprop': 'claims'})
            if claims_elem:
                claims_text = ""
                claim_divs = claims_elem.find_all('div', class_='claim')
                for i, claim in enumerate(claim_divs, 1):
                    claim_text = claim.get_text(strip=True)
                    claims_text += f"Claim {i}: {claim_text}\n\n"
                return claims_text
            return ""
        except:
            return ""
    
    def extract_inventors(self, soup: BeautifulSoup) -> List[str]:
        try:
            inventors = []
            inventor_elems = soup.find_all('dd', {'itemprop': 'inventor'})
            for elem in inventor_elems:
                inventors.append(elem.get_text(strip=True))
            return inventors
        except:
            return []
    
    def extract_assignee(self, soup: BeautifulSoup) -> str:
        try:
            assignee_elem = soup.find('dd', {'itemprop': 'assignee'})
            return assignee_elem.get_text(strip=True) if assignee_elem else ""
        except:
            return ""
    
    def extract_filing_date(self, soup: BeautifulSoup) -> str:
        try:
            filing_elem = soup.find('time', {'itemprop': 'filingDate'})
            return filing_elem.get('datetime', '') if filing_elem else ""
        except:
            return ""
    
    def extract_publication_date(self, soup: BeautifulSoup) -> str:
        try:
            pub_elem = soup.find('time', {'itemprop': 'publicationDate'})
            return pub_elem.get('datetime', '') if pub_elem else ""
        except:
            return ""
    
    def extract_classifications(self, soup: BeautifulSoup) -> List[str]:
        try:
            classifications = []
            class_elems = soup.find_all('span', {'itemprop': 'cpcClassifications'})
            for elem in class_elems:
                classifications.append(elem.get_text(strip=True))
            return classifications
        except:
            return []

class DeepPatentAnalyzer:
    def __init__(self, patent_engine, patent_scraper):
        self.patent_engine = patent_engine
        self.scraper = patent_scraper
    
    def analyze_patents_deeply(self, query: str, num_initial_results: int = 20, num_deep_analysis: int = 5, excluded_urls: Set[str] = None) -> Dict:
        """Perform deep analysis of patents with optional exclusions"""
        
        excluded_urls = excluded_urls or set()
        
        st.info(f"Step 1: Searching for {num_initial_results} initial patents...")
        if excluded_urls:
            st.info(f"Excluding {len(excluded_urls)} previously analyzed patents")
        
        initial_results = self.patent_engine.search_patents(query, num_initial_results, excluded_urls)
        
        if not initial_results:
            return {"error": "No initial patents found (or all results were previously analyzed)"}
        
        st.info(f"Step 2: Analyzing {len(initial_results)} patents to select top candidates...")
        patents = self.patent_engine.extract_patent_info(initial_results)
        
        top_patents = patents[:num_deep_analysis]
        
        st.info(f"Step 3: Scraping detailed information from {len(top_patents)} patents...")
        detailed_patents = []
        
        progress_bar = st.progress(0)
        for i, patent in enumerate(top_patents, 1):
            st.text(f"Scraping patent {i}/{len(top_patents)}: {patent['patent_id']}")
            
            detailed_patent = self.scraper.scrape_patent_details(patent['link'])
            detailed_patent.update({
                'initial_snippet': patent['snippet'],
                'search_rank': i
            })
            detailed_patents.append(detailed_patent)
            
            progress_bar.progress(i / len(top_patents))
        
        st.info("Step 4: Performing deep AI analysis...")
        analysis_result = self.generate_deep_analysis(query, detailed_patents)
        
        # Track analyzed URLs for future exclusion
        analyzed_urls = {patent['url'] for patent in detailed_patents if patent.get('url')}
        
        return {
            'query': query,
            'initial_results_count': len(initial_results),
            'deep_analysis_count': len(detailed_patents),
            'detailed_patents': detailed_patents,
            'ai_analysis': analysis_result,
            'analyzed_urls': analyzed_urls,
            'success': True
        }
    
    def generate_deep_analysis(self, query: str, detailed_patents: List[Dict]) -> str:
        """Generate comprehensive AI analysis of detailed patent information"""
        
        context_parts = []
        
        for i, patent in enumerate(detailed_patents, 1):
            if patent.get('scrape_success', False):
                context_part = f"""
PATENT {i} - {patent.get('patent_id', 'Unknown ID')}
========================================
Title: {patent.get('title', 'N/A')}
Filing Date: {patent.get('filing_date', 'N/A')}
Inventors: {', '.join(patent.get('inventors', []))}
Assignee: {patent.get('assignee', 'N/A')}

Abstract:
{patent.get('abstract', 'N/A')[:1000]}...

Description (Key Sections):
{patent.get('description', 'N/A')[:2000]}...

Claims (First Few):
{patent.get('claims', 'N/A')[:1500]}...

URL: {patent.get('url', 'N/A')}
"""
            else:
                context_part = f"""
PATENT {i} - {patent.get('patent_id', 'Unknown ID')}
========================================
Title: {patent.get('title', 'N/A')}
Error: {patent.get('error', 'Unknown error during scraping')}
URL: {patent.get('url', 'N/A')}
"""
            
            context_parts.append(context_part)
        
        full_context = "\n\n".join(context_parts)
        
        prompt = f"""
You are an expert patent analyst. A user has asked: "{query}"

I have performed a deep analysis by scraping the full content of the most relevant patents. Below is the detailed information from {len(detailed_patents)} patents:

{full_context}

Please provide a comprehensive analysis that includes:

1. **Direct Answer**: Provide a specific, detailed answer to the user's question based on the full patent content
2. **Specific Evidence**: Quote specific sections from the patents that support your answer
3. **Patent-by-Patent Analysis**: Analyze each patent individually in relation to the query
4. **Technical Details**: Include relevant technical information, sequences, methods, or data found in the patents
5. **Comparative Analysis**: Compare and contrast the approaches across different patents
6. **Limitations**: Note any limitations in the available information
7. **Recommendations**: Suggest which patents are most relevant and why

Be thorough and specific. If you find the exact information the user is looking for (like specific sequences, methods, or technical details), quote them directly from the patent content.
"""

        # Use the selected AI model for analysis
        if st.session_state.ai_model_type == 'ollama':
            try:
                response = ollama.generate(
                    model=st.session_state.ollama_model,
                    prompt=prompt,
                    options={
                        'temperature': 0.2,
                        'top_p': 0.9,
                        'num_predict': 4000,
                    }
                )
                return response['response']
            except Exception as e:
                return f"Error generating deep analysis with Ollama: {e}"
        else:
            try:
                message = self.patent_engine.anthropic.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4000,
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text
            except Exception as e:
                return f"Error generating deep analysis: {e}"
    
    def search_patent_content(self, detailed_patents: List[Dict], search_terms: List[str]) -> Dict:
        """Search for specific terms within patent content"""
        results = {}
        
        for patent in detailed_patents:
            if not patent.get('scrape_success', False):
                continue
                
            patent_id = patent.get('patent_id', 'Unknown')
            full_text = patent.get('full_text', '').lower()
            
            patent_matches = {}
            for term in search_terms:
                term_lower = term.lower()
                matches = []
                
                for match in re.finditer(re.escape(term_lower), full_text):
                    start = max(0, match.start() - 100)
                    end = min(len(full_text), match.end() + 100)
                    context = full_text[start:end]
                    matches.append({
                        'context': context,
                        'position': match.start()
                    })
                
                if matches:
                    patent_matches[term] = matches
            
            if patent_matches:
                results[patent_id] = {
                    'title': patent.get('title', ''),
                    'url': patent.get('url', ''),
                    'matches': patent_matches
                }
        
        return results

def initialize_session_state():
    """Initialize session state variables"""
    if 'patent_engine' not in st.session_state:
        st.session_state.patent_engine = PatentSearchEngine()
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = VectorDatabase()
    if 'patent_scraper' not in st.session_state:
        st.session_state.patent_scraper = PatentScraper()
    if 'deep_analyzer' not in st.session_state:
        st.session_state.deep_analyzer = DeepPatentAnalyzer(
            st.session_state.patent_engine, 
            st.session_state.patent_scraper
        )
    if 'available_databases' not in st.session_state:
        st.session_state.available_databases = []
    if 'analyzed_patent_urls' not in st.session_state:
        st.session_state.analyzed_patent_urls = set()
    if 'last_deep_search_query' not in st.session_state:
        st.session_state.last_deep_search_query = ""
    if 'deep_search_results' not in st.session_state:
        st.session_state.deep_search_results = None

def display_deep_analysis_results(result, include_search_terms, search_terms):
    """Helper function to display deep analysis results"""
    
    st.markdown("**Comprehensive AI Analysis**")
    st.markdown(result['ai_analysis'])
    
    if include_search_terms and search_terms.strip():
        st.markdown("**Specific Term Search Results**")
        terms_list = [term.strip() for term in search_terms.split(',') if term.strip()]
        
        search_results = st.session_state.deep_analyzer.search_patent_content(
            result['detailed_patents'], terms_list
        )
        
        if search_results:
            for patent_id, patent_data in search_results.items():
                with st.expander(f"Matches in Patent {patent_id}: {patent_data['title'][:50]}..."):
                    st.write(f"**Patent Title:** {patent_data['title']}")
                    st.write(f"**URL:** [View Patent]({patent_data['url']})")
                    
                    for term, matches in patent_data['matches'].items():
                        st.write(f"**Found '{term}' - {len(matches)} occurrences:**")
                        for i, match in enumerate(matches[:3], 1):
                            st.write(f"*Match {i}:*")
                            st.code(match['context'])
                        if len(matches) > 3:
                            st.info(f"... and {len(matches) - 3} more occurrences")
        else:
            st.warning("No specific terms found in the analyzed patents")
    
    # Display detailed patent information
    st.markdown("**Detailed Patent Information**")
    
    successful_scrapes = [p for p in result['detailed_patents'] if p.get('scrape_success')]
    failed_scrapes = [p for p in result['detailed_patents'] if not p.get('scrape_success')]
    
    if successful_scrapes:
        st.success(f"Successfully scraped {len(successful_scrapes)} patents")
        
        for i, patent in enumerate(successful_scrapes, 1):
            with st.expander(f"Patent {i}: {patent.get('title', 'Unknown Title')[:80]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Patent ID:** {patent.get('patent_id', 'N/A')}")
                    st.write(f"**Filing Date:** {patent.get('filing_date', 'N/A')}")
                    st.write(f"**Inventors:** {', '.join(patent.get('inventors', []))}")
                    st.write(f"**Assignee:** {patent.get('assignee', 'N/A')}")
                
                with col2:
                    st.write(f"**Search Rank:** #{patent.get('search_rank', 'N/A')}")
                    st.write(f"**Publication Date:** {patent.get('publication_date', 'N/A')}")
                    st.write(f"**URL:** [View Patent]({patent.get('url', '#')})")
                
                st.write("**Abstract:**")
                st.write(patent.get('abstract', 'N/A')[:500] + "...")
                
                if st.checkbox(f"Show full description - Patent {i}", key=f"desc_{i}_{id(result)}"):
                    st.write("**Full Description:**")
                    st.text_area("", patent.get('description', 'N/A')[:2000] + "...", 
                               height=200, key=f"desc_text_{i}_{id(result)}")
                
                if st.checkbox(f"Show claims - Patent {i}", key=f"claims_{i}_{id(result)}"):
                    st.write("**Claims:**")
                    st.text_area("", patent.get('claims', 'N/A')[:1500] + "...", 
                               height=150, key=f"claims_text_{i}_{id(result)}")
    
    if failed_scrapes:
        st.warning(f"Failed to scrape {len(failed_scrapes)} patents")
        with st.expander("View failed scrapes"):
            for patent in failed_scrapes:
                st.write(f"**Patent ID:** {patent.get('patent_id', 'N/A')}")
                st.write(f"**Error:** {patent.get('error', 'Unknown error')}")
                st.write(f"**URL:** {patent.get('url', 'N/A')}")
                st.write("---")
    
    # Summary statistics
    st.markdown("**Analysis Summary**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Initial Results", result['initial_results_count'])
    with col2:
        st.metric("Deep Analysis", result['deep_analysis_count'])
    with col3:
        st.metric("Successful Scrapes", len(successful_scrapes))
    with col4:
        st.metric("Total Excluded", len(st.session_state.analyzed_patent_urls))

def main():
    st.set_page_config(
        page_title="NUTI Patent Search and RAG",
        page_icon="🔬",
        layout="wide"
    )
    
    # Custom CSS for professional appearance
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .status-success {
        color: #10b981;
        font-weight: 500;
    }
    .status-error {
        color: #ef4444;
        font-weight: 500;
    }
    .status-warning {
        color: #f59e0b;
        font-weight: 500;
    }
    .search-again-container {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0ea5e9;
        margin: 1rem 0;
    }
    .model-indicator {
        background-color: #fef3c7;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #f59e0b;
        margin-bottom: 1rem;
    }
    
    /* Hide sidebar on page load */
    .css-1d391kg {
        width: 0rem;
    }
    .css-1d391kg .css-1544g2n {
        width: 0rem;
    }
    section[data-testid="stSidebar"] {
        width: 0rem !important;
        min-width: 0rem !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 0rem !important;
        min-width: 0rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-header">🔬 NUTI Patent Search and RAG</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Neural Unified Technology Interface</p>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Model indicator
    if st.session_state.ai_model_type == 'ollama':
        model_status = f"🤖 Using Ollama Model: {st.session_state.ollama_model}"
        if not OLLAMA_AVAILABLE or not st.session_state.patent_engine.test_ollama_connection():
            model_status += " ⚠️ (Connection Issue)"
    else:
        model_status = f"🧠 Using Anthropic Claude"
        if not st.session_state.patent_engine.anthropic:
            model_status += " ⚠️ (Not Configured)"
    
    st.markdown(f'<div class="model-indicator">{model_status}</div>', unsafe_allow_html=True)
    
    # Create tabs
    tabs = st.tabs([
        "Patent Search", 
        "Deep Patent Analysis", 
        "Document Upload", 
        "Ask Questions", 
        "Database Management",
        "Settings"
    ])
    
    # Tab 1: Patent Search
    with tabs[0]:
        st.header("Patent Search")
        
        query = st.text_input(
            "Enter your patent-related question:",
            placeholder="e.g., What are the latest patents on solar panel efficiency?",
            key="patent_query"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            num_results = st.slider("Number of patents", 5, 20, 10)
        with col2:
            search_button = st.button("Search Patents", type="primary")
        
        if search_button and query:
            with st.spinner("Searching patents..."):
                search_results = st.session_state.patent_engine.search_patents(query, num_results)
                
                if search_results:
                    patents = st.session_state.patent_engine.extract_patent_info(search_results)
                    
                    patent_context = "\n".join([
                        f"Patent: {p['title']}\nID: {p['patent_id']}\nSummary: {p['snippet']}\nLink: {p['link']}\n"
                        for p in patents
                    ])
                    
                    ai_response = st.session_state.patent_engine.generate_ai_response(
                        query, patent_context, "patent"
                    )
                    
                    st.subheader("AI Analysis")
                    st.markdown(ai_response)
                    
                    st.subheader("Patent Details")
                    for i, patent in enumerate(patents, 1):
                        with st.expander(f"Patent {i}: {patent['title'][:100]}..."):
                            st.write(f"**Patent ID:** {patent['patent_id']}")
                            st.write(f"**Title:** {patent['title']}")
                            st.write(f"**Summary:** {patent['snippet']}")
                            st.write(f"**Link:** [View Patent]({patent['link']})")
    
    # Tab 2: Deep Patent Analysis
    with tabs[1]:
        st.header("Deep Patent Analysis")
        st.info("Comprehensive analysis with full patent content scraping")
        
        deep_query = st.text_area(
            "Enter your detailed patent research question:",
            placeholder="""Examples:
- Find patents containing specific DNA/RNA sequences like 'UUA codon'
- Analyze patents for specific technical methods or processes
- Search for patents with particular chemical compounds or formulations
- Find patents by specific inventors or companies with detailed analysis""",
            height=100,
            key="deep_query"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_results = st.slider("Initial search results", 10, 50, 20)
        
        with col2:
            deep_analysis_count = st.slider("Deep analysis count", 3, 10, 5)
        
        with col3:
            include_search_terms = st.checkbox("Specific term search", value=False)
        
        if include_search_terms:
            search_terms = st.text_input(
                "Enter specific terms to search for (comma-separated):",
                placeholder="e.g., UUA, codon, mRNA, spike protein"
            )
        else:
            search_terms = ""
        
        # Show current exclusion status
        if st.session_state.analyzed_patent_urls:
            st.info(f"Currently excluding {len(st.session_state.analyzed_patent_urls)} previously analyzed patents from searches")
            if st.button("Clear Patent Exclusions"):
                st.session_state.analyzed_patent_urls = set()
                st.session_state.last_deep_search_query = ""
                st.session_state.deep_search_results = None
                st.success("Patent exclusions cleared")
                st.rerun()
        
        # Main search button
        col_search, col_reset = st.columns(2)
        
        with col_search:
            start_analysis_button = st.button("Start Deep Analysis", type="primary", use_container_width=True)
        
        with col_reset:
            if st.session_state.deep_search_results:
                new_search_button = st.button("New Search (Reset Exclusions)", use_container_width=True)
                if new_search_button:
                    st.session_state.analyzed_patent_urls = set()
                    st.session_state.last_deep_search_query = ""
                    st.session_state.deep_search_results = None
                    st.success("Reset complete. Ready for new search.")
                    st.rerun()
        
        if start_analysis_button and deep_query.strip():
            start_time = time.time()
            
            # Check if this is the same query and offer to exclude previous results
            is_repeat_search = (deep_query == st.session_state.last_deep_search_query and 
                              st.session_state.analyzed_patent_urls)
            
            result = st.session_state.deep_analyzer.analyze_patents_deeply(
                deep_query, 
                initial_results, 
                deep_analysis_count, 
                st.session_state.analyzed_patent_urls if is_repeat_search else None
            )
            
            end_time = time.time()
            
            if result.get('success'):
                # Update session state
                st.session_state.last_deep_search_query = deep_query
                st.session_state.deep_search_results = result
                st.session_state.analyzed_patent_urls.update(result.get('analyzed_urls', set()))
                
                st.success(f"Deep analysis completed in {end_time - start_time:.1f} seconds")
                
                # Show "Search Again" option
                if result.get('analyzed_urls'):
                    st.markdown("""
                    <div class="search-again-container">
                        <h4>Search Again Option</h4>
                        <p>If these results don't contain what you're looking for, you can run the same search again 
                        to find different patents. The next search will automatically exclude the patents analyzed above.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("Search Again (Exclude Previous Results)", type="secondary", use_container_width=True):
                        start_time_2 = time.time()
                        
                        result_2 = st.session_state.deep_analyzer.analyze_patents_deeply(
                            deep_query, 
                            initial_results, 
                            deep_analysis_count, 
                            st.session_state.analyzed_patent_urls
                        )
                        
                        end_time_2 = time.time()
                        
                        if result_2.get('success'):
                            st.session_state.analyzed_patent_urls.update(result_2.get('analyzed_urls', set()))
                            st.success(f"Follow-up analysis completed in {end_time_2 - start_time_2:.1f} seconds")
                            
                            # Display second search results
                            st.subheader("Follow-up Analysis Results")
                            display_deep_analysis_results(result_2, include_search_terms, search_terms)
                        else:
                            st.warning(f"Follow-up analysis: {result_2.get('error', 'Unknown error')}")
                
                # Display main results
                st.subheader("Initial Analysis Results")
                display_deep_analysis_results(result, include_search_terms, search_terms)
                
            else:
                st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        
        elif start_analysis_button and not deep_query.strip():
            st.error("Please enter a research question")

    # Tab 3: Document Upload
    with tabs[2]:
        st.header("Document Upload & Processing")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Document")
            uploaded_file = st.file_uploader(
                "Choose a document to analyze",
                type=['pdf', 'docx', 'txt'],
                help="Upload PDF, DOCX, or TXT files for analysis"
            )
            
            if uploaded_file:
                st.success(f"File uploaded: {uploaded_file.name}")
                
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size / 1024:.1f} KB",
                    "File type": uploaded_file.type
                }
                
                for key, value in file_details.items():
                    st.text(f"{key}: {value}")
                
                st.write("---")
                
                # Database options
                st.subheader("Database Options")
                
                databases = st.session_state.vector_db.list_databases()
                
                db_option = st.radio(
                    "Choose database action:",
                    ["Create new database", "Add to existing database"],
                    help="Create a new database or add this document to an existing one"
                )
                
                if db_option == "Create new database":
                    st.write("**New Database Settings**")
                    
                    use_custom_name = st.checkbox("Use custom database name", value=False)
                    
                    if use_custom_name:
                        custom_db_name = st.text_input(
                            "Database name:",
                            placeholder="e.g., Building Regulations, Floor Management Requirements",
                            help="Enter a descriptive name for your database"
                        )
                    else:
                        custom_db_name = None
                        st.info(f"Database will be named after the document: {uploaded_file.name}")
                    
                    process_button_text = "Create New Database & Process Document"
                    
                else:  # Add to existing database
                    if not databases:
                        st.warning("No existing databases found. Please create a new database first.")
                        st.stop()
                    
                    st.write("**Select Existing Database**")
                    
                    # Show existing databases with details
                    db_options = []
                    for db in databases:
                        doc_list = ", ".join(db['documents'][:3])  # Show first 3 documents
                        if len(db['documents']) > 3:
                            doc_list += f" and {len(db['documents']) - 3} more"
                        
                        db_options.append(f"{db['document_name']} ({len(db['documents'])} docs: {doc_list})")
                    
                    selected_db_idx = st.selectbox(
                        "Select database to add document to:",
                        range(len(databases)),
                        format_func=lambda x: db_options[x]
                    )
                    
                    selected_db = databases[selected_db_idx]
                    
                    st.info(f"Will add document to: **{selected_db['document_name']}**")
                    st.write(f"Current documents in database: {', '.join(selected_db['documents'])}")
                    st.write(f"Current chunk count: {selected_db['chunk_count']}")
                    
                    process_button_text = f"Add to '{selected_db['document_name']}' Database"
                    custom_db_name = None
                
                # Process button
                if st.button(process_button_text, type="primary", use_container_width=True):
                    with st.spinner("Processing document and creating embeddings..."):
                        full_text, chunks = st.session_state.doc_processor.process_document(uploaded_file)
                        
                        if chunks:
                            if db_option == "Create new database":
                                # Create new database
                                db_id = st.session_state.vector_db.build_index(
                                    chunks, uploaded_file.name, custom_db_name
                                )
                                if db_id:
                                    st.session_state.current_document = uploaded_file.name
                                    st.session_state.document_processed = True
                                    st.session_state.current_db_id = db_id
                                    
                                    display_name = custom_db_name if custom_db_name else uploaded_file.name
                                    st.success(f"New database '{display_name}' created successfully")
                                    st.info(f"Created {len(chunks)} searchable chunks")
                            
                            else:
                                # Add to existing database
                                success = st.session_state.vector_db.append_to_database(
                                    selected_db['db_id'], chunks, uploaded_file.name
                                )
                                
                                if success:
                                    # Load the updated database
                                    st.session_state.vector_db.load_database(selected_db['db_id'])
                                    st.session_state.document_processed = True
                                    st.session_state.current_db_id = selected_db['db_id']
                                    
                                    st.success(f"Document added to '{selected_db['document_name']}' successfully")
                                    st.info(f"Added {len(chunks)} new chunks to the database")
                            
                            st.info("You can now ask questions about your documents in the 'Ask Questions' tab")
        
        with col2:
            st.subheader("Processing Information")
            st.write("""
            **Supported formats:**
            - PDF files
            - Word documents (.docx)
            - Text files (.txt)
            
            **Database Options:**
            
            **Create New Database:**
            - Creates a fresh database
            - Option for custom naming
            - Ideal for new topics/categories
            
            **Add to Existing:**
            - Appends to existing database
            - Combines multiple documents
            - Perfect for related documents
            - Example: Adding more building regulations to existing "Building Regs" database
            
            **After processing:**
            - Ask questions about all documents in the database
            - Get AI-powered answers across all content
            - View relevant excerpts from any document
            """)
            
            if hasattr(st.session_state, 'document_processed') and st.session_state.document_processed:
                st.success("Database ready for questions")
                current_doc = getattr(st.session_state, 'current_document_name', 'Current database')
                st.info(f"Current: {current_doc}")
                
                # Show documents in current database
                if hasattr(st.session_state, 'current_db_documents'):
                    st.write("**Documents in current database:**")
                    for doc in st.session_state.current_db_documents:
                        st.write(f"• {doc}")

    # Tab 4: Ask Questions
    with tabs[3]:
        st.header("Ask Questions About Your Documents")
        
        if not (hasattr(st.session_state, 'document_processed') and st.session_state.document_processed):
            st.warning("No database is currently loaded")
            st.write("""
            **To get started:**
            1. Go to the 'Document Upload' tab
            2. Upload your document(s) (PDF, DOCX, or TXT)
            3. Choose to create a new database or add to existing
            4. Click 'Process & Index Document'
            5. Return here to ask questions
            
            Or load a previously processed database from the 'Database Management' tab.
            """)
        else:
            current_doc = getattr(st.session_state, 'current_document_name', 'Unknown Database')
            current_documents = getattr(st.session_state, 'current_db_documents', [])
            
            st.success(f"Current Database: {current_doc}")
            
            # Show which documents are being searched
            if current_documents and len(current_documents) > 1:
                with st.expander(f"Documents in this database ({len(current_documents)})", expanded=False):
                    for i, doc in enumerate(current_documents, 1):
                        st.write(f"{i}. {doc}")
            elif current_documents:
                st.info(f"Searching in: {current_documents[0]}")
            
            st.subheader("What would you like to know?")
            doc_query = st.text_area(
                "Enter your question about the documents:",
                placeholder="""Examples:
- What are the main findings or conclusions?
- Can you summarize the key points from all documents?
- What methodology was used in this research?
- Are there any specific recommendations mentioned?
- What are the limitations discussed?
- Can you explain the technical details about...?
- Compare the information between the different documents""",
                height=150,
                key="doc_query"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                ask_button = st.button("Ask Question", type="primary", use_container_width=True)
            
            with col2:
                num_chunks = st.selectbox("Chunks to analyze", [3, 5, 7, 10], index=1)
            
            with col3:
                show_excerpts = st.checkbox("Show excerpts", value=True)
            
            if ask_button and doc_query.strip():
                with st.spinner("Searching documents and generating response..."):
                    results = st.session_state.vector_db.search(doc_query, k=num_chunks)
                    
                    if results:
                        context = "\n\n".join([
                            f"Document Excerpt {i+1} (from {r['metadata']['document']}, Relevance Score: {r['score']:.3f}):\n{r['chunk']}"
                            for i, r in enumerate(results)
                        ])
                        
                        ai_response = st.session_state.patent_engine.generate_ai_response(
                            doc_query, context, "document"
                        )
                        
                        st.subheader("AI Response")
                        st.markdown(ai_response)
                        
                        if show_excerpts:
                            st.subheader("Relevant Document Excerpts")
                            st.write("These are the most relevant sections found in your documents:")
                            
                            for i, result in enumerate(results, 1):
                                relevance_indicator = "High" if result['score'] > 0.7 else "Medium" if result['score'] > 0.5 else "Low"
                                source_doc = result['metadata']['document']
                                
                                with st.expander(f"Excerpt {i} - {source_doc} - Relevance: {relevance_indicator} ({result['score']:.3f})"):
                                    st.markdown(result['chunk'])
                                    st.caption(f"Source: {source_doc} | Chunk ID: {result['metadata']['chunk_id']}")
                    else:
                        st.warning("No relevant information found in the documents for your question")
                        st.info("Try rephrasing your question or using different keywords")
            
            elif ask_button and not doc_query.strip():
                st.error("Please enter a question before clicking 'Ask Question'")
            
            # Enhanced example questions for multi-document databases
            with st.expander("Example Questions"):
                if len(current_documents) > 1:
                    st.write(f"""
                    **Multi-Document Analysis (searching across {len(current_documents)} documents):**
                    - "Compare the approaches mentioned in the different documents"
                    - "What are the common themes across all documents?"
                    - "Are there any contradictions between the documents?"
                    - "Summarize the key points from each document"
                    
                    **General Analysis:**
                    - "What is the main topic covered in these documents?"
                    - "Can you provide a comprehensive summary of all the content?"
                    - "What are the key findings or conclusions from all documents?"
                    
                    **Specific Information:**
                    - "What methodology was used?" (will search across all documents)
                    - "Are there any recommendations provided?"
                    - "What are the limitations mentioned in any of the documents?"
                    - "Can you explain the technical details about [specific topic]?"
                    """)
                else:
                    st.write("""
                    **General Analysis:**
                    - "What is this document about?"
                    - "Can you provide a summary of the main points?"
                    - "What are the key findings or conclusions?"
                    
                    **Specific Information:**
                    - "What methodology was used?"
                    - "Are there any recommendations provided?"
                    - "What are the limitations mentioned?"
                    - "Can you explain the technical details about [specific topic]?"
                    
                    **Comparative Questions:**
                    - "How does this compare to existing solutions?"
                    - "What are the advantages and disadvantages discussed?"
                    - "What alternatives are mentioned?"
                    """)

    # Tab 5: Database Management
    with tabs[4]:
        st.header("Database Management")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Available Document Databases")
            databases = st.session_state.vector_db.list_databases()
            
            if databases:
                # Create enhanced dataframe with document details
                db_data = []
                for db in databases:
                    doc_list = ", ".join(db['documents'][:2])  # Show first 2 documents
                    if len(db['documents']) > 2:
                        doc_list += f" +{len(db['documents']) - 2} more"
                    
                    db_data.append({
                        "Database Name": db['document_name'],
                        "Documents": f"{len(db['documents'])} files",
                        "Document List": doc_list,
                        "Created": db['created_date'],
                        "Chunks": db['chunk_count'],
                        "Database ID": db['db_id']
                    })
                
                df = pd.DataFrame(db_data)
                st.dataframe(df, use_container_width=True)
                
                st.write("---")
                
                # Database selection and actions
                selected_idx = st.selectbox(
                    "Select a database:",
                    range(len(databases)),
                    format_func=lambda x: f"{databases[x]['document_name']} ({len(databases[x]['documents'])} documents, {databases[x]['chunk_count']} chunks)"
                )
                
                selected_db = databases[selected_idx]
                
                # Show detailed information about selected database
                with st.expander("Database Details", expanded=True):
                    st.write(f"**Database Name:** {selected_db['document_name']}")
                    st.write(f"**Database ID:** {selected_db['db_id']}")
                    st.write(f"**Created:** {selected_db['created_date']}")
                    st.write(f"**Last Updated:** {selected_db.get('last_updated', 'N/A')}")
                    st.write(f"**Total Chunks:** {selected_db['chunk_count']}")
                    st.write(f"**Documents ({len(selected_db['documents'])}):**")
                    for i, doc in enumerate(selected_db['documents'], 1):
                        st.write(f"   {i}. {doc}")
                
                # Action buttons
                col_load, col_rename, col_delete = st.columns(3)
                
                with col_load:
                    if st.button("Load Database", type="primary", use_container_width=True):
                        if st.session_state.vector_db.load_database(selected_db['db_id']):
                            st.success(f"Loaded: {selected_db['document_name']}")
                            st.session_state.document_processed = True
                            st.rerun()
                        else:
                            st.error("Failed to load database")
                
                with col_rename:
                    if st.button("Rename Database", use_container_width=True):
                        st.session_state.show_rename_dialog = selected_db['db_id']
                        st.rerun()
                
                with col_delete:
                    if st.button("Delete Database", use_container_width=True):
                        st.session_state.show_delete_dialog = selected_db['db_id']
                        st.rerun()
                
                # Rename dialog
                if hasattr(st.session_state, 'show_rename_dialog') and st.session_state.show_rename_dialog == selected_db['db_id']:
                    st.write("---")
                    st.subheader("Rename Database")
                    
                    new_name = st.text_input(
                        "New database name:",
                        value=selected_db['document_name'],
                        placeholder="Enter new name for the database"
                    )
                    
                    col_confirm, col_cancel = st.columns(2)
                    
                    with col_confirm:
                        if st.button("Confirm Rename", type="primary", use_container_width=True):
                            if new_name.strip() and new_name != selected_db['document_name']:
                                if st.session_state.vector_db.rename_database(selected_db['db_id'], new_name.strip()):
                                    st.session_state.show_rename_dialog = None
                                    st.rerun()
                            else:
                                st.error("Please enter a different name")
                    
                    with col_cancel:
                        if st.button("Cancel", use_container_width=True):
                            st.session_state.show_rename_dialog = None
                            st.rerun()
                
                # Delete confirmation dialog
                if hasattr(st.session_state, 'show_delete_dialog') and st.session_state.show_delete_dialog == selected_db['db_id']:
                    st.write("---")
                    st.subheader("⚠️ Delete Database")
                    st.error(f"Are you sure you want to delete '{selected_db['document_name']}'?")
                    st.write("This action cannot be undone. All documents and chunks will be permanently removed.")
                    
                    confirm_text = st.text_input(
                        f"Type '{selected_db['document_name']}' to confirm deletion:",
                        placeholder="Type database name here"
                    )
                    
                    col_confirm, col_cancel = st.columns(2)
                    
                    with col_confirm:
                        if st.button("DELETE DATABASE", type="primary", use_container_width=True):
                            if confirm_text == selected_db['document_name']:
                                if st.session_state.vector_db.delete_database(selected_db['db_id']):
                                    st.session_state.show_delete_dialog = None
                                    st.success("Database deleted successfully")
                                    st.rerun()
                            else:
                                st.error("Database name doesn't match. Please type the exact name.")
                    
                    with col_cancel:
                        if st.button("Cancel Delete", use_container_width=True):
                            st.session_state.show_delete_dialog = None
                            st.rerun()
                            
            else:
                st.info("No databases found. Upload and process a document first.")
        
        with col2:
            st.subheader("Current Database Status")
            
            if hasattr(st.session_state, 'document_processed') and st.session_state.document_processed:
                current_doc = getattr(st.session_state, 'current_document_name', 'Unknown Database')
                current_documents = getattr(st.session_state, 'current_db_documents', [])
                
                st.success("Database Loaded")
                st.info(f"Database: {current_doc}")
                
                if st.session_state.vector_db.chunks:
                    st.metric("Total Chunks", len(st.session_state.vector_db.chunks))
                    st.metric("Documents", len(current_documents))
                    st.metric("Embedding Dimension", st.session_state.vector_db.dimension)
                    
                    # Show documents in current database
                    if current_documents:
                        st.write("**Documents in database:**")
                        for i, doc in enumerate(current_documents, 1):
                            st.write(f"{i}. {doc}")
                    
                    if st.checkbox("Show sample chunk"):
                        sample_chunk = st.session_state.vector_db.chunks[0]
                        st.text_area("Sample text chunk:", sample_chunk[:500] + "...", height=100, disabled=True)
                    
                    st.write("---")
                    
                    if st.button("Clear Current Database", use_container_width=True):
                        st.session_state.vector_db.index = None
                        st.session_state.vector_db.chunks = []
                        st.session_state.vector_db.metadata = []
                        st.session_state.document_processed = False
                        if hasattr(st.session_state, 'current_document_name'):
                            delattr(st.session_state, 'current_document_name')
                        if hasattr(st.session_state, 'current_db_documents'):
                            delattr(st.session_state, 'current_db_documents')
                        st.success("Database cleared")
                        st.rerun()
            else:
                st.info("No database currently loaded")
                st.write("""
                **To load a database:**
                1. Select from available databases above, or
                2. Upload a new document in the 'Document Upload' tab
                
                **Database Features:**
                - **Multi-document databases**: Combine related documents
                - **Custom naming**: Give meaningful names to your databases
                - **Easy management**: Rename, load, or delete databases
                - **Document tracking**: See which documents are in each database
                """)

    # Tab 6: Settings
    with tabs[5]:
        st.header("Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("AI Model Selection")
            
            # Model selection
            model_options = ["anthropic"]
            if OLLAMA_AVAILABLE:
                model_options.append("ollama")
            
            current_model_idx = 0 if st.session_state.ai_model_type == "anthropic" else 1
            
            selected_model = st.radio(
                "Choose AI Model:",
                model_options,
                index=current_model_idx,
                format_func=lambda x: "Anthropic Claude" if x == "anthropic" else "Local Ollama"
            )
            
            if selected_model != st.session_state.ai_model_type:
                st.session_state.ai_model_type = selected_model
                st.success(f"Switched to {selected_model.title()} model")
                st.rerun()
            
            # Ollama-specific settings
            if selected_model == "ollama" and OLLAMA_AVAILABLE:
                st.write("**Ollama Settings**")
                
                # Test connection
                if st.button("Test Ollama Connection"):
                    if st.session_state.patent_engine.test_ollama_connection():
                        st.success("✅ Ollama is connected and running")
                    else:
                        st.error("❌ Cannot connect to Ollama. Make sure it's running.")
                
                # Model selection
                available_models = st.session_state.patent_engine.get_available_ollama_models()
                if available_models:
                    current_model_idx = 0
                    if st.session_state.ollama_model in available_models:
                        current_model_idx = available_models.index(st.session_state.ollama_model)
                    
                    selected_ollama_model = st.selectbox(
                        "Select Ollama Model:",
                        available_models,
                        index=current_model_idx
                    )
                    
                    if selected_ollama_model != st.session_state.ollama_model:
                        st.session_state.ollama_model = selected_ollama_model
                        st.success(f"Selected Ollama model: {selected_ollama_model}")
                else:
                    st.warning("No Ollama models found. Please install models using 'ollama pull <model_name>'")
                    st.code("ollama pull llama2\nollama pull mistral\nollama pull codellama")
            
            elif selected_model == "ollama" and not OLLAMA_AVAILABLE:
                st.error("Ollama integration not available. Install with: pip install ollama")
            
            # Anthropic settings
            if selected_model == "anthropic":
                st.write("**Anthropic Settings**")
                anthropic_status = "✅ Connected" if st.session_state.patent_engine.anthropic else "❌ Not configured"
                st.write(f"Status: {anthropic_status}")
                
                if not st.session_state.patent_engine.anthropic:
                    st.info("Set ANTHROPIC_API environment variable to use Anthropic models")
            
            st.write("---")
            st.subheader("Deep Analysis Settings")
            
            new_delay = st.slider(
                "Scraping delay (seconds)",
                min_value=1.0,
                max_value=10.0,
                value=float(getattr(st.session_state.patent_scraper, 'delay', 2.0)),
                step=0.5,
                help="Delay between patent scraping requests"
            )
            
            if st.button("Update Scraping Delay"):
                st.session_state.patent_scraper.delay = new_delay
                st.success(f"Scraping delay updated to {new_delay} seconds")
            
            st.write("---")
            st.subheader("Patent Exclusion Management")
            
            if st.session_state.analyzed_patent_urls:
                st.write(f"Currently excluding {len(st.session_state.analyzed_patent_urls)} patents from deep analysis")
                if st.button("View Excluded Patents"):
                    for i, url in enumerate(st.session_state.analyzed_patent_urls, 1):
                        st.text(f"{i}. {url}")
                
                if st.button("Clear All Exclusions"):
                    st.session_state.analyzed_patent_urls = set()
                    st.session_state.last_deep_search_query = ""
                    st.session_state.deep_search_results = None
                    st.success("All patent exclusions cleared")
                    st.rerun()
            else:
                st.info("No patents currently excluded from searches")
        
        with col2:
            st.subheader("System Information")
            
            st.write("**API Status:**")
            serper_status = "Connected" if st.session_state.patent_engine.serper_api_key else "Not configured"
            anthropic_status = "Connected" if st.session_state.patent_engine.anthropic_api_key else "Not configured"
            ollama_status = "Available" if OLLAMA_AVAILABLE else "Not installed"
            
            st.write(f"- SERPER API: {serper_status}")
            st.write(f"- Anthropic API: {anthropic_status}")
            st.write(f"- Ollama: {ollama_status}")
            
            if OLLAMA_AVAILABLE:
                ollama_connection = "Connected" if st.session_state.patent_engine.test_ollama_connection() else "Disconnected"
                st.write(f"- Ollama Connection: {ollama_connection}")
            
            st.write("**Current AI Model:**")
            if st.session_state.ai_model_type == "ollama":
                st.write(f"- Type: Local Ollama")
                st.write(f"- Model: {st.session_state.ollama_model}")
            else:
                st.write(f"- Type: Anthropic Claude")
                st.write(f"- Model: claude-3-sonnet-20240229")
            
            st.write("**Database Status:**")
            databases = st.session_state.vector_db.list_databases()
            st.write(f"- Stored databases: {len(databases)}")
            
            doc_status = "Loaded" if hasattr(st.session_state, 'document_processed') and st.session_state.document_processed else "None"
            st.write(f"- Current document: {doc_status}")
            
            st.write("**Deep Analysis Status:**")
            st.write(f"- Excluded patents: {len(st.session_state.analyzed_patent_urls)}")
            st.write(f"- Last search query: {st.session_state.last_deep_search_query[:50] + '...' if len(st.session_state.last_deep_search_query) > 50 else st.session_state.last_deep_search_query}")
            
            st.write("**Performance:**")
            st.write(f"- Scraping delay: {getattr(st.session_state.patent_scraper, 'delay', 2.0)}s")
            st.write(f"- Embedding model: all-MiniLM-L6-v2")
            st.write(f"- Vector dimension: {st.session_state.vector_db.dimension}")
            
            st.write("---")
            st.subheader("Test Patent Scraping")
            
            test_url = st.text_input(
                "Test patent URL:",
                placeholder="https://patents.google.com/patent/US..."
            )
            
            if st.button("Test Scrape") and test_url:
                with st.spinner("Testing patent scraping..."):
                    result = st.session_state.patent_scraper.scrape_patent_details(test_url)
                    
                    if result.get('scrape_success'):
                        st.success("Scraping successful")
                        st.json(result)
                    else:
                        st.error("Scraping failed")
                        st.json(result)

    # Sidebar
    with st.sidebar:
        st.write("## NUTI Platform")
        st.write("**Neural Unified Technology Interface**")
        
        st.write("### Features")
        st.write("""
        - Basic Patent Search with AI analysis
        - Deep Patent Analysis with full content scraping
        - Iterative search with patent exclusions
        - Document Processing (PDF, DOCX, TXT)
        - Multi-document databases
        - Custom database naming
        - RAG-powered Q&A system
        - Local vector storage
        - Multiple AI models (Anthropic/Ollama)
        - Advanced settings
        """)
        
        st.write("### System Status")
        serper_indicator = "✓" if st.session_state.patent_engine.serper_api_key else "✗"
        
        if st.session_state.ai_model_type == 'ollama':
            ai_indicator = "✓" if OLLAMA_AVAILABLE and st.session_state.patent_engine.test_ollama_connection() else "✗"
            ai_model = f"Ollama ({st.session_state.ollama_model})"
        else:
            ai_indicator = "✓" if st.session_state.patent_engine.anthropic else "✗"
            ai_model = "Anthropic Claude"
        
        doc_indicator = "✓" if hasattr(st.session_state, 'document_processed') and st.session_state.document_processed else "✗"
        
        st.write(f"**SERPER API:** {serper_indicator}")
        st.write(f"**AI Model:** {ai_indicator} {ai_model}")
        st.write(f"**Document Ready:** {doc_indicator}")
        
        if st.session_state.analyzed_patent_urls:
            st.write(f"**Excluded Patents:** {len(st.session_state.analyzed_patent_urls)}")
        
        st.write("### Analysis Modes")
        st.write("""
        **Basic Search:**
        - Fast results from patent summaries
        - Good for general research
        - Lower resource usage
        
        **Deep Analysis:**
        - Full patent content analysis
        - Specific term searching
        - Iterative search capabilities
        - Comprehensive technical details
        - Higher accuracy for specific queries
        
        **Document Analysis:**
        - Multi-document databases
        - Custom database names
        - Cross-document search
        - Persistent storage
        """)

if __name__ == "__main__":
    main()