import streamlit as st
import os
import requests
from dotenv import load_dotenv
from anthropic import Anthropic
import time

# Configure page
st.set_page_config(
    page_title="API Engine Testing",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 API Engine Testing")

# Load environment variables
load_dotenv()

st.subheader("Testing PatentSearchEngine")
try:
    class PatentSearchEngine:
        def __init__(self):
            self.serper_api_key = os.getenv('SERPER_API_KEY')
            self.anthropic_api_key = os.getenv('ANTHROPIC_API')
            self.anthropic = Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None
    
    patent_engine = PatentSearchEngine()
    st.success("✅ PatentSearchEngine created successfully")
    st.write(f"SERPER API Key: {'Found' if patent_engine.serper_api_key else 'Not found'}")
    st.write(f"Anthropic API Key: {'Found' if patent_engine.anthropic_api_key else 'Not found'}")
except Exception as e:
    st.error(f"❌ PatentSearchEngine failed: {e}")

st.subheader("Testing NCBISearchEngine")
try:
    class NCBISearchEngine:
        def __init__(self):
            self.serper_api_key = os.getenv('SERPER_API_KEY')
            self.base_url = "https://www.ncbi.nlm.nih.gov"
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            self.anthropic_api_key = os.getenv('ANTHROPIC_API')
            self.anthropic = Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None
    
    ncbi_engine = NCBISearchEngine()
    st.success("✅ NCBISearchEngine created successfully")
except Exception as e:
    st.error(f"❌ NCBISearchEngine failed: {e}")

st.subheader("Testing UniProtSearchEngine")
try:
    class UniProtSearchEngine:
        def __init__(self):
            self.base_url = "https://www.uniprot.org"
            self.api_url = "https://rest.uniprot.org"
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            self.anthropic_api_key = os.getenv('ANTHROPIC_API')
            self.anthropic = Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None
    
    uniprot_engine = UniProtSearchEngine()
    st.success("✅ UniProtSearchEngine created successfully")
except Exception as e:
    st.error(f"❌ UniProtSearchEngine failed: {e}")

st.subheader("Testing putting engines in session state")
try:
    if 'test_patent_engine' not in st.session_state:
        st.session_state.test_patent_engine = PatentSearchEngine()
    if 'test_ncbi_engine' not in st.session_state:
        st.session_state.test_ncbi_engine = NCBISearchEngine()
    if 'test_uniprot_engine' not in st.session_state:
        st.session_state.test_uniprot_engine = UniProtSearchEngine()
    
    st.success("✅ All engines stored in session state successfully")
except Exception as e:
    st.error(f"❌ Storing engines in session state failed: {e}")

st.info("API Engine testing complete!")

# Test a simple network request
st.subheader("Testing Network Connection")
try:
    response = requests.get("https://httpbin.org/get", timeout=5)
    if response.status_code == 200:
        st.success("✅ Network connection working")
    else:
        st.warning(f"⚠️ Network response: {response.status_code}")
except Exception as e:
    st.error(f"❌ Network test failed: {e}")
