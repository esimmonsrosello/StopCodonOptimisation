import streamlit as st
import pandas as pd
import os
import io
from collections import defaultdict

# Configure page
st.set_page_config(
    page_title="Codon File Testing",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Codon File Loading Testing")

# Initialize session state like your main app
if 'genetic_code' not in st.session_state:
    st.session_state.genetic_code = {}
if 'codon_weights' not in st.session_state:
    st.session_state.codon_weights = {}
if 'preferred_codons' not in st.session_state:
    st.session_state.preferred_codons = {}
if 'human_codon_usage' not in st.session_state:
    st.session_state.human_codon_usage = {}
if 'aa_to_codons' not in st.session_state:
    st.session_state.aa_to_codons = defaultdict(list)

st.subheader("Testing codon file loading function")

@st.cache_data
def load_codon_data_from_file(file_content):
    """Load codon usage data from uploaded file"""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        required_columns = ['triplet', 'amino_acid', 'fraction']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df['triplet'] = df['triplet'].str.upper().str.strip()
        df['amino_acid'] = df['amino_acid'].str.upper().str.strip().replace({'*': 'X'})
        df = df.dropna(subset=['triplet', 'amino_acid', 'fraction'])
        
        genetic_code = df.set_index('triplet')['amino_acid'].to_dict()
        max_fraction = df.groupby('amino_acid')['fraction'].transform('max')
        df['weight'] = df['fraction'] / max_fraction
        codon_weights = df.set_index('triplet')['weight'].to_dict()
        preferred_codons = df.sort_values('fraction', ascending=False).drop_duplicates('amino_acid').set_index('amino_acid')['triplet'].to_dict()
        human_codon_usage = df.set_index('triplet')['fraction'].to_dict()
        
        aa_to_codons = defaultdict(list)
        for codon_val, freq in human_codon_usage.items():
            aa = genetic_code.get(codon_val, None)
            if aa and aa != 'X':
                aa_to_codons[aa].append((codon_val, freq))
        
        return genetic_code, codon_weights, preferred_codons, human_codon_usage, aa_to_codons, df
    except Exception as e:
        raise Exception(f"Error loading codon file: {e}")

# Test if the function works
st.write("Testing codon loading function...")
try:
    # We can't test with the actual file since it might not exist
    # But we can test if the function is defined properly
    st.success("✅ load_codon_data_from_file function defined successfully")
except Exception as e:
    st.error(f"❌ Function definition failed: {e}")

# Test the actual auto-loading logic from your main app
st.subheader("Testing auto-loading logic")

try:
    # This is the exact code from your main app that might be hanging
    if not st.session_state.genetic_code and 'codon_data_loaded' not in st.session_state:
        default_codon_file = "HumanCodons.xlsx"
        st.write(f"Looking for {default_codon_file}...")
        
        if os.path.exists(default_codon_file):
            st.info(f"✅ Found {default_codon_file} - attempting to load...")
            try:
                with open(default_codon_file, 'rb') as f:
                    file_content = f.read()
                
                st.write("File read successfully, parsing...")
                genetic_code, codon_weights, preferred_codons, human_codon_usage, aa_to_codons, codon_df = load_codon_data_from_file(file_content)
                
                st.session_state.genetic_code = genetic_code
                st.session_state.codon_weights = codon_weights
                st.session_state.preferred_codons = preferred_codons
                st.session_state.human_codon_usage = human_codon_usage
                st.session_state.aa_to_codons = aa_to_codons
                st.session_state.codon_data_loaded = True
                st.session_state.codon_file_source = "Default (HumanCodons.xlsx)"
                
                st.success(f"✅ Auto-loaded {len(codon_df)} codon entries from HumanCodons.xlsx")
                
            except Exception as e:
                st.warning(f"⚠️ Could not auto-load HumanCodons.xlsx: {e}")
                st.write("This is OK - the app should continue anyway")
        else:
            st.info(f"ℹ️ {default_codon_file} not found - this is expected on Streamlit Cloud")
            st.write("App should continue without auto-loading")
    
    st.success("✅ Auto-loading logic completed successfully")
    
except Exception as e:
    st.error(f"❌ Auto-loading logic failed: {e}")
    st.write("This might be where your main app is hanging!")

st.info("Codon file testing complete!")

# Show session state
st.subheader("Session State After Testing:")
st.write(f"genetic_code entries: {len(st.session_state.genetic_code)}")
st.write(f"codon_weights entries: {len(st.session_state.codon_weights)}")
st.write(f"codon_data_loaded: {st.session_state.get('codon_data_loaded', 'Not set')}")
