import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
import logging
from collections import defaultdict, Counter
from Bio.Seq import Seq
from openpyxl import load_workbook
import io
import tempfile
import requests
import time
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse, quote
from dotenv import load_dotenv
from typing import List, Dict, Set
from anthropic import Anthropic
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="DNA Codon Optimization and Analysis Tool",
    page_icon=":dna:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧬 Session State Testing")

# Test session state initialization step by step
try:
    if 'config' not in st.session_state:
        st.session_state.config = {
            "codon_file_path": "HumanCodons.xlsx",
            "bias_weight": 1,
            "auto_open_files": True,
            "default_output_dir": "."
        }
    st.success("✅ config initialized")
except Exception as e:
    st.error(f"❌ config failed: {e}")

try:
    if 'active_theme' not in st.session_state:
        st.session_state.active_theme = "Default"
    st.success("✅ active_theme initialized")
except Exception as e:
    st.error(f"❌ active_theme failed: {e}")

try:
    if 'accumulated_results' not in st.session_state:
        st.session_state.accumulated_results = []
    st.success("✅ accumulated_results initialized")
except Exception as e:
    st.error(f"❌ accumulated_results failed: {e}")

try:
    if 'genetic_code' not in st.session_state:
        st.session_state.genetic_code = {}
    if 'codon_weights' not in st.session_state:
        st.session_state.codon_weights = {}
    if 'preferred_codons' not in st.session_state:
        st.session_state.preferred_codons = {}
    st.success("✅ codon dictionaries initialized")
except Exception as e:
    st.error(f"❌ codon dictionaries failed: {e}")

# Test creating the API engines WITHOUT initializing them in session state yet
try:
    # Just test if we can create the classes
    st.write("Testing API engine creation...")
    
    # Test creating without storing in session state
    from anthropic import Anthropic
    test_anthropic = Anthropic(api_key="test") if "test" else None
    st.success("✅ Can create Anthropic class")
    
    st.success("✅ API engines can be created")
except Exception as e:
    st.error(f"❌ API engine creation failed: {e}")

st.info("Session state testing complete!")

# Show what's in session state
st.subheader("Current Session State:")
st.write(dict(st.session_state))
