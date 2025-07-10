import streamlit as st

# Configure page
st.set_page_config(
    page_title="DNA Codon Tool - Import Test",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Import Testing")

# Test imports one by one
try:
    import pandas as pd
    st.success("✅ pandas imported")
except Exception as e:
    st.error(f"❌ pandas failed: {e}")

try:
    import numpy as np
    st.success("✅ numpy imported")
except Exception as e:
    st.error(f"❌ numpy failed: {e}")

try:
    import matplotlib.pyplot as plt
    st.success("✅ matplotlib imported")
except Exception as e:
    st.error(f"❌ matplotlib failed: {e}")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    st.success("✅ plotly imported")
except Exception as e:
    st.error(f"❌ plotly failed: {e}")

try:
    import os
    import json
    import logging
    st.success("✅ standard libraries imported")
except Exception as e:
    st.error(f"❌ standard libraries failed: {e}")

try:
    from collections import defaultdict, Counter
    st.success("✅ collections imported")
except Exception as e:
    st.error(f"❌ collections failed: {e}")

try:
    from Bio.Seq import Seq
    st.success("✅ biopython imported")
except Exception as e:
    st.error(f"❌ biopython failed: {e}")

try:
    from openpyxl import load_workbook
    st.success("✅ openpyxl imported")
except Exception as e:
    st.error(f"❌ openpyxl failed: {e}")

try:
    import requests
    st.success("✅ requests imported")
except Exception as e:
    st.error(f"❌ requests failed: {e}")

try:
    from bs4 import BeautifulSoup
    st.success("✅ beautifulsoup4 imported")
except Exception as e:
    st.error(f"❌ beautifulsoup4 failed: {e}")

try:
    from dotenv import load_dotenv
    st.success("✅ python-dotenv imported")
except Exception as e:
    st.error(f"❌ python-dotenv failed: {e}")

try:
    from anthropic import Anthropic
    st.success("✅ anthropic imported")
except Exception as e:
    st.error(f"❌ anthropic failed: {e}")

st.info("Import testing complete! Check which ones failed above.")
