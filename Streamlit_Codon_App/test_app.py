import streamlit as st
import pandas as pd
import numpy as np

# Configure page
st.set_page_config(
    page_title="DNA Codon Tool - Test",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 DNA Codon Optimization Tool")
st.write("Testing basic functionality...")

# Test 1: Basic Streamlit
st.success("✅ Streamlit is working!")

# Test 2: Check for Excel file
import os
if os.path.exists("HumanCodons.xlsx"):
    st.success("✅ HumanCodons.xlsx found!")
else:
    st.warning("⚠️ HumanCodons.xlsx not found - but that's OK for now")

# Test 3: Simple sequence input
st.subheader("Simple Sequence Test")
test_sequence = st.text_area("Enter a DNA sequence:", "ATGAAATAA")

if st.button("Test Sequence"):
    st.write(f"You entered: {test_sequence}")
    st.write(f"Length: {len(test_sequence)} bases")

st.info("If you can see this message, the basic app is working!")
