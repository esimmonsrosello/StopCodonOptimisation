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

# Initialize session state safely
def init_session_state():
    """Initialize session state with error handling"""
    defaults = {
        'config': {
            "codon_file_path": "HumanCodons.xlsx",
            "bias_weight": 1,
            "auto_open_files": True,
            "default_output_dir": "."
        },
        'active_theme': "Default",
        'accumulated_results': [],
        'batch_accumulated_results': [],
        'run_counter': 0,
        'genetic_code': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TAT': 'Y', 'TAC': 'Y', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D',
            'GAA': 'E', 'GAG': 'E', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'TGT': 'C', 'TGC': 'C', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGT': 'S', 'AGC': 'S',
            'AGA': 'R', 'AGG': 'R', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
            'TAA': '*', 'TAG': '*', 'TGA': '*'
        },
        'codon_weights': {
            'TTT': 0.45, 'TTC': 1.0, 'TTA': 0.07, 'TTG': 0.13, 'CTT': 0.13, 'CTC': 0.20, 'CTA': 0.07, 'CTG': 1.0,
            'ATT': 0.36, 'ATC': 1.0, 'ATA': 0.17, 'GTT': 0.18, 'GTC': 0.24, 'GTA': 0.11, 'GTG': 1.0,
            'TAT': 0.43, 'TAC': 1.0, 'CAT': 0.41, 'CAC': 1.0, 'CAA': 0.25, 'CAG': 1.0,
            'AAT': 0.46, 'AAC': 1.0, 'AAA': 0.42, 'AAG': 1.0, 'GAT': 0.46, 'GAC': 1.0,
            'GAA': 0.42, 'GAG': 1.0, 'TCT': 0.18, 'TCC': 0.22, 'TCA': 0.15, 'TCG': 0.05,
            'CCT': 0.28, 'CCC': 0.33, 'CCA': 0.27, 'CCG': 0.11, 'ACT': 0.24, 'ACC': 1.0, 'ACA': 0.28, 'ACG': 0.12,
            'GCT': 0.26, 'GCC': 1.0, 'GCA': 0.23, 'GCG': 0.11, 'TGT': 0.45, 'TGC': 1.0, 'TGG': 1.0,
            'CGT': 0.08, 'CGC': 0.19, 'CGA': 0.11, 'CGG': 0.21, 'AGT': 0.15, 'AGC': 1.0,
            'AGA': 0.20, 'AGG': 0.20, 'GGT': 0.16, 'GGC': 1.0, 'GGA': 0.25, 'GGG': 0.25,
            'TAA': 1.0, 'TAG': 0.20, 'TGA': 0.52
        },
        'preferred_codons': {
            'F': 'TTC', 'L': 'CTG', 'I': 'ATC', 'V': 'GTG', 'Y': 'TAC', 'H': 'CAC', 'Q': 'CAG',
            'N': 'AAC', 'K': 'AAG', 'D': 'GAC', 'E': 'GAG', 'S': 'AGC', 'P': 'CCC', 'T': 'ACC',
            'A': 'GCC', 'C': 'TGC', 'W': 'TGG', 'R': 'CGC', 'G': 'GGC', '*': 'TAA'
        },
        'human_codon_usage': {},
        'aa_to_codons': defaultdict(list)
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Load environment variables
load_dotenv()

# Your utility functions
def validate_dna_sequence(sequence):
    """Validate DNA sequence and return cleaned version"""
    if not sequence:
        return False, "", "No DNA sequence provided"
    cleaned = sequence.upper().replace('\n', '').replace(' ', '').replace('\t', '').replace('U', 'T')
    invalid_bases = set(cleaned) - set('ATGC')
    if invalid_bases:
        return False, "", f"Invalid characters found: {', '.join(invalid_bases)}. Only A, T, G, C allowed."
    return True, cleaned, ""

def translate_dna(seq):
    """Translate DNA sequence to protein"""
    protein = ""
    genetic_code = st.session_state.genetic_code
    for i in range(0, len(seq) - 2, 3):
        codon_val = seq[i:i+3].upper()
        aa = genetic_code.get(codon_val, '?')
        protein += aa
    return protein

def calculate_gc_content(sequence):
    """Calculate GC content percentage of DNA sequence"""
    if not sequence:
        return 0.0
    clean_seq = sequence.upper().replace(' ', '').replace('\n', '')
    valid_bases = [base for base in clean_seq if base in 'ATGC']
    if not valid_bases:
        return 0.0
    gc_count = sum(1 for base in valid_bases if base in 'GC')
    return (gc_count / len(valid_bases)) * 100

def number_of_plus1_stops(dna_seq):
    """Count stop codons in +1 frame across the entire sequence"""
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    stop_codons_set = {"TAA", "TAG", "TGA"}
    
    counts = Counter()
    for i in range(1, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        if codon in stop_codons_set:
            counts[codon] += 1
    
    total_stops = sum(counts.values())
    return {'TAA': counts['TAA'], 'TAG': counts['TAG'], 'TGA': counts['TGA'], 'total': total_stops}

def codon_optimize(protein_seq):
    """Standard codon optimization using most frequent codons"""
    preferred_codons = st.session_state.preferred_codons
    optimized = ''.join(preferred_codons.get(aa, 'NNN') for aa in protein_seq if aa != 'X')
    return optimized

def get_codon_weights_row(dna_seq):
    """Calculate CAI weights for DNA sequence"""
    codon_weights = st.session_state.codon_weights
    codons_list = [dna_seq[i:i+3].upper() for i in range(0, len(dna_seq) - 2, 3)]
    weights = [codon_weights.get(c, 1e-6) for c in codons_list]
    return weights, codons_list

def run_single_optimization(sequence, method, bias_weight=None):
    """Run single sequence optimization"""
    is_valid, clean_seq, error_msg = validate_dna_sequence(sequence)
    if not is_valid:
        return None, error_msg
    
    try:
        protein_seq = translate_dna(clean_seq)
        
        if method == "Standard Codon Optimization":
            optimized = codon_optimize(protein_seq)
            weights, _ = get_codon_weights_row(optimized)
            result = {
                'Original_DNA': clean_seq,
                'Protein': protein_seq,
                'Optimized_DNA': optimized,
                'CAI_Weights': ','.join(f"{w:.4f}" for w in weights),
                'Method': method
            }
        elif method == "In-Frame Analysis":
            weights, codons_list = get_codon_weights_row(clean_seq)
            result = {
                'Position': list(range(1, len(codons_list) + 1)),
                'DNA_Codon': codons_list,
                'CAI_Weight': weights,
                'Amino_Acid': [st.session_state.genetic_code.get(c, '?') for c in codons_list],
                'Method': method
            }
        elif method == "+1 Frame Analysis":
            plus1_stop_counts = number_of_plus1_stops(clean_seq)
            gc_content = calculate_gc_content(clean_seq)
            
            result = {
                'Sequence_Length': len(clean_seq),
                'Protein_Length': len(protein_seq),
                'GC_Content': gc_content,
                'Plus1_TAA_Count': plus1_stop_counts['TAA'],
                'Plus1_TAG_Count': plus1_stop_counts['TAG'],
                'Plus1_TGA_Count': plus1_stop_counts['TGA'],
                'Plus1_Total_Stops': plus1_stop_counts['total'],
                'Method': method
            }
        
        return result, None
    except Exception as e:
        return None, str(e)

def create_download_link(df, filename):
    """Create download link for DataFrame as Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    processed_data = output.getvalue()
    return processed_data

def main():
    """Main Streamlit application"""
    st.write("🚀 Starting application...")
    
    # Initialize session state
    init_session_state()
    st.write("✅ Session state initialized")
    
    st.title("DNA Codon Optimization and Analysis Tool")
    st.markdown("DNA sequence optimization and analysis")
    st.write("✅ Title displayed")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.success("✅ Built-in human codon usage loaded")
        st.info("Codon data is built-in - no file upload needed")
        
        # Settings
        st.subheader("Algorithm Settings")
        bias_weight = st.slider(
            "Bias Weight", 
            min_value=0.1, 
            max_value=5.0, 
            value=1.0,
            step=0.1
        )
    
    st.write("✅ Sidebar created")
    
    # Main tabs
    st.write("📋 Creating tabs...")
    tab1, tab2, tab3 = st.tabs(["Single Sequence", "Batch Optimization", "About"])
    st.write("✅ Tabs created")

    with tab1:
        st.header("Single Sequence Optimization")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sequence_input = st.text_area(
                "DNA Sequence",
                height=150,
                placeholder="Enter DNA sequence (A, T, G, C only)..."
            )
        
        with col2:
            optimization_method = st.selectbox(
                "Optimization Method",
                ["In-Frame Analysis", "+1 Frame Analysis", "Standard Codon Optimization"]
            )
            
            run_optimization_button = st.button("Run Optimization", type="primary")
        
        # Results section
        if run_optimization_button:
            if not sequence_input.strip():
                st.error("Please enter a DNA sequence")
            else:
                with st.spinner("Processing sequence..."):
                    result, error = run_single_optimization(sequence_input, optimization_method, bias_weight)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.success("Optimization completed successfully")
                    
                    if optimization_method == "In-Frame Analysis":
                        df = pd.DataFrame(result)
                        st.subheader("In-Frame Analysis Results")
                        
                        if not df.empty and 'CAI_Weight' in df.columns:
                            # Statistics
                            cai_weights = df['CAI_Weight'].tolist()
                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                            with col_stat1:
                                st.metric("Average CAI", f"{np.mean(cai_weights):.3f}")
                            with col_stat2:
                                st.metric("Min CAI", f"{np.min(cai_weights):.3f}")
                            with col_stat3:
                                st.metric("Max CAI", f"{np.max(cai_weights):.3f}")
                            with col_stat4:
                                gc_content = calculate_gc_content(sequence_input)
                                st.metric("GC Content", f"{gc_content:.1f}%")
                            
                            st.dataframe(df, use_container_width=True)
                    
                    elif optimization_method == "+1 Frame Analysis":
                        st.subheader("+1 Frame Analysis Results")
                        
                        # Create metrics display
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        with metric_col1:
                            st.metric("Sequence Length", f"{result['Sequence_Length']} bp")
                        with metric_col2:
                            st.metric("Protein Length", f"{result['Protein_Length']} aa")
                        with metric_col3:
                            st.metric("GC Content", f"{result['GC_Content']:.1f}%")
                        with metric_col4:
                            st.metric("Total +1 Stops", result['Plus1_Total_Stops'])
                        
                        # Stop codon breakdown
                        if result['Plus1_Total_Stops'] > 0:
                            st.subheader("Stop Codon Distribution (+1 Frame)")
                            stop_data = {
                                'Codon': ['TAA', 'TAG', 'TGA'],
                                'Count': [result['Plus1_TAA_Count'], result['Plus1_TAG_Count'], result['Plus1_TGA_Count']]
                            }
                            stop_df = pd.DataFrame(stop_data)
                            stop_df = stop_df[stop_df['Count'] > 0]
                            
                            if not stop_df.empty:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(stop_df)]
                                ax.pie(stop_df['Count'], labels=stop_df['Codon'], colors=colors, autopct='%1.1f%%')
                                ax.set_title('Stop Codon Distribution in +1 Frame')
                                st.pyplot(fig)
                                plt.close()
                    
                    else:  # Standard Codon Optimization
                        st.subheader("Optimization Results")
                        
                        col_seq1, col_seq2 = st.columns(2)
                        
                        with col_seq1:
                            st.text_area("Original Sequence", result['Original_DNA'], height=120)
                        with col_seq2:
                            st.text_area("Optimized Sequence", result['Optimized_DNA'], height=120)
                        
                        # Comparison metrics
                        orig_gc = calculate_gc_content(result['Original_DNA'])
                        opt_gc = calculate_gc_content(result['Optimized_DNA'])
                        
                        st.subheader("Comparison")
                        comp_col1, comp_col2, comp_col3 = st.columns(3)
                        with comp_col1:
                            st.metric("Original GC%", f"{orig_gc:.1f}%")
                        with comp_col2:
                            st.metric("Optimized GC%", f"{opt_gc:.1f}%")
                        with comp_col3:
                            st.metric("GC Change", f"{opt_gc-orig_gc:+.1f}%")
    
    with tab2:
        st.header("Batch Optimization")
        st.markdown("Upload multiple sequences for batch optimization")
        
        batch_file = st.file_uploader(
            "Upload Sequence File",
            type=['txt', 'fasta', 'fa'],
            help="Upload a text file with sequences or FASTA format file"
        )
        
        if batch_file is not None:
            try:
                content = batch_file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                
                sequences = []
                
                if content.strip().startswith('>'):
                    # FASTA format
                    lines = content.strip().splitlines()
                    current_seq, current_name = "", ""
                    for line in lines:
                        line = line.strip()
                        if line.startswith('>'):
                            if current_seq:
                                sequences.append((current_name, current_seq))
                            current_name, current_seq = line[1:].strip(), ""
                        else:
                            current_seq += line.upper()
                    if current_seq:
                        sequences.append((current_name, current_seq))
                else:
                    # Text format
                    lines = [line.strip() for line in content.splitlines() if line.strip()]
                    for i, line in enumerate(lines):
                        sequences.append((f"Sequence_{i+1}", line.upper()))
                
                if sequences:
                    st.success(f"Loaded {len(sequences)} sequences")
                    
                    batch_method = st.selectbox(
                        "Batch Optimization Method",
                        ["In-Frame Analysis", "+1 Frame Analysis", "Standard Codon Optimization"]
                    )
                    
                    if st.button("Process Batch", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results = []
                        
                        for i, (name, seq) in enumerate(sequences):
                            status_text.text(f"Processing {name}...")
                            progress_bar.progress((i + 1) / len(sequences))
                            
                            result, error = run_single_optimization(seq, batch_method, bias_weight)
                            if error:
                                results.append({'Sequence_Name': name, 'Error': error})
                            else:
                                result_with_name = result.copy()
                                result_with_name['Sequence_Name'] = name
                                results.append(result_with_name)
                        
                        status_text.text("Batch processing completed!")
                        
                        if results:
                            batch_df = pd.DataFrame(results)
                            cols = ['Sequence_Name'] + [col for col in batch_df.columns if col != 'Sequence_Name']
                            batch_df = batch_df[cols]
                            
                            st.subheader("Batch Results")
                            st.dataframe(batch_df, use_container_width=True)
                            
                            # Download
                            excel_data = create_download_link(batch_df, f"Batch_{batch_method}_{len(sequences)}_sequences.xlsx")
                            st.download_button(
                                label="Download Results (Excel)",
                                data=excel_data,
                                file_name=f"Batch_{batch_method}_{len(sequences)}_sequences.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.header("About")
        st.markdown("""
        ### DNA Codon Optimization Tool
        
        **Working Version with Core Features**
        
        **Features:**
        - ✅ Single sequence optimization and analysis
        - ✅ Batch processing
        - ✅ In-frame CAI analysis
        - ✅ +1 frame stop codon analysis
        - ✅ Standard codon optimization
        - ✅ Built-in human codon usage data
        - ✅ Results download
        
        **Methods:**
        - **In-Frame Analysis**: Calculates CAI weights for sequence assessment
        - **+1 Frame Analysis**: Analyzes stop codons and GC content
        - **Standard Codon Optimization**: Uses most frequent codons
        
        This version includes built-in codon usage data and should work immediately.
        """)
    
    st.success("🎉 Application loaded successfully!")

if __name__ == "__main__":
    main()
