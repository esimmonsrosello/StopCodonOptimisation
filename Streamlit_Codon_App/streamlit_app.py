import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from collections import defaultdict, Counter
from Bio.Seq import Seq
import io

# Configure page
st.set_page_config(
    page_title="DNA Codon Optimization Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple theme colors
COLORS = {
    'primary': '#4ECDC4',
    'secondary': '#FF6B6B',
    'accent': '#45B7D1'
}

# Initialize session state
if 'genetic_code' not in st.session_state:
    st.session_state.genetic_code = {
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
    }

if 'preferred_codons' not in st.session_state:
    st.session_state.preferred_codons = {
        'F': 'TTC', 'L': 'CTG', 'I': 'ATC', 'V': 'GTG', 'Y': 'TAC', 'H': 'CAC', 'Q': 'CAG',
        'N': 'AAC', 'K': 'AAG', 'D': 'GAC', 'E': 'GAG', 'S': 'AGC', 'P': 'CCC', 'T': 'ACC',
        'A': 'GCC', 'C': 'TGC', 'W': 'TGG', 'R': 'CGC', 'G': 'GGC', '*': 'TAA'
    }

# Utility functions
def validate_dna_sequence(sequence):
    """Validate DNA sequence"""
    if not sequence:
        return False, "", "No DNA sequence provided"
    cleaned = sequence.upper().replace('\n', '').replace(' ', '').replace('\t', '').replace('U', 'T')
    invalid_bases = set(cleaned) - set('ATGC')
    if invalid_bases:
        return False, "", f"Invalid characters found: {', '.join(invalid_bases)}"
    return True, cleaned, ""

def translate_dna(seq):
    """Translate DNA sequence to protein"""
    protein = ""
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3].upper()
        aa = st.session_state.genetic_code.get(codon, '?')
        protein += aa
    return protein

def calculate_gc_content(sequence):
    """Calculate GC content percentage"""
    if not sequence:
        return 0.0
    clean_seq = sequence.upper().replace(' ', '').replace('\n', '')
    valid_bases = [base for base in clean_seq if base in 'ATGC']
    if not valid_bases:
        return 0.0
    gc_count = sum(1 for base in valid_bases if base in 'GC')
    return (gc_count / len(valid_bases)) * 100

def codon_optimize(protein_seq):
    """Simple codon optimization"""
    preferred_codons = st.session_state.preferred_codons
    optimized = ''.join(preferred_codons.get(aa, 'NNN') for aa in protein_seq if aa != 'X')
    return optimized

def number_of_plus1_stops(dna_seq):
    """Count stop codons in +1 frame"""
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    stop_codons_set = {"TAA", "TAG", "TGA"}
    counts = Counter()
    
    for i in range(1, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        if codon in stop_codons_set:
            counts[codon] += 1
    
    total_stops = sum(counts.values())
    return {'TAA': counts['TAA'], 'TAG': counts['TAG'], 'TGA': counts['TGA'], 'total': total_stops}

def main():
    """Main application"""
    st.title("🧬 DNA Codon Optimization Tool")
    st.markdown("**Streamlined version - fully functional**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.success("✅ Built-in human codon usage loaded")
        st.info("This is a simplified version with core functionality")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🧪 Single Sequence", "📊 Batch Processing", "ℹ️ About"])
    
    with tab1:
        st.header("Single Sequence Optimization")
        
        # Input
        sequence_input = st.text_area(
            "DNA Sequence:",
            height=150,
            placeholder="Enter DNA sequence (A, T, G, C only)..."
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            method = st.selectbox(
                "Method:",
                ["Sequence Analysis", "Standard Codon Optimization"]
            )
        with col2:
            analyze_btn = st.button("🔬 Analyze", type="primary")
        
        if analyze_btn and sequence_input.strip():
            is_valid, clean_seq, error_msg = validate_dna_sequence(sequence_input)
            
            if not is_valid:
                st.error(error_msg)
            else:
                st.success("✅ Analysis complete!")
                
                # Basic analysis
                protein_seq = translate_dna(clean_seq)
                gc_content = calculate_gc_content(clean_seq)
                plus1_stops = number_of_plus1_stops(clean_seq)
                
                # Display results
                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                with col_res1:
                    st.metric("DNA Length", f"{len(clean_seq)} bp")
                with col_res2:
                    st.metric("Protein Length", f"{len(protein_seq)} aa")
                with col_res3:
                    st.metric("GC Content", f"{gc_content:.1f}%")
                with col_res4:
                    st.metric("+1 Frame Stops", plus1_stops['total'])
                
                # Show sequences
                if method == "Standard Codon Optimization":
                    optimized_seq = codon_optimize(protein_seq)
                    
                    st.subheader("📋 Results")
                    col_seq1, col_seq2 = st.columns(2)
                    
                    with col_seq1:
                        st.text_area("Original DNA:", clean_seq, height=120)
                    with col_seq2:
                        st.text_area("Optimized DNA:", optimized_seq, height=120)
                    
                    # Comparison
                    orig_gc = calculate_gc_content(clean_seq)
                    opt_gc = calculate_gc_content(optimized_seq)
                    
                    st.markdown("**📊 Comparison:**")
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        st.metric("Original GC%", f"{orig_gc:.1f}%")
                    with comp_col2:
                        st.metric("Optimized GC%", f"{opt_gc:.1f}%", delta=f"{opt_gc-orig_gc:+.1f}%")
                
                else:
                    st.subheader("📋 Sequence Analysis")
                    st.text_area("Translated Protein:", protein_seq, height=120)
                    
                    # Stop codon breakdown
                    if plus1_stops['total'] > 0:
                        st.markdown("**🛑 +1 Frame Stop Codons:**")
                        stop_data = {
                            'Stop Codon': ['TAA', 'TAG', 'TGA'],
                            'Count': [plus1_stops['TAA'], plus1_stops['TAG'], plus1_stops['TGA']]
                        }
                        st.dataframe(pd.DataFrame(stop_data), use_container_width=True)
    
    with tab2:
        st.header("Batch Processing")
        st.info("📁 Upload a FASTA file or text file with multiple sequences")
        
        uploaded_file = st.file_uploader("Choose file", type=['txt', 'fasta', 'fa'])
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            
            # Simple parsing
            sequences = []
            if content.startswith('>'):
                # FASTA format
                lines = content.split('\n')
                current_seq = ""
                current_name = ""
                for line in lines:
                    if line.startswith('>'):
                        if current_seq:
                            sequences.append((current_name, current_seq))
                        current_name = line[1:].strip()
                        current_seq = ""
                    else:
                        current_seq += line.strip().upper()
                if current_seq:
                    sequences.append((current_name, current_seq))
            else:
                # Plain text
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                for i, line in enumerate(lines):
                    sequences.append((f"Sequence_{i+1}", line.upper()))
            
            if sequences:
                st.success(f"✅ Loaded {len(sequences)} sequences")
                
                if st.button("🔬 Process All"):
                    results = []
                    progress = st.progress(0)
                    
                    for i, (name, seq) in enumerate(sequences):
                        is_valid, clean_seq, _ = validate_dna_sequence(seq)
                        if is_valid:
                            protein = translate_dna(clean_seq)
                            gc_content = calculate_gc_content(clean_seq)
                            plus1_stops = number_of_plus1_stops(clean_seq)
                            
                            results.append({
                                'Name': name,
                                'DNA_Length': len(clean_seq),
                                'Protein_Length': len(protein),
                                'GC_Content': f"{gc_content:.1f}%",
                                'Plus1_Stops': plus1_stops['total']
                            })
                        
                        progress.progress((i + 1) / len(sequences))
                    
                    if results:
                        st.subheader("📊 Batch Results")
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results (CSV)",
                            csv,
                            "batch_results.csv",
                            "text/csv"
                        )
    
    with tab3:
        st.header("About")
        st.markdown("""
        ### 🧬 DNA Codon Optimization Tool
        
        **Streamlined Version - Fully Functional**
        
        **Features:**
        - ✅ DNA sequence validation and analysis
        - ✅ Protein translation
        - ✅ GC content calculation
        - ✅ +1 frame stop codon analysis
        - ✅ Standard codon optimization
        - ✅ Batch processing of multiple sequences
        - ✅ Results download
        
        **Built-in Human Codon Usage:**
        This version includes optimized human codon preferences for immediate use.
        
        **Usage:**
        1. Paste your DNA sequence in the Single Sequence tab
        2. Choose analysis type
        3. Click Analyze to see results
        4. For multiple sequences, use the Batch Processing tab
        
        **Version:** Streamlined v1.0
        """)

if __name__ == "__main__":
    main()
