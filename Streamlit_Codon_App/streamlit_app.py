#Add mouse codon usage table
#Make it so i can 3' tag vaccines with a peptide if/+1


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
from openpyxl.drawing.image import Image
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

# Configuration and Constants
BIAS_WEIGHT_DEFAULT = 1
FRAME_OFFSET = 1
VALID_DNA_BASES = 'ATGC'
CONFIG_FILE = "codon_optimizer_config.json"
DEFAULT_CONFIG = {
    "codon_file_path": "HumanCodons.xlsx",
    "bias_weight": BIAS_WEIGHT_DEFAULT,
    "auto_open_files": True,
    "default_output_dir": "."
}

# --- Theme Definitions ---
THEMES = {
    "Default": {
        "info": "Default color scheme with vibrant, high-contrast colors.",
        "colors": {
            "utr5": "#1900FF",
            "cds": "#4ECDC4",
            "utr3": "#FF6B6B",
            "signal_peptide": "#8A2BE2",
            "optimization": {'original': '#FF8A80', 'optimized': '#4ECDC4'},
            "analysis": ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD'],
            "gradient": ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2']
        }
    },
    "Oceanic": {
        "info": "A cool-toned theme inspired by the ocean.",
        "colors": {
            "utr5": "#006994",
            "cds": "#00A5AD",
            "utr3": "#88D8B0",
            "signal_peptide": "#58A4B0",
            "optimization": {'original': '#F9A825', 'optimized': '#00A5AD'},
            "analysis": ['#00A5AD', '#58A4B0', '#88D8B0', '#B3E5FC', '#4DD0E1', '#26C6DA', '#00BCD4', '#00ACC1'],
            "gradient": ['#E0F7FA', '#B2EBF2', '#80DEEA', '#4DD0E1', '#26C6DA', '#00BCD4', '#00ACC1', '#0097A7']
        }
    },
    "Sunset": {
        "info": "A warm-toned theme reminiscent of a sunset.",
        "colors": {
            "utr5": "#D9534F",
            "cds": "#F0AD4E",
            "utr3": "#5CB85C",
            "signal_peptide": "#E57373",
            "optimization": {'original': '#D9534F', 'optimized': '#F0AD4E'},
            "analysis": ['#F0AD4E', '#E57373', '#FF8A65', '#FFB74D', '#FFD54F', '#FFF176', '#DCE775', '#AED581'],
            "gradient": ['#FFF3E0', '#FFE0B2', '#FFCC80', '#FFB74D', '#FFA726', '#FF9800', '#FB8C00', '#F57C00']
        }
    }
}

# --- App Theme CSS --- (for styling the Streamlit UI itself)
APP_THEMES_CSS = {
    "Default": "",  # No custom CSS for the default theme
    "Oceanic": """
        <style>
            [data-testid="stAppViewContainer"] {
                background-color: #F0F8FF;
            }
            [data-testid="stSidebar"] {
                background-color: #E0F7FA;
            }
            h1, h2, h3, h4, h5, h6, p, label, .st-emotion-cache-16txtl3, .st-emotion-cache-1jicfl2 {
                color: #004D40;
            }
        </style>
    """,
    "Sunset": """
        <style>
            [data-testid="stAppViewContainer"] {
                background-color: #FFF3E0;
            }
            [data-testid="stSidebar"] {
                background-color: #FFE0B2;
            }
            h1, h2, h3, h4, h5, h6, p, label, .st-emotion-cache-16txtl3, .st-emotion-cache-1jicfl2 {
                color: #5D4037;
            }
        </style>
    """
}

def inject_app_theme():
    """Injects the CSS for the currently selected theme."""
    theme_css = APP_THEMES_CSS.get(st.session_state.active_theme, "")
    if theme_css:
        st.markdown(theme_css, unsafe_allow_html=True)


# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = DEFAULT_CONFIG.copy()
if 'active_theme' not in st.session_state:
    st.session_state.active_theme = "Default"
if 'accumulated_results' not in st.session_state:
    st.session_state.accumulated_results = []
if 'batch_accumulated_results' not in st.session_state:
    st.session_state.batch_accumulated_results = []
if 'mrna_design_cds_paste' not in st.session_state:
    st.session_state.mrna_design_cds_paste = ""
if 'run_counter' not in st.session_state:
    st.session_state.run_counter = 0
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
# Database search results caching
if 'cached_search_results' not in st.session_state:
    st.session_state.cached_search_results = None
if 'cached_search_query' not in st.session_state:
    st.session_state.cached_search_query = ""
if 'cached_download_df' not in st.session_state:
    st.session_state.cached_download_df = None

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('codon_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
Slippery_Motifs = {"TTTT", "TTTC"}
PLUS1_STOP_CODONS = {"TAA", "TAG"}
PLUS1_STOP_MOTIFS = {"TAATAA", "TAGTAG", "TAGTAA", "TAATAG"}
STANDARD_GENETIC_CODE = {
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

synonymous_codons = defaultdict(list)
for codon_val, aa_val in STANDARD_GENETIC_CODE.items(): 
    synonymous_codons[aa_val].append(codon_val)
    
FIRST_AA_CANDIDATES = ['L', 'I', 'V']
SECOND_AA_CANDIDATES = ['V', 'I']

# Utility Functions
def calculate_gc_window(sequence, position, window_size=25):
    """Calculate GC content for a sliding window around a given position"""
    # Convert position from 1-based to 0-based indexing
    center_pos = (position - 1) * 3  # Convert amino acid position to nucleotide position
    
    # Calculate window boundaries
    start = max(0, center_pos - window_size // 2)
    end = min(len(sequence), center_pos + window_size // 2 + 1)
    
    # Extract window sequence
    window_seq = sequence[start:end]
    
    if len(window_seq) == 0:
        return 0.0
    
    # Calculate GC content
    gc_count = sum(1 for base in window_seq.upper() if base in 'GC')
    return (gc_count / len(window_seq)) * 100

def get_consistent_color_palette(n_colors, palette_type="optimization"):
    """Generate consistent color palettes for charts based on the active theme"""
    theme_colors = THEMES[st.session_state.active_theme]["colors"]
    
    if palette_type == "optimization":
        return theme_colors["optimization"]
    elif palette_type == "analysis":
        base_colors = theme_colors["analysis"]
        return [base_colors[i % len(base_colors)] for i in range(n_colors)]
    elif palette_type == "gradient":
        return theme_colors["gradient"]

def display_copyable_sequence(sequence, label, key_suffix=""):
    """Display sequence in a copyable format"""
    st.text_area(
        label,
        sequence,
        height=120,
        key=f"copy_{key_suffix}",
        help="Click in the text area and use Ctrl+A to select all, then Ctrl+C to copy"
    )

def display_colored_mrna_sequence(utr5_seq, cds_seq, utr3_seq, signal_peptide_seq="", key_suffix=""):
    """Display mRNA sequence with 5'UTR, CDS, and 3'UTR highlighted in different colors."""
    st.subheader("Full mRNA Sequence (Colored)")

    # Define colors for each section from the active theme
    theme_colors = THEMES[st.session_state.active_theme]["colors"]
    color_utr5 = theme_colors["utr5"]
    color_cds = theme_colors["cds"]
    color_utr3 = theme_colors["utr3"]
    color_signal_peptide = theme_colors["signal_peptide"]

    # Create HTML string with colored spans
    if signal_peptide_seq:
        colored_html = f"""
        <div style="font-family: monospace; white-space: pre-wrap; word-break: break-all; background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.8em;">
            <span style="color: {color_utr5}; font-weight: bold;">{utr5_seq}</span><span style="color: {color_signal_peptide}; font-weight: bold;">{signal_peptide_seq}</span><span style="color: {color_cds}; font-weight: bold;">{cds_seq}</span><span style="color: {color_utr3}; font-weight: bold;">{utr3_seq}</span>
        </div>
        """
        full_sequence = utr5_seq + signal_peptide_seq + cds_seq + utr3_seq
    else:
        colored_html = f"""
        <div style="font-family: monospace; white-space: pre-wrap; word-break: break-all; background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.8em;">
            <span style="color: {color_utr5}; font-weight: bold;">{utr5_seq}</span><span style="color: {color_cds}; font-weight: bold;">{cds_seq}</span><span style="color: {color_utr3}; font-weight: bold;">{utr3_seq}</span>
        </div>
        """
        full_sequence = utr5_seq + cds_seq + utr3_seq
    st.markdown(colored_html, unsafe_allow_html=True)

    # Also provide a copyable text area for the full sequence
    full_sequence = utr5_seq + cds_seq + utr3_seq
    st.text_area(
        "Copy Full mRNA Sequence:",
        full_sequence,
        height=120,
        key=f"copy_full_mrna_{key_suffix}",
        help="Click in the text area and use Ctrl+A to select all, then Ctrl+C to copy"
    )
    # Update legend
    legend_html = f"""
    <div style="font-size: 0.8em; color: gray;">
        <span style="color: {color_utr5};">■</span> 5' UTR ({len(utr5_seq)} bp) &nbsp;&nbsp;
    """
    if signal_peptide_seq:
        legend_html += f"""<span style="color: {color_signal_peptide};">■</span> Signal Peptide ({len(signal_peptide_seq)} bp) &nbsp;&nbsp;"""
    
    legend_html += f"""
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    
    
def create_geneious_like_visualization(utr5_seq, cds_seq, utr3_seq, signal_peptide_seq="", key_suffix=""):
    """
    Create a Geneious-like visualization of the mRNA sequence with nucleotides and amino acids.
    Amino acids are only shown for the coding sequence (signal peptide + CDS).
    """
    
    # Generate a unique suffix based on key_suffix and a random value
    unique_id = f"{key_suffix}_{id(utr5_seq)}"
    
    # Add controls for the visualization
    st.markdown("### Visualization Controls")
    
    # Toggle button for full sequence view
    toggle_key = f"show_full_sequence_{unique_id}"
    if st.button("Toggle Full Sequence View", key=f"toggle_full_seq_{unique_id}"):
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = True
        else:
            st.session_state[toggle_key] = not st.session_state[toggle_key]
    
    # Show full sequence in a scrollable area if requested
    if st.session_state.get(toggle_key, False):
        st.subheader("Full mRNA Sequence (Scroll to View)")
        full_sequence = utr5_seq + (signal_peptide_seq if signal_peptide_seq else "") + cds_seq + utr3_seq
        st.text_area(
            "Full mRNA Sequence",
            full_sequence,
            height=120,
            key=f"full_mrna_sequence_{unique_id}",
            help="Click in the text area and use Ctrl+A to select all, then Ctrl+C to copy"
        )
    
    # Create the detailed visualization
    st.markdown("### Sequence Visualization")
    
    # Get theme colors
    theme_colors = THEMES[st.session_state.active_theme]["colors"]
    color_utr5 = theme_colors["utr5"]
    color_cds = theme_colors["cds"]
    color_utr3 = theme_colors["utr3"]
    color_signal_peptide = theme_colors["signal_peptide"]
    
    # Create the visualization sections
    sections = []
    
    # 5' UTR Section
    if utr5_seq:
        sections.append({
            'name': "5' UTR",
            'sequence': utr5_seq,
            'color': color_utr5,
            'show_aa': False
        })
    
    # Signal Peptide Section
    if signal_peptide_seq:
        sections.append({
            'name': "Signal Peptide",
            'sequence': signal_peptide_seq,
            'color': color_signal_peptide,
            'show_aa': True
        })
    
    # CDS Section
    if cds_seq:
        sections.append({
            'name': "CDS",
            'sequence': cds_seq,
            'color': color_cds,
            'show_aa': True
        })
    
    # 3' UTR Section
    if utr3_seq:
        sections.append({
            'name': "3' UTR",
            'sequence': utr3_seq,
            'color': color_utr3,
            'show_aa': False
        })
    
    # Display each section
    for section_idx, section in enumerate(sections):
        st.markdown(f"#### {section['name']} ({len(section['sequence'])} bp)")
        
        seq = section['sequence']
        color = section['color']
        show_aa = section['show_aa']
        
        # For coding sequences, we want to align codons properly
        # Use chunk size that's divisible by 3 for coding regions
        if show_aa:
            # Use 60 nucleotides (20 codons) for coding regions
            chunk_size = 60
        else:
            # Use 60 nucleotides for non-coding regions
            chunk_size = 60
        
        for chunk_idx, i in enumerate(range(0, len(seq), chunk_size)):
            chunk = seq[i:i+chunk_size]
            start_pos = i + 1
            end_pos = min(i + chunk_size, len(seq))
            
            # Display position info
            st.markdown(f"**Position {start_pos}-{end_pos}**")
            
            if show_aa and len(chunk) >= 3:
                # For coding sequences, create aligned nucleotide and amino acid display
                
                # Split nucleotides into codons for better visualization
                codons = []
                amino_acids = []
                
                for j in range(0, len(chunk) - 2, 3):
                    codon = chunk[j:j+3]
                    if len(codon) == 3:
                        codons.append(codon)
                        aa = st.session_state.genetic_code.get(codon.upper(), 'X')
                        amino_acids.append(aa)
                
                # Handle remaining nucleotides (less than 3)
                remaining = chunk[len(codons)*3:]
                if remaining:
                    codons.append(remaining + " " * (3 - len(remaining)))  # Pad with spaces
                    amino_acids.append(" ")  # Space for incomplete codon
                
                
                # Create spaced codon display
                spaced_codons = "   ".join(codons)  # 3 spaces between codons

                # Center each AA under its codon
                spaced_aas = "   ".join([f" {aa} " for aa in amino_acids])  # pad each AA with 1 space

                # Display nucleotides (codons)
                nucleotide_html = f"""
                <div style="
                    font-family: 'Courier New', monospace; 
                    background: #f8f9fa; 
                    padding: 10px; 
                    border-radius: 5px; 
                    border-left: 4px solid {color};
                    margin: 5px 0;
                ">
                    <div style="
                        color: {color}; 
                        font-weight: bold; 
                        font-size: 1.1em; 
                        letter-spacing: 1px;
                        word-break: break-all;
                    ">
                        {spaced_codons}
                    </div>
                </div>
                """
                st.markdown(nucleotide_html, unsafe_allow_html=True)
                
                # Display amino acids aligned with codons
                aa_html = f"""
                <div style="
                    font-family: 'Courier New', monospace; 
                    background: #fff; 
                    padding: 5px 10px; 
                    border-radius: 3px; 
                    border-left: 4px solid {color};
                    margin: 0 0 10px 0;
                ">
                    <div style="
                        color: #333; 
                        font-size: 1.0em; 
                        letter-spacing: 13.5px;
                        font-weight: bold;
                    ">
                        {spaced_aas}
                    </div>
                </div>
                """
                st.markdown(aa_html, unsafe_allow_html=True)
                
            else:
                # For non-coding sequences, just display nucleotides
                nucleotide_html = f"""
                <div style="
                    font-family: 'Courier New', monospace; 
                    background: #f8f9fa; 
                    padding: 10px; 
                    border-radius: 5px; 
                    border-left: 4px solid {color};
                    margin: 5px 0;
                ">
                    <div style="
                        color: {color}; 
                        font-weight: bold; 
                        font-size: 1.1em; 
                        letter-spacing: 1px;
                        word-break: break-all;
                    ">
                        {chunk}
                    </div>
                </div>
                """
                st.markdown(nucleotide_html, unsafe_allow_html=True)
            
            # Add some spacing between chunks
            if i + chunk_size < len(seq):
                st.markdown("---")
    
    # Add summary information
    st.markdown("### Sequence Summary")
    
    total_length = len(utr5_seq) + len(signal_peptide_seq) + len(cds_seq) + len(utr3_seq)
    coding_length = len(signal_peptide_seq) + len(cds_seq)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Length", f"{total_length} bp")
    with col2:
        st.metric("Coding Length", f"{coding_length} bp")
    with col3:
        if coding_length > 0:
            protein_length = coding_length // 3
            st.metric("Protein Length", f"{protein_length} aa")
        else:
            st.metric("Protein Length", "0 aa")
    with col4:
        full_seq = utr5_seq + signal_peptide_seq + cds_seq + utr3_seq
        gc_content = calculate_gc_content(full_seq) if full_seq else 0
        st.metric("GC Content", f"{gc_content:.1f}%")
    
    # Legend
    st.markdown("### Legend")
    legend_items = []
    
    if utr5_seq:
        legend_items.append(f'<span style="color: {color_utr5}; font-weight: bold;">■</span> 5\' UTR')
    if signal_peptide_seq:
        legend_items.append(f'<span style="color: {color_signal_peptide}; font-weight: bold;">■</span> Signal Peptide')
    if cds_seq:
        legend_items.append(f'<span style="color: {color_cds}; font-weight: bold;">■</span> CDS')
    if utr3_seq:
        legend_items.append(f'<span style="color: {color_utr3}; font-weight: bold;">■</span> 3\' UTR')
    
    legend_html = f"""
    <div style="font-size: 0.9em; margin: 10px 0;">
        {' &nbsp;&nbsp; '.join(legend_items)}
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    
    # Add explanation
    st.info("💡 **Reading Guide**: In coding regions, nucleotides are grouped by codons (3 letters) with the corresponding amino acid shown below each codon.")

def create_interactive_cai_gc_plot(positions, cai_weights, amino_acids, sequence, seq_name, color='#4ECDC4'):
    """Create interactive plot combining CAI weights and GC content"""
    
    # Calculate 10bp window GC content for each position
    gc_content_10bp = [calculate_gc_window(sequence, pos, 25) for pos in positions]
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=[f'CAI Weights and 25bp GC Content - {seq_name}']
    )
    
    # Add CAI weights trace
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=cai_weights,
            mode='lines+markers',
            name='CAI Weight',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate='<b>Position:</b> %{x}<br><b>CAI Weight:</b> %{y:.3f}<br><b>AA:</b> %{customdata}<extra></extra>',
            customdata=amino_acids
        ),
        secondary_y=False,
    )
    
    # Add GC content trace
    theme_colors = get_consistent_color_palette(1, "optimization")
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=gc_content_10bp,
            mode='lines',
            name='25bp GC Content',
            line=dict(color=theme_colors['original'], width=2, dash='dot'),
            hovertemplate='<b>Position:</b> %{x}<br><b>25bp GC Content:</b> %{y:.1f}%<extra></extra>',
            opacity=0.7
        ),
        secondary_y=True,
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text="Amino Acid Position")
    
    # Set y-axes titles
    fig.update_yaxes(title_text="CAI Weight", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="GC Content (%)", secondary_y=True, range=[0, 100])
    
    # Update layout
    fig.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_interactive_cai_stop_codon_plot(positions, cai_weights, amino_acids, stop_codon_positions, seq_name, frame_type, color='#4ECDC4'):
    """Create interactive plot combining CAI weights and stop codon locations"""
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add CAI weights trace
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=cai_weights,
            mode='lines+markers',
            name='CAI Weight',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate='<b>Position:</b> %{x}<br><b>CAI Weight:</b> %{y:.3f}<br><b>AA:</b> %{customdata}<extra></extra>',
            customdata=amino_acids
        ),
        secondary_y=False,
    )
    
    # Add stop codon bars
    if stop_codon_positions:
        theme_colors = get_consistent_color_palette(1, "optimization")
        fig.add_trace(
            go.Bar(
                x=stop_codon_positions,
                y=[1] * len(stop_codon_positions), # Bars will go up to y=1 on secondary axis
                name=f'{frame_type} Stop Codons',
                marker_color=theme_colors['original'],  # Use theme color for stops
                opacity=0.6,
                width=0.8,
                hovertemplate='<b>Position:</b> %{x}<br><b>Stop Codon</b><extra></extra>'
            ),
            secondary_y=True,
        )

    # Set x-axis title
    fig.update_xaxes(title_text="Amino Acid Position")
    
    # Set y-axes titles
    fig.update_yaxes(title_text="CAI Weight", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="Stop Codon", secondary_y=True, showticklabels=False, range=[0, 1])
    
    # Update layout
    fig.update_layout(
        title=f'CAI Weights and {frame_type} Stop Codon Locations - {seq_name}',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_interactive_bar_chart(x_data, y_data, labels, title, color_scheme='viridis'):
    """Create interactive bar chart using the active theme"""
    theme_analysis_colors = get_consistent_color_palette(len(x_data), "analysis")
    fig = go.Figure(data=go.Bar(
        x=x_data,
        y=y_data,
        text=[f'{val:.1f}' for val in y_data],
        textposition='auto',
        marker_color=theme_analysis_colors,
        hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Category",
        yaxis_title="Value",
        height=400,
        showlegend=False
    )
    
    return fig

def create_interactive_pie_chart(values, labels, title):
    """Create interactive pie chart using the active theme"""
    theme_analysis_colors = get_consistent_color_palette(len(labels), "analysis")
    fig = go.Figure(data=go.Pie(
        labels=labels,
        values=values,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        textinfo='label+percent',
        marker=dict(colors=theme_analysis_colors)
    ))
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=True
    )
    
    return fig

def create_interactive_comparison_chart(sequences, original_values, optimized_values, metric_name, y_title):
    """Create interactive before/after comparison chart"""
    fig = go.Figure()
    
    colors = get_consistent_color_palette(1, "optimization")
    
    # Add original values
    fig.add_trace(go.Bar(
        name='Original',
        x=sequences,
        y=original_values,
        marker_color=colors['original'],
        hovertemplate='<b>%{x}</b><br>Original ' + metric_name + ': %{y}<extra></extra>'
    ))
    
    # Add optimized values
    fig.add_trace(go.Bar(
        name='Optimized',
        x=sequences,
        y=optimized_values,
        marker_color=colors['optimized'],
        hovertemplate='<b>%{x}</b><br>Optimized ' + metric_name + ': %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{metric_name}: Before vs After Optimization',
        xaxis_title='Sequence',
        yaxis_title=y_title,
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_interactive_stacked_bar_chart(x_data, y_data_dict, title, y_title):
    """Create interactive stacked bar chart"""
    fig = go.Figure()
    
    colors = get_consistent_color_palette(len(y_data_dict), "analysis")
    
    for i, (label, values) in enumerate(y_data_dict.items()):
        fig.add_trace(go.Bar(
            name=label,
            x=x_data,
            y=values,
            marker_color=colors[i % len(colors)],
            hovertemplate=f'<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Sequence',
        yaxis_title=y_title,
        barmode='stack',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_enhanced_chart(data, chart_type, title, colors=None, xlabel="Sequence", ylabel="Value"):
    """Create enhanced charts with consistent styling"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set consistent styling
    ax.set_facecolor('#F8F9FA')
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    
    if colors is None:
        colors = get_consistent_color_palette(len(data), "analysis")
    
    if chart_type == "bar":
        bars = ax.bar(range(len(data)), data, color=colors, 
                      edgecolor='#2C3E50', linewidth=1.5, alpha=0.9)
        
        # Add value labels with consistent styling
        for bar, value in zip(bars, data):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + max(data) * 0.02,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='none', alpha=0.8))
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def calculate_enhanced_summary_stats(result, original_seq=""):
    """Calculate enhanced summary statistics"""
    stats = {}
    
    # Basic metrics
    if 'Sequence_Length' in result:
        stats['Sequence_Length_bp'] = result['Sequence_Length']
    if 'Protein_Length' in result:
        stats['Protein_Length_aa'] = result['Protein_Length']
    
    # GC Content
    if 'GC_Content' in result:
        stats['GC_Content_percent'] = f"{result['GC_Content']:.1f}%"
    
    # Stop codon change (instead of reduction)
    if 'Plus1_Total_Stops' in result:
        stats['Plus1_Stop_Count'] = result['Plus1_Total_Stops']
        if original_seq:
            orig_stops = number_of_plus1_stops(original_seq)
            change = result['Plus1_Total_Stops'] - orig_stops['total']
            stats['Stop_Codon_Change'] = f"{change:+d}"
    
    # Slippery motifs
    if 'Slippery_Motifs' in result:
        stats['Slippery_Motifs'] = result['Slippery_Motifs']
    
    # CAI metrics
    if 'CAI_Weights' in result and result['CAI_Weights']:
        try:
            weights = [float(w) for w in result['CAI_Weights'].split(',')]
            stats['Average_CAI'] = f"{sum(weights)/len(weights):.3f}"
            stats['Min_CAI'] = f"{min(weights):.3f}"
            stats['Max_CAI'] = f"{max(weights):.3f}"
        except:
            pass
    
    # Advanced metrics
    if original_seq and 'Optimized_DNA' in result:
        orig_gc = calculate_gc_content(original_seq)
        opt_gc = calculate_gc_content(result['Optimized_DNA'])
        stats['GC_Content_Change'] = f"{opt_gc - orig_gc:+.1f}%"
    
    return stats

def count_specific_slippery_motifs(dna_seq):
    """Count specific slippery motifs (TTTT and TTTC) in coding sequence"""
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    start_pos, end_pos = find_coding_sequence_bounds(dna_seq_upper)
    
    counts = {'TTTT': 0, 'TTTC': 0, 'total': 0}
    
    if start_pos is None:
        return counts
    
    search_end = end_pos if end_pos is not None else len(dna_seq_upper) - 3
    
    for i in range(start_pos, search_end, 3):
        if i+4 <= len(dna_seq_upper):
            motif = dna_seq_upper[i:i+4]
            if motif == 'TTTT':
                counts['TTTT'] += 1
            elif motif == 'TTTC':
                counts['TTTC'] += 1
    
    counts['total'] = counts['TTTT'] + counts['TTTC']
    return counts

def validate_dna_sequence(sequence):
    """Validate DNA sequence and return cleaned version"""
    if not sequence:
        return False, "", "No DNA sequence provided"
    cleaned = sequence.upper().replace('\n', '').replace(' ', '').replace('\t', '').replace('U', 'T')
    invalid_bases = set(cleaned) - set(VALID_DNA_BASES)
    if invalid_bases:
        return False, "", f"Invalid characters found: {', '.join(invalid_bases)}. Only A, T, G, C allowed."
    if len(cleaned) % 3 != 0:
        logger.warning(f"Sequence length ({len(cleaned)}) is not a multiple of 3")
    return True, cleaned, ""

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

def calculate_local_gc_content(sequence, window_size=10, step_size=1):
    """
    Calculate GC content for overlapping windows of a given sequence.
    Returns a list of GC percentages for each window.
    """
    gc_percentages = []
    for i in range(0, len(sequence) - window_size + 1, step_size):
        window = sequence[i:i+window_size]
        gc_count = sum(1 for base in window.upper() if base in 'GC')
        gc_percentage = (gc_count / window_size) * 100
        gc_percentages.append(gc_percentage)
    return gc_percentages

def get_codon_gc_content(codon):
    """Calculate the GC content of a single 3-base codon."""
    if len(codon) != 3:
        return 0
    return (codon.upper().count('G') + codon.upper().count('C')) / 3.0 * 100

def adjust_gc_content(sequence, max_gc=70.0, min_gc=55.0):
    """
    Adjusts the GC content of a sequence to be within a target range by using synonymous codons.
    Prioritizes swapping high-GC codons for low-GC codons.
    """
    # Auto-load default codon file if available and not already loaded
    if not st.session_state.genetic_code and 'codon_data_loaded' not in st.session_state:
        default_codon_file = "HumanCodons.xlsx"
        if os.path.exists(default_codon_file):
            try:
                with open(default_codon_file, 'rb') as f:
                    file_content = f.read()
                genetic_code, codon_weights, preferred_codons, human_codon_usage, aa_to_codons, codon_df = load_codon_data_from_file(file_content)
                st.session_state.genetic_code = genetic_code
                st.session_state.codon_weights = codon_weights
                st.session_state.preferred_codons = preferred_codons
                st.session_state.human_codon_usage = human_codon_usage
                st.session_state.aa_to_codons = aa_to_codons
                st.session_state.codon_data_loaded = True
                st.session_state.codon_file_source = "Default (HumanCodons.xlsx)"
                st.success(f"Auto-loaded {len(codon_df)} codon entries from HumanCodons.xlsx")
            except Exception as e:
                st.warning(f"Could not auto-load HumanCodons.xlsx: {e}")
                # Don't stop the app, just continue without auto-loading
        else:
        # File doesn't exist, just continue - user can upload manually
            pass

    current_gc = calculate_gc_content(sequence)
    if current_gc <= max_gc:
        st.info(f"Initial GC content ({current_gc:.1f}%) is already within the target range (<= {max_gc}%) No adjustment needed.")
        return sequence

    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    new_codons = list(codons)
    
    # Create a list of potential swaps, prioritized by GC content reduction
    potential_swaps = []
    for i, codon in enumerate(codons):
        aa = st.session_state.genetic_code.get(codon)
        if not aa or aa == '*':
            continue

        current_codon_gc = get_codon_gc_content(codon)
        
        # Find synonymous codons with lower GC content
        for syn_codon, freq in st.session_state.aa_to_codons.get(aa, []):
            syn_codon_gc = get_codon_gc_content(syn_codon)
            if syn_codon_gc < current_codon_gc:
                gc_reduction = current_codon_gc - syn_codon_gc
                # Store index, new codon, and the reduction amount for prioritization
                potential_swaps.append({'index': i, 'new_codon': syn_codon, 'reduction': gc_reduction, 'original_codon': codon})

    # Sort swaps by the amount of GC reduction (descending)
    potential_swaps.sort(key=lambda x: x['reduction'], reverse=True)

    # Apply swaps until GC content is acceptable
    swapped_indices = set()
    for swap in potential_swaps:
        if current_gc <= max_gc:
            break # Stop if we've reached the target
        
        idx = swap['index']
        if idx not in swapped_indices:
            new_codons[idx] = swap['new_codon']
            swapped_indices.add(idx)
            # Recalculate GC content after the swap
            current_gc = calculate_gc_content("".join(new_codons))

    final_sequence = "".join(new_codons)
    final_gc = calculate_gc_content(final_sequence)
    st.success(f"GC content adjusted from {calculate_gc_content(sequence):.1f}% to {final_gc:.1f}%.")
    
    return final_sequence

def enforce_local_gc_content(sequence, target_max_gc=70.0, window_size=10, step_size=1):
    """
    Enforces local GC content by adjusting codons in windows exceeding target_max_gc.
    Attempts to maintain protein sequence.
    """
    if not st.session_state.aa_to_codons or not st.session_state.genetic_code:
        st.error("Codon usage data not loaded. Cannot enforce local GC adjustment.")
        return sequence

    current_sequence = list(sequence) # Convert to list for mutability
    original_protein = translate_dna("".join(current_sequence))
    changes_made = 0

    # Iterate and adjust
    for i in range(0, len(current_sequence) - window_size + 1, step_size):
        window_start = i
        window_end = i + window_size
        window_seq = "".join(current_sequence[window_start:window_end])
        
        local_gc = calculate_gc_content(window_seq) # Use the existing calculate_gc_content for the window

        if local_gc > target_max_gc:
            # Identify codons within this window that can be swapped
            # This is the most complex part:
            # 1. Find codons in the window.
            # 2. For each codon, find synonymous codons with lower GC content.
            # 3. Prioritize swaps that reduce GC and are within the window.
            # 4. Apply swap and re-check local GC.
            
            # For simplicity in this first pass, let's try a greedy approach:
            # Iterate through codons in the window and try to swap them if they are high GC
            
            # Map nucleotide position to codon index
            codon_indices_in_window = set()
            for bp_idx in range(window_start, window_end):
                codon_idx = bp_idx // 3
                codon_indices_in_window.add(codon_idx)

            # Sort to ensure consistent processing
            sorted_codon_indices = sorted(list(codon_indices_in_window))

            for codon_idx in sorted_codon_indices:
                codon_start_bp = codon_idx * 3
                codon_end_bp = codon_start_bp + 3

                # Ensure the codon is fully within the original sequence bounds
                if codon_end_bp <= len(sequence):
                    original_codon = "".join(current_sequence[codon_start_bp:codon_end_bp])
                    aa = st.session_state.genetic_code.get(original_codon)

                    if aa and aa != '*': # Don't optimize stop codons
                        original_codon_gc = get_codon_gc_content(original_codon)
                        
                        best_syn_codon = original_codon
                        max_gc_reduction = 0

                        # Find a synonymous codon with lower GC
                        for syn_c, _ in st.session_state.aa_to_codons.get(aa, []):
                            syn_c_gc = get_codon_gc_content(syn_c)
                            if syn_c_gc < original_codon_gc:
                                if (original_codon_gc - syn_c_gc) > max_gc_reduction:
                                    max_gc_reduction = original_codon_gc - syn_c_gc
                                    best_syn_codon = syn_c
                        
                        if best_syn_codon != original_codon:
                            # Temporarily apply the swap and check if it helps the local GC
                            temp_sequence_list = list(current_sequence)
                            temp_sequence_list[codon_start_bp:codon_end_bp] = list(best_syn_codon)
                            
                            temp_window_seq = "".join(temp_sequence_list[window_start:window_end])
                            temp_local_gc = calculate_gc_content(temp_window_seq)

                            if temp_local_gc <= target_max_gc: # If this swap fixes the window
                                current_sequence[codon_start_bp:codon_end_bp] = list(best_syn_codon)
                                changes_made += 1
                                # Re-check the current window's GC after a change
                                local_gc = calculate_gc_content("".join(current_sequence[window_start:window_end]))
                                if local_gc <= target_max_gc:
                                    break # Move to next window if this one is fixed
                            # else: # If the swap doesn't fix it, try another codon in the window or move on
    
    final_sequence = "".join(current_sequence)
    final_protein = translate_dna(final_sequence)

    if original_protein != final_protein:
        st.warning("Local GC adjustment changed protein sequence. Reverting to original CDS.")
        return sequence # Revert if protein sequence changed

    if changes_made > 0:
        st.success(f"Local GC content adjusted. {changes_made} codon swaps performed.")
    else:
        st.info("No local GC content adjustments needed or possible.")
    
    return final_sequence

def generate_detailed_mrna_summary(processed_cds, final_mrna_sequence, utr_5, utr_3):
    """Generate a detailed summary DataFrame for the designed mRNA."""
    
    # Basic lengths
    summary_data = {
        "Metric": ["Final mRNA Length", "5' UTR Length", "CDS Length", "3' UTR Length"],
        "Value": [f"{len(final_mrna_sequence)} bp", f"{len(utr_5)} bp", f"{len(processed_cds)} bp", f"{len(utr_3)} bp"]
    }
    
    # GC Content
    summary_data["Metric"].append("CDS GC Content")
    summary_data["Value"].append(f"{calculate_gc_content(processed_cds):.1f}%")
    
    # CAI
    cai_weights, _ = get_codon_weights_row(processed_cds)
    if cai_weights:
        summary_data["Metric"].extend(["Average CAI", "Min CAI", "Max CAI"])
        summary_data["Value"].extend([
            f"{sum(cai_weights)/len(cai_weights):.3f}",
            f"{min(cai_weights):.3f}",
            f"{max(cai_weights):.3f}"
        ])

    # +1 Stops
    plus1_stops = number_of_plus1_stops(processed_cds)
    summary_data["Metric"].extend(["+1 Total Stops", "+1 TAA", "+1 TAG", "+1 TGA"])
    summary_data["Value"].extend([
        plus1_stops['total'],
        plus1_stops['TAA'],
        plus1_stops['TAG'],
        plus1_stops['TGA']
    ])

    # -1 Stops
    minus1_stops = number_of_minus1_stops(processed_cds)
    summary_data["Metric"].extend(["-1 Total Stops", "-1 TAA", "-1 TAG", "-1 TGA"])
    summary_data["Value"].extend([
        minus1_stops['total'],
        minus1_stops['TAA'],
        minus1_stops['TAG'],
        minus1_stops['TGA']
    ])

    # Slippery Motifs
    slippery_count = number_of_slippery_motifs(utr_5 + processed_cds)
    summary_data["Metric"].append("Slippery Motifs")
    summary_data["Value"].append(slippery_count)
    
    return pd.DataFrame(summary_data)

def calculate_stops_per_100bp(sequence, plus1_stops):
    """Calculate +1 frame stops per 100bp"""
    if not sequence:
        return 0.0
    
    sequence_length_bp = len(sequence)
    if sequence_length_bp == 0:
        return 0.0
    
    stops_per_100bp = (plus1_stops / sequence_length_bp) * 100
    return stops_per_100bp

def translate_dna(seq):
    """Translate DNA sequence to protein"""
    protein = ""
    genetic_code = st.session_state.genetic_code
    for i in range(0, len(seq) - 2, 3):
        codon_val = seq[i:i+3].upper()
        aa = genetic_code.get(codon_val, '?')
        protein += aa
    return protein

def reverse_translate_highest_cai(protein_seq):
    """Reverse translates a protein sequence into DNA using the highest CAI codons."""
    if not st.session_state.preferred_codons:
        st.error("Codon usage data not loaded. Cannot reverse translate.")
        return ""
    
    dna_seq = ""
    for aa in protein_seq:
        # Handle stop codons if they appear in the protein sequence (e.g., from a partial sequence)
        if aa == '*':
            # Use TAA as a default stop codon for reverse translation
            dna_seq += "TAA"
        else:
            codon = st.session_state.preferred_codons.get(aa)
            if codon:
                dna_seq += codon
            else:
                # Fallback if no preferred codon found (should not happen for standard AAs)
                st.warning(f"No preferred codon found for amino acid: {aa}. Using NNN.")
                dna_seq += "NNN" # NNN for unknown codon
    return dna_seq

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

def find_coding_sequence_bounds(dna_seq):
    """Find start and stop positions of coding sequence, prioritizing ACCATG."""
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    stop_codons = {"TAA", "TAG", "TGA"}
    
    start_pos = None
    
    # Always prioritize finding the ACCATG Kozak sequence.
    accatg_pos = dna_seq_upper.find('ACCATG')
    if accatg_pos != -1:
        # The actual start codon (ATG) begins 3 bases into "ACCATG".
        start_pos = accatg_pos + 3
    else:
        # Fallback: if no ACCATG, find the first occurrence of ATG.
        atg_pos = dna_seq_upper.find('ATG')
        if atg_pos != -1:
            # The sequence starts at the beginning of the first ATG found.
            start_pos = atg_pos
            
    if start_pos is None:
        # If no start codon is found at all, we can't proceed.
        return None, None
    
    # Find end position - first in-frame stop codon, starting from our found start_pos.
    end_pos = None
    for i in range(start_pos, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        if len(codon) == 3 and codon in stop_codons:
            end_pos = i  # Position of the stop codon itself.
            break
            
    return start_pos, end_pos

def number_of_slippery_motifs(dna_seq):
    """Count slippery motifs in coding sequence"""
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    start_pos, end_pos = find_coding_sequence_bounds(dna_seq_upper)
    if start_pos is None:
        return 0
    
    search_end = end_pos if end_pos is not None else len(dna_seq_upper) - 3
    slippery_count = sum(1 for i in range(start_pos, search_end, 3) 
                        if dna_seq_upper[i:i+4] in Slippery_Motifs and i+4 <= len(dna_seq_upper))
    return slippery_count

def number_of_plus1_stops(dna_seq):
    """Count stop codons in +1 frame across the entire sequence"""
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    stop_codons_set = {"TAA", "TAG", "TGA"}
    
    counts = Counter()
    # Iterate through the sequence starting from the 1st base (0-indexed)
    # and check codons in the +1 frame (offset by 1 base)
    for i in range(1, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        if codon in stop_codons_set:
            counts[codon] += 1
    
    total_stops = sum(counts.values())
    return {'TAA': counts['TAA'], 'TAG': counts['TAG'], 'TGA': counts['TGA'], 'total': total_stops}

def number_of_minus1_stops(dna_seq):
    """Count stop codons in -1 frame across the entire sequence"""
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    stop_codons_set = {"TAA", "TAG", "TGA"}
    
    counts = Counter()
    # Iterate through the sequence starting from the 2nd base (0-indexed)
    # and check codons in the -1 frame (offset by 2 bases)
    for i in range(2, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        if codon in stop_codons_set:
            counts[codon] += 1
    
    total_stops = sum(counts.values())
    return {'TAA': counts['TAA'], 'TAG': counts['TAG'], 'TGA': counts['TGA'], 'total': total_stops}

def get_plus1_stop_positions(dna_seq):
    """Get positions of stop codons in +1 frame"""
    positions = []
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    start_pos, end_pos = find_coding_sequence_bounds(dna_seq_upper)
    if start_pos is None:
        return positions
     
    stop_codons_set = {"TAA", "TAG", "TGA"}
    plus1_start = start_pos + 1
    search_end = end_pos if end_pos is not None else len(dna_seq_upper) - 2
    
    for i in range(plus1_start, search_end, 3):
        if i+3 <= len(dna_seq_upper):
            codon = dna_seq_upper[i:i+3]
            if codon in stop_codons_set:
                aa_position = ((i - start_pos) // 3) + 1
                positions.append(aa_position)
    return positions

def get_minus1_stop_positions(dna_seq):
    """Get positions of stop codons in -1 frame"""
    positions = []
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    start_pos, end_pos = find_coding_sequence_bounds(dna_seq_upper)
    if start_pos is None:
        return positions
     
    stop_codons_set = {"TAA", "TAG", "TGA"}
    minus1_start = start_pos + 2
    search_end = end_pos if end_pos is not None else len(dna_seq_upper) - 2
    
    for i in range(minus1_start, search_end, 3):
        if i+3 <= len(dna_seq_upper):
            codon = dna_seq_upper[i:i+3]
            if codon in stop_codons_set:
                aa_position = ((i - start_pos) // 3) + 1
                positions.append(aa_position)
    return positions

def balanced_optimisation(dna_seq, bias_weight_input=None):
    """Balanced optimization considering codon usage and +1 frame stops"""
    bias_weight = bias_weight_input if bias_weight_input is not None else st.session_state.config.get("bias_weight", BIAS_WEIGHT_DEFAULT)
    
    dna_seq_upper = dna_seq.upper()
    genetic_code = st.session_state.genetic_code
    aa_to_codons = st.session_state.aa_to_codons
    
    # Protein translation
    protein_str = ""
    for i in range(0, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        protein_str += genetic_code.get(codon, str(Seq(codon).translate()))
    
    optimised_seq = ""
    idx = 0
    while idx < len(dna_seq_upper) - 2:
        current_codon = dna_seq_upper[idx:idx+3]
        aa = genetic_code.get(current_codon, str(Seq(current_codon).translate()))

        if idx < len(dna_seq_upper) - 5:  # Check for two-codon substitutions
            next_codon_val = dna_seq_upper[idx+3:idx+6]
            aa2 = genetic_code.get(next_codon_val, str(Seq(next_codon_val).translate()))
            candidates = []
            
            if aa in aa_to_codons and aa2 in aa_to_codons:
                for c1, f1 in aa_to_codons[aa]:
                    for c2, f2 in aa_to_codons[aa2]:
                        combined = c1 + c2
                        codon1_plus1 = combined[1:4]
                        bonus = 0
                        if codon1_plus1 in PLUS1_STOP_CODONS and combined[2:5] in PLUS1_STOP_CODONS:
                            bonus += 2
                        elif codon1_plus1 in PLUS1_STOP_CODONS:
                            bonus += 1
                        
                        score = (f1 * f2) + bias_weight * bonus
                        candidates.append((score, c1, c2))
            
            if candidates:
                _, best1, best2 = max(candidates)
                optimised_seq += best1 + best2
                idx += 6
                continue
        
        # Single codon substitution
        best_codon_val = current_codon
        current_codon_freq = 0
        for syn_c, freq_val in aa_to_codons.get(aa, []):
            if syn_c == current_codon:
                current_codon_freq = freq_val
                break
        
        temp_seq_orig = optimised_seq + current_codon + dna_seq_upper[idx+3:]
        plus1_window_orig_start = len(optimised_seq) + 1
        bonus_orig = 0
        if plus1_window_orig_start < len(temp_seq_orig) - 2:
            codon_plus1_orig = temp_seq_orig[plus1_window_orig_start:plus1_window_orig_start+3]
            if codon_plus1_orig in PLUS1_STOP_CODONS:
                bonus_orig = bias_weight
        best_score = current_codon_freq + bonus_orig
        
        for syn_codon, freq in aa_to_codons.get(aa, []):
            temp_seq = optimised_seq + syn_codon + dna_seq_upper[idx+3:]
            plus1_codon_start_in_temp = len(optimised_seq) + 1
            
            bonus_val = 0
            if plus1_codon_start_in_temp < len(temp_seq) - 2:
                codon_plus1 = temp_seq[plus1_codon_start_in_temp:plus1_codon_start_in_temp+3]
                if codon_plus1 in PLUS1_STOP_CODONS:
                    bonus_val = bias_weight
            
            score = freq + bonus_val
            if score > best_score:
                best_score = score
                best_codon_val = syn_codon
        
        optimised_seq += best_codon_val
        idx += 3

    if idx < len(dna_seq_upper):
        optimised_seq += dna_seq_upper[idx:]
    
    # Verify protein sequence unchanged
    final_protein_str = ""
    for i in range(0, len(optimised_seq) - 2, 3):
        codon = optimised_seq[i:i+3]
        final_protein_str += genetic_code.get(codon, str(Seq(codon).translate()))

    if final_protein_str != protein_str:
        logger.error("Protein sequence changed in balanced optimization!")
        return dna_seq_upper
    
    return optimised_seq

def nc_stop_codon_optimisation(dna_seq):
    """NC stop codon optimization"""
    dna_seq_upper = dna_seq.upper()
    genetic_code = st.session_state.genetic_code
    
    protein_str = ""
    for i in range(0, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        protein_str += genetic_code.get(codon, str(Seq(codon).translate()))

    synonymous_codons_local = defaultdict(list)
    for c, aa_val in genetic_code.items():
        synonymous_codons_local[aa_val].append(c)
    
    optimised_seq = ""
    idx = 0
    while idx < len(dna_seq_upper) - 2:
        codon_val = dna_seq_upper[idx:idx+3]
        aa = genetic_code.get(codon_val, str(Seq(codon_val).translate()))

        if idx < len(dna_seq_upper) - 5:  # Try double substitution
            codon2 = dna_seq_upper[idx+3:idx+6]
            aa2 = genetic_code.get(codon2, str(Seq(codon2).translate()))
            if aa in synonymous_codons_local and aa2 in synonymous_codons_local:
                double_subs_orig_check = [(c1, c2) for c1 in synonymous_codons_local[aa] 
                                        for c2 in synonymous_codons_local[aa2] 
                                        if (c1 + c2)[1:7] in {"TAATAA", "TAGTAG"}]
                if double_subs_orig_check:
                    best_c1, best_c2 = double_subs_orig_check[0]
                    optimised_seq += best_c1 + best_c2
                    idx += 6
                    continue
        
        best_codon_val = codon_val
        # For single codon, check if any synonym creates TAA or TAG in +1 frame
        if idx + 3 < len(dna_seq_upper):
            next_actual_codon = dna_seq_upper[idx+3:idx+6]
            for syn_c in synonymous_codons_local.get(aa, []):
                plus1_codon = syn_c[1:3] + next_actual_codon[0:1]
                if plus1_codon in {"TAA", "TAG"}:
                    best_codon_val = syn_c
                    break
        optimised_seq += best_codon_val
        idx += 3

    # Verify protein sequence unchanged
    final_protein_str = ""
    for i in range(0, len(optimised_seq) - 2, 3):
        codon = optimised_seq[i:i+3]
        final_protein_str += genetic_code.get(codon, str(Seq(codon).translate()))

    if final_protein_str != protein_str:
        logger.error("Protein sequence changed in nc_stop_codon_optimisation!")
        return dna_seq_upper
    
    return optimised_seq

def third_aa_has_A_G_synonymous(aa):
    """Check if amino acid has synonymous codons starting with A or G"""
    for codon_val in synonymous_codons.get(aa, []):
        if codon_val.startswith(('A', 'G')):
            return True
    return False

def JT_Plus1_Stop_Optimized(seq_input):
    """JT Plus1 stop optimization"""
    seq = seq_input.upper()
    out_seq = ''
    idx = 0
    while idx <= len(seq) - 9:
        c1, c2, c3 = seq[idx:idx+3], seq[idx+3:idx+6], seq[idx+6:idx+9]
        aa1 = STANDARD_GENETIC_CODE.get(c1, '?')
        aa2 = STANDARD_GENETIC_CODE.get(c2, '?')
        aa3 = STANDARD_GENETIC_CODE.get(c3, '?')

        if (aa1 in FIRST_AA_CANDIDATES and aa2 in SECOND_AA_CANDIDATES and
            aa3 in synonymous_codons and third_aa_has_A_G_synonymous(aa3)):
            found_motif = False
            for syn1 in synonymous_codons.get(aa1, []):
                if not syn1.endswith('TA'):
                    continue
                for syn2 in synonymous_codons.get(aa2, []):
                    if not syn2.startswith(('A', 'G')):
                        continue
                    for syn3 in synonymous_codons.get(aa3, []):
                        if not syn3.startswith(('A', 'G')):
                            continue
                        motif_check = syn1[1:] + syn2 + syn3[:1]
                        if motif_check in PLUS1_STOP_MOTIFS:
                            out_seq += syn1 + syn2 + syn3
                            idx += 9
                            found_motif = True
                            break
                    if found_motif:
                        break
                if found_motif:
                    break
            if not found_motif:
                out_seq += c1
                idx += 3
        else:
            out_seq += c1
            idx += 3
    out_seq += seq[idx:]
    return out_seq

class PatentSearchEngine:
    def __init__(self):
        self.serper_api_key = os.getenv('SERPER_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API')
        self.anthropic = Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None
    
    def search_patents(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search Google Patents using SERPER API"""
        if not self.serper_api_key:
            st.error("SERPER API key is not configured. Please check your .env file.")
            return []
        
        url = "https://google.serper.dev/search"
        patent_query = f"site:patents.google.com {query}"
        
        payload = {"q": patent_query, "num": num_results}
        headers = {'X-API-KEY': self.serper_api_key, 'Content-Type': 'application/json'}
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 403:
                st.error("SERPER API Key is invalid or doesn't have permission")
                return []
            elif response.status_code != 200:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return []
            
            results = response.json().get('organic', [])
            return results
            
        except requests.exceptions.ConnectionError as e:
            st.error(f"Connection error: Unable to reach SERPER API. Please check your internet connection.")
            st.info("Troubleshooting tips:\n- Check your internet connection\n- Try again in a few moments\n- Verify your network allows HTTPS requests")
            return []
        except requests.exceptions.Timeout as e:
            st.error("Request timeout: SERPER API took too long to respond. Please try again.")
            return []
        except requests.exceptions.RequestException as e:
            st.error(f"Network error searching patents: {str(e)}")
            st.info("This may be a temporary network issue. Please try again in a few moments.")
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
    
    def extract_mrna_sequences_from_patents(self, patents: List[Dict]) -> List[Dict]:
        """Extract potential mRNA/nucleotide sequences from patent information"""
        sequence_findings = []
        
        for patent in patents:
            findings = {
                'patent_id': patent['patent_id'],
                'title': patent['title'],
                'link': patent['link'],
                'sequences_found': [],
                'mrna_indicators': [],
                'confidence_score': 0
            }
            
            # Combine title and snippet for analysis
            text_content = f"{patent['title']} {patent['snippet']}"
            
            # Look for ACCATG sequences (5' UTR + start codon)
            accatg_matches = re.finditer(r'ACCATG[ATGC]{20,}', text_content.upper())
            for match in accatg_matches:
                seq = match.group()
                findings['sequences_found'].append({
                    'type': 'ACCATG_sequence',
                    'sequence': seq,
                    'length': len(seq),
                    'confidence': 'high'
                })
                findings['confidence_score'] += 3
            
            # Look for other long nucleotide sequences
            long_seq_pattern = r'[ATGCUN]{30,}'
            long_sequences = re.finditer(long_seq_pattern, text_content.upper())
            for match in long_sequences:
                seq = match.group().replace('U', 'T')  # Convert RNA to DNA notation
                if seq not in [s['sequence'] for s in findings['sequences_found']]:  # Avoid duplicates
                    findings['sequences_found'].append({
                        'type': 'long_nucleotide',
                        'sequence': seq,
                        'length': len(seq),
                        'confidence': 'medium'
                    })
                    findings['confidence_score'] += 2
            
            # Look for mRNA-related keywords
            mrna_keywords = [
                'mRNA', 'messenger RNA', 'nucleotide sequence', 'coding sequence',
                'CDS', 'open reading frame', 'ORF', 'start codon', 'stop codon',
                '5\' UTR', '3\' UTR', 'poly(A)', 'cap structure', 'codon optimization',
                'translation', 'ribosome', 'protein expression', 'ACCATG'
            ]
            
            for keyword in mrna_keywords:
                if keyword.lower() in text_content.lower():
                    findings['mrna_indicators'].append(keyword)
                    findings['confidence_score'] += 1
            
            # Only include patents with actual sequence findings or strong mRNA indicators
            if findings['sequences_found'] or len(findings['mrna_indicators']) >= 3:
                sequence_findings.append(findings)
        
        # Sort by confidence score (highest first)
        sequence_findings.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return sequence_findings
    
    def generate_ai_analysis(self, query: str, context: str) -> str:
        """Generate AI analysis using Anthropic"""
        if not self.anthropic:
            return "Anthropic API is not configured. Please check your .env file."
        
        prompt = f"""
You are a bioinformatics and patent research assistant specializing in DNA, RNA, and molecular biology technologies.

A user has asked: "{query}"

Based on the following patent search results, provide a comprehensive response:

{context}

Please:
1. Provide a clear answer focused on the molecular biology aspects
2. Reference specific patents with IDs and titles
3. Explain key DNA/RNA technologies and innovations
4. Highlight any codon optimization, sequence analysis, or related molecular techniques
5. Include relevant patent links
6. Compare different molecular approaches if applicable

Focus particularly on any DNA sequences, codon usage, protein expression, or related biotechnology innovations.
"""

        try:
            message = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error generating AI response: {e}"

    def generate_sequence_analysis(self, query: str, sequence_findings: List[Dict]) -> str:
        """Generate AI analysis specifically for sequence findings"""
        if not self.anthropic or not sequence_findings:
            return "No sequence analysis available."
        
        # Prepare context about sequence findings
        context = ""
        for finding in sequence_findings[:5]:  # Top 5 most relevant
            context += f"""
Patent: {finding['title']}
ID: {finding['patent_id']}
Confidence Score: {finding['confidence_score']}
mRNA Indicators: {', '.join(finding['mrna_indicators'])}
Sequences Found: {len(finding['sequences_found'])}
"""
            for seq in finding['sequences_found']:
                context += f"- {seq['type']}: {seq['sequence'][:50]}{'...' if len(seq['sequence']) > 50 else ''} ({seq['length']} bp, {seq['confidence']} confidence)\n"
            context += f"Link: {finding['link']}\n\n"
        
        prompt = f"""
You are a molecular biology expert analyzing patent-derived nucleotide sequences related to the query: "{query}"

Here are the sequence findings from patents:

{context}

Please analyze these findings and provide:

1. **Most Promising Sequences**: Identify which sequences are most likely to be relevant to the user's search
2. **Sequence Characteristics**: Analyze the structure and features of the found sequences
3. **mRNA/Coding Potential**: Assess which sequences might be functional mRNA or coding sequences
4. **Patent Relevance**: Explain how these sequences relate to the patents' innovations
5. **Recommendations**: Suggest which sequences the user should focus on for their research

Focus on practical insights about sequence utility, coding potential, and research applications.
"""

        try:
            message = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error generating sequence analysis: {e}"


class NCBISearchEngine:
    def __init__(self):
        self.serper_api_key = os.getenv('SERPER_API_KEY')
        self.base_url = "https://www.ncbi.nlm.nih.gov"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.anthropic_api_key = os.getenv('ANTHROPIC_API')
        self.anthropic = Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None
    
    def search_nucleotide_sequences(self, query: str, max_results: int = 10, quoted_terms: List[str] = None) -> List[Dict]:
        """Search NCBI nucleotide database using Google search with two-pass filtering"""
        if not self.serper_api_key:
            st.error("SERPER API key is required for NCBI search. Please check your .env file.")
            return []
        
        # Two-pass approach
        # Pass 1: Search using the entire query to find relevant candidates
        st.write(f"🔍 **Pass 1:** Searching for candidates matching entire query")
            
        try:
            url = "https://google.serper.dev/search"
            ncbi_query = f"site:ncbi.nlm.nih.gov/nuccore {query}"
            
            payload = {"q": ncbi_query, "num": max_results * 3}  # Get more candidates for filtering
            headers = {'X-API-KEY': self.serper_api_key, 'Content-Type': 'application/json'}
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 403:
                st.error("SERPER API Key is invalid or doesn't have permission")
                return []
            elif response.status_code != 200:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return []
            
            search_results = response.json().get('organic', [])
            candidates = []
            
            for result in search_results:
                try:
                    title = result.get('title', '')
                    link = result.get('link', '')
                    snippet = result.get('snippet', '')
                    
                    accession = self.extract_accession_number(link, title)
                    clean_title = title.replace(' - Nucleotide - NCBI', '').replace(' - NCBI', '').strip()
                    
                    candidates.append({
                        'title': clean_title,
                        'accession': accession,
                        'description': snippet,
                        'link': link
                    })
                except Exception as e:
                    continue
            
            st.write(f"✅ **Pass 1 complete:** Found {len(candidates)} candidate sequences")
            
            # Pass 2: If quoted terms exist, filter candidates by relevance to the full query
            if quoted_terms:
                st.write(f"🎯 **Pass 2:** Filtering candidates for relevance (before CDS analysis)")
                filtered_candidates = []
                
                for candidate in candidates:
                    # Check if the candidate is relevant to the overall query context
                    searchable_text = f"{candidate['title']} {candidate['description']}".lower()
                    
                    # Simple relevance scoring - must contain some key terms from the query
                    query_words = [word.strip().lower() for word in query.replace('"', '').split() if len(word.strip()) > 2]
                    
                    # Count how many query words appear in the candidate
                    matches = sum(1 for word in query_words if word in searchable_text)
                    relevance_score = matches / len(query_words) if query_words else 0
                    
                    # Keep candidates with reasonable relevance (at least 30% of query words)
                    if relevance_score >= 0.3:
                        candidate['relevance_score'] = relevance_score
                        filtered_candidates.append(candidate)
                
                # Sort by relevance score
                filtered_candidates.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                
                # Take top candidates up to max_results
                results = filtered_candidates[:max_results]
                
                st.write(f"✅ **Pass 2 complete:** {len(results)} relevant candidates selected for CDS analysis")
                
                if len(results) < len(candidates):
                    st.write(f"📊 **Filtered out:** {len(candidates) - len(results)} less relevant candidates")
            else:
                # No quoted terms, just take top results
                results = candidates[:max_results]
                st.write(f"ℹ️ **No quoted terms:** Taking top {len(results)} candidates")
            
            return results
            
        except Exception as e:
            st.error(f"Error searching NCBI: {str(e)}")
            return []
    
    def extract_accession_number(self, link: str, title: str) -> str:
        """Extract accession number from NCBI link or title"""
        try:
            if '/nuccore/' in link:
                parts = link.split('/nuccore/')
                if len(parts) > 1:
                    accession = parts[1].split('/')[0].split('?')[0]
                    return accession
            
            patterns = [
                r'\b([A-Z]{1,2}\d{5,8})\b',
                r'\b([A-Z]{2}_\d{6,9})\b',
                r'\b([A-Z]{3}\d{5})\b',
                r'\b([A-Z]{1}\d{5})\b',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, title)
                if match:
                    return match.group(1)
            
            return ""
        except:
            return ""

    def scrape_ncbi_page(self, accession: str, original_query: str = "") -> Dict:
        """Get NCBI data using structured formats with quote-based filtering"""
        try:
            result = {
                'accession': accession,
                'url': f"{self.base_url}/nuccore/{accession}",
                'success': True,
                'cds_sequences': [],
                'organism': '',
                'definition': '',
                'length': 0,
                'filtered_terms': []
            }
            
            st.write(f"🔍 **Getting structured data for {accession}**")
            
            # Extract quoted terms for filtering
            quoted_terms = self.extract_quoted_terms(original_query)
            if quoted_terms:
                st.write(f"🎯 **Filtering for quoted terms:** {', '.join(quoted_terms)}")
                result['filtered_terms'] = quoted_terms
            
            # Step 1: Get data (GenBank or FASTA format)
            raw_data = self.get_genbank_format(accession)
            
            if raw_data:
                st.write(f"✅ **Data retrieved:** {len(raw_data)} characters")
                
                # Check if we got FASTA format instead of GenBank
                if raw_data.startswith('FASTA_FORMAT\n'):
                    st.write(f"📄 **Processing FASTA format data**")
                    fasta_content = raw_data[13:]  # Remove "FASTA_FORMAT\n" prefix
                    result = self.process_fasta_data(fasta_content, accession)
                    
                    # Apply filtering to FASTA data if needed
                    if quoted_terms and result.get('cds_sequences'):
                        original_count = len(result['cds_sequences'])
                        filtered_cds = []
                        
                        for cds in result['cds_sequences']:
                            # For FASTA, check if the header contains the quoted terms
                            header = cds.get('header', '').lower()
                            definition = result.get('definition', '').lower()
                            
                            matches = any(term in header or term in definition for term in quoted_terms)
                            if matches:
                                filtered_cds.append(cds)
                        
                        result['cds_sequences'] = filtered_cds
                        st.write(f"🎯 **Filtered FASTA results:** {original_count} → {len(filtered_cds)} sequences")
                    
                else:
                    st.write(f"📄 **Processing GenBank format data**")
                    
                    # Extract metadata
                    metadata = self.parse_genbank_metadata(raw_data)
                    result.update(metadata)
                    
                    # Extract ORIGIN sequence
                    origin_sequence = self.extract_origin_from_genbank(raw_data)
                    
                    if origin_sequence:
                        st.write(f"✅ **ORIGIN sequence:** {len(origin_sequence)} bases")
                        st.write(f"**Sample:** {origin_sequence[:50]}...")
                        
                        # Extract CDS features with filtering
                        cds_features = self.parse_cds_features_from_genbank(raw_data, origin_sequence, accession, quoted_terms)
                        
                        # Strict filtering: if quoted terms provided and NO CDS match, reject the entire entry
                        if quoted_terms and not cds_features:
                            st.write(f"❌ **REJECTED:** No CDS matching '{', '.join(quoted_terms)}' - skipping entry**")
                            return {
                                'accession': accession,
                                'success': False,
                                'error': f"No CDS found matching quoted terms: {', '.join(quoted_terms)}",
                                'cds_sequences': [],
                                'filtered_terms': quoted_terms
                            }
                        
                        result['cds_sequences'] = cds_features
                        
                        if cds_features:
                            st.write(f"✅ **Found {len(cds_features)} matching CDS features**")
                            for i, cds in enumerate(cds_features):
                                st.write(f"  - {cds['protein_name']} ({cds['start_position']}-{cds['end_position']}, {cds['length']} bp)")
                        else:
                            st.write("❌ **No CDS features found in GenBank data**")
                    else:
                        st.write("❌ **No ORIGIN sequence found - trying FASTA fallback**")
                        result = self.fallback_fasta_approach(accession)
            else:
                st.write("❌ **Failed to retrieve any data**")
                result = self.fallback_fasta_approach(accession)
            
            return result
            
        except Exception as e:
            st.error(f"Error processing {accession}: {str(e)}")
            return {
                'accession': accession,
                'success': False,
                'error': str(e),
                'cds_sequences': []
            }
        
        
    def extract_quoted_terms(self, query: str) -> List[str]:
        """Extract terms in quotes from the search query"""
        try:
            # Find all terms in double quotes
            quoted_terms = re.findall(r'"([^"]+)"', query)
            
            # Also look for single quotes as backup
            if not quoted_terms:
                quoted_terms = re.findall(r"'([^']+)'", query)
            
            # Clean and normalize terms
            cleaned_terms = []
            for term in quoted_terms:
                cleaned_term = term.strip().lower()
                if cleaned_term:
                    cleaned_terms.append(cleaned_term)
            
            return cleaned_terms
            
        except Exception as e:
            st.write(f"❌ **Error extracting quoted terms:** {str(e)}")
            return []

    def matches_quoted_terms(self, cds_info: Dict, quoted_terms: List[str]) -> bool:
        """Check if a CDS matches any of the quoted terms"""
        if not quoted_terms:
            return True  # No filtering if no quoted terms
        
        try:
            # Fields to check for matches (expanded to include more annotation fields)
            searchable_fields = [
                cds_info.get('gene_name', '').lower(),
                cds_info.get('product', '').lower(),
                cds_info.get('protein_name', '').lower(),
                cds_info.get('locus_tag', '').lower(),
                cds_info.get('note', '').lower(),  # Added note field
                cds_info.get('gene_synonym', '').lower(),  # Added gene synonym
                cds_info.get('function', '').lower()  # Added function field
            ]
            
            # Check if any quoted term matches any field
            for term in quoted_terms:
                for field in searchable_fields:
                    if field and term in field:
                        return True
            
            return False
            
        except Exception as e:
            st.write(f"❌ **Error checking CDS match:** {str(e)}")
            return False  # Changed to False - if error, don't match

    def process_fasta_data(self, fasta_content: str, accession: str) -> Dict:
        """Process FASTA format data"""
        try:
            lines = fasta_content.strip().split('\n')
            if not lines or not lines[0].startswith('>'):
                raise ValueError("Invalid FASTA format")
            
            header = lines[0][1:]  # Remove '>'
            sequence = ''.join(lines[1:]).upper()
            
            # Clean sequence
            sequence = re.sub(r'[^ATGCN]', '', sequence)
            
            if sequence and len(sequence) > 100:  # Reasonable sequence length
                return {
                    'accession': accession,
                    'url': f"{self.base_url}/nuccore/{accession}",
                    'success': True,
                    'cds_sequences': [{
                        'accession': accession,
                        'protein_name': f"Complete_sequence_{accession}",
                        'gene_name': '',
                        'product': 'Complete nucleotide sequence',
                        'locus_tag': '',
                        'start_position': 1,
                        'end_position': len(sequence),
                        'header': f">{header}",
                        'sequence': sequence,
                        'length': len(sequence),
                        'url': f"{self.base_url}/nuccore/{accession}",
                        'valid_dna': True
                    }],
                    'organism': '',
                    'definition': header,
                    'length': len(sequence)
                }
            else:
                raise ValueError("Invalid or too short sequence")
                
        except Exception as e:
            return {
                'accession': accession,
                'success': False,
                'error': str(e),
                'cds_sequences': []
            }
    
    def get_genbank_format(self, accession: str) -> str:
        """Get GenBank format data directly using correct URLs"""
        try:
            # Use the correct NCBI E-utilities API endpoints that return raw text
            genbank_urls = [
                # E-utilities API - most reliable
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=gb&retmode=text",
                
                # Alternative E-utilities format
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&id={accession}&rettype=genbank&retmode=text",
                
                # NCBI sviewer with explicit text format
                f"https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi?tool=portal&sendto=on&log$=seqview&db=nuccore&dopt=genbank&sort=&val={accession}&retmode=text",
                
                # Direct nuccore with specific parameters
                f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}?report=genbank&format=text&retmode=text",
            ]
            
            for url in genbank_urls:
                try:
                    time.sleep(1)
                    response = self.session.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        content = response.text
                        
                        # Check if this is actually GenBank format (not HTML)
                        if content.startswith('<?xml') or content.startswith('<!DOCTYPE'):
                            continue
                        
                        # Check if this looks like GenBank format
                        if content.startswith('LOCUS') and ('ORIGIN' in content or 'FEATURES' in content):
                            return content
                        elif 'LOCUS' in content and 'DEFINITION' in content:
                            return content
                        
                except Exception as e:
                    continue
            
            # If all GenBank URLs fail, try getting FASTA directly here
            fasta_urls = [
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text",
                f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}?report=fasta&format=text",
            ]
            
            for url in fasta_urls:
                try:
                    time.sleep(1)
                    response = self.session.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        content = response.text
                        if content.startswith('>') and not content.startswith('<?xml'):
                            # Return a special marker so we know this is FASTA
                            return f"FASTA_FORMAT\n{content}"
                        
                except Exception as e:
                    continue
            
            return ""
            
        except Exception as e:
            return ""
        
    def parse_genbank_metadata(self, genbank_data: str) -> Dict:
        """Parse metadata from GenBank format"""
        metadata = {}
        
        try:
            # Extract definition
            def_match = re.search(r'DEFINITION\s+(.*?)(?=\nACCESSION|\nVERSION|\nKEYWORDS)', genbank_data, re.DOTALL)
            if def_match:
                metadata['definition'] = re.sub(r'\s+', ' ', def_match.group(1).strip())
            
            # Extract organism
            organism_match = re.search(r'ORGANISM\s+(.*?)(?=\n\s*REFERENCE|\n\s*COMMENT|\n\s*FEATURES)', genbank_data, re.DOTALL)
            if organism_match:
                organism_text = organism_match.group(1).strip()
                # Get just the first line (species name)
                metadata['organism'] = organism_text.split('\n')[0].strip()
            
            # Extract length from LOCUS line
            locus_match = re.search(r'LOCUS\s+\S+\s+(\d+)\s+bp', genbank_data)
            if locus_match:
                metadata['length'] = int(locus_match.group(1))
            
            return metadata
            
        except Exception as e:
            st.write(f"❌ **Error parsing GenBank metadata:** {str(e)}")
            return {}
    
    def extract_origin_from_genbank(self, genbank_data: str) -> str:
        """Extract ORIGIN sequence from GenBank format"""
        try:
            # Try multiple ORIGIN patterns
            origin_patterns = [
                r'ORIGIN\s*(.*?)(?=//)',                    # Original pattern
                r'ORIGIN\s*(.*?)(?=\n//)',                  # With newline before //
                r'ORIGIN\s*(.*?)$',                         # Until end of string
                r'ORIGIN\s*\n(.*?)(?=//)',                  # With explicit newline after ORIGIN
                r'ORIGIN\s*\n(.*?)(?=\n//)',               # With newlines
                r'ORIGIN[^\n]*\n(.*?)(?=//)',              # Skip ORIGIN line, start from next line
            ]
            
            origin_text = None
            
            for pattern in origin_patterns:
                try:
                    match = re.search(pattern, genbank_data, re.DOTALL)
                    if match:
                        origin_text = match.group(1)
                        break
                except Exception:
                    continue
            
            if not origin_text:
                # Try a simple substring approach
                origin_pos = genbank_data.find('ORIGIN')
                if origin_pos != -1:
                    origin_section = genbank_data[origin_pos:]
                    end_pos = origin_section.find('//')
                    if end_pos != -1:
                        origin_text = origin_section[6:end_pos]  # Skip "ORIGIN"
                else:
                    return ""
            
            if not origin_text:
                return ""
            
            # Clean the sequence
            clean_sequence = ""
            lines = origin_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip lines that don't look like sequence lines
                if not re.search(r'\d+', line):
                    continue
                    
                # Remove line numbers (first token) and keep DNA bases
                parts = line.split()
                if parts:  # Make sure there are parts
                    for part in parts[1:]:  # Skip first part (line number)
                        # Keep only DNA bases
                        dna_bases = re.sub(r'[^ATGCatgcNn]', '', part)
                        clean_sequence += dna_bases.upper()
            
            return clean_sequence
            
        except Exception as e:
            return ""
        



    def parse_cds_features_from_genbank(self, genbank_data: str, origin_sequence: str, accession: str, quoted_terms: List[str] = None) -> List[Dict]:
        """Parse CDS features from GenBank format with optional filtering"""
        cds_sequences = []
        
        try:
            # Find FEATURES section
            features_match = re.search(r'FEATURES\s+Location/Qualifiers\s*(.*?)(?=ORIGIN|CONTIG|//)', genbank_data, re.DOTALL)
            
            if not features_match:
                st.write("❌ **No FEATURES section found**")
                return []
            
            features_text = features_match.group(1)
            st.write(f"✅ **Found FEATURES section:** {len(features_text)} characters")
            
            # Find CDS features
            cds_pattern = r'^\s+CDS\s+(.*?)(?=^\s+\w+\s+|\Z)'
            cds_matches = re.finditer(cds_pattern, features_text, re.MULTILINE | re.DOTALL)
            
            total_cds_found = 0
            filtered_cds_count = 0
            
            for i, match in enumerate(cds_matches):
                try:
                    total_cds_found += 1
                    cds_block = match.group(1)
                    
                    # Extract location - handle simple ranges and joins
                    location_line = cds_block.split('\n')[0].strip()
                    
                    # Parse coordinates
                    coordinates = self.parse_cds_coordinates(location_line)
                    
                    if coordinates:
                        start_pos, end_pos = coordinates[0], coordinates[-1]
                        
                        # Extract sequence
                        if len(origin_sequence) >= end_pos:
                            # For simple ranges, extract directly
                            if len(coordinates) == 2:
                                cds_sequence = origin_sequence[start_pos-1:end_pos]
                            else:
                                # For complex joins, concatenate segments
                                cds_sequence = ""
                                for j in range(0, len(coordinates), 2):
                                    if j+1 < len(coordinates):
                                        seg_start, seg_end = coordinates[j], coordinates[j+1]
                                        cds_sequence += origin_sequence[seg_start-1:seg_end]
                            
                            if cds_sequence:
                                # Extract gene information
                                gene_info = self.extract_gene_info_from_cds_block(cds_block)
                                
                                cds_info = {
                                    'accession': accession,
                                    'protein_name': gene_info.get('protein_name', f"CDS_{i+1}"),
                                    'gene_name': gene_info.get('gene_name', ''),
                                    'product': gene_info.get('product', ''),
                                    'locus_tag': gene_info.get('locus_tag', ''),
                                    'start_position': start_pos,
                                    'end_position': end_pos,
                                    'header': f">{accession}:{start_pos}-{end_pos} {gene_info.get('protein_name', f'CDS_{i+1}')}",
                                    'sequence': cds_sequence,
                                    'length': len(cds_sequence),
                                    'url': f"{self.base_url}/nuccore/{accession}",
                                    'valid_dna': self.is_valid_dna_sequence(cds_sequence)
                                }
                                
                                # Apply filtering if quoted terms are provided
                                if quoted_terms:
                                    if self.matches_quoted_terms(cds_info, quoted_terms):
                                        cds_sequences.append(cds_info)
                                        filtered_cds_count += 1
                                        st.write(f"  ✅ **Matched:** {gene_info.get('protein_name', f'CDS_{i+1}')} ({len(cds_sequence)} bp)")
                                    else:
                                        st.write(f"  ⏭️ **Skipped:** {gene_info.get('protein_name', f'CDS_{i+1}')} (no match)")
                                else:
                                    # No filtering, add all CDS
                                    cds_sequences.append(cds_info)
                                    st.write(f"  ✅ **Added:** {gene_info.get('protein_name', f'CDS_{i+1}')} ({len(cds_sequence)} bp)")
                                    
                except Exception as e:
                    st.write(f"  ❌ **Error processing CDS {i+1}:** {str(e)}")
                    continue
            
            # Summary
            if quoted_terms:
                st.write(f"🎯 **Filtering summary:** {total_cds_found} total CDS → {filtered_cds_count} matching '{', '.join(quoted_terms)}'**")
            else:
                st.write(f"📊 **Total CDS found:** {len(cds_sequences)}")
            
            return cds_sequences
            
        except Exception as e:
            st.write(f"❌ **Error parsing CDS features:** {str(e)}")
            return []
        
    def parse_cds_coordinates(self, location_str: str) -> List[int]:
        """Parse CDS coordinates from location string"""
        try:
            coordinates = []
            
            # Handle simple range: "266..13483"
            simple_match = re.match(r'(\d+)\.\.(\d+)', location_str)
            if simple_match:
                start = int(simple_match.group(1))
                end = int(simple_match.group(2))
                return [start, end]
            
            # Handle join: "join(266..13483,13484..21555)"
            join_match = re.search(r'join\((.*?)\)', location_str)
            if join_match:
                segments = join_match.group(1).split(',')
                for segment in segments:
                    segment = segment.strip()
                    range_match = re.match(r'(\d+)\.\.(\d+)', segment)
                    if range_match:
                        coordinates.extend([int(range_match.group(1)), int(range_match.group(2))])
                return coordinates
            
            # Handle complement: "complement(266..13483)"
            comp_match = re.search(r'complement\((\d+)\.\.(\d+)\)', location_str)
            if comp_match:
                start = int(comp_match.group(1))
                end = int(comp_match.group(2))
                return [start, end]
            
            return []
            
        except Exception as e:
            return []
    
    def extract_gene_info_from_cds_block(self, cds_block: str) -> Dict:
        """Extract gene information from CDS feature block with enhanced fields"""
        gene_info = {
            'protein_name': '',
            'gene_name': '',
            'product': '',
            'locus_tag': '',
            'note': '',
            'gene_synonym': '',
            'function': ''
        }
        
        try:
            # Extract gene name
            gene_match = re.search(r'/gene="([^"]+)"', cds_block)
            if gene_match:
                gene_info['gene_name'] = gene_match.group(1)
                gene_info['protein_name'] = gene_match.group(1)
            
            # Extract product (preferred for protein name)
            product_match = re.search(r'/product="([^"]+)"', cds_block)
            if product_match:
                gene_info['product'] = product_match.group(1)
                gene_info['protein_name'] = product_match.group(1)
            
            # Extract locus tag
            locus_match = re.search(r'/locus_tag="([^"]+)"', cds_block)
            if locus_match:
                gene_info['locus_tag'] = locus_match.group(1)
            
            # Extract note field (often contains descriptive information)
            note_match = re.search(r'/note="([^"]+)"', cds_block)
            if note_match:
                gene_info['note'] = note_match.group(1)
            
            # Extract gene synonym
            synonym_match = re.search(r'/gene_synonym="([^"]+)"', cds_block)
            if synonym_match:
                gene_info['gene_synonym'] = synonym_match.group(1)
            
            # Extract function (if present)
            function_match = re.search(r'/function="([^"]+)"', cds_block)
            if function_match:
                gene_info['function'] = function_match.group(1)
            
            # If no product or gene, try protein_id
            if not gene_info['protein_name']:
                protein_id_match = re.search(r'/protein_id="([^"]+)"', cds_block)
                if protein_id_match:
                    gene_info['protein_name'] = protein_id_match.group(1)
            
            return gene_info
            
        except Exception as e:
            return gene_info
    
    def fallback_fasta_approach(self, accession: str) -> Dict:
        """Fallback approach using FASTA format"""
        try:
            # Try to get FASTA format
            fasta_urls = [
                f"{self.base_url}/nuccore/{accession}?report=fasta&format=text",
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text"
            ]
            
            for url in fasta_urls:
                try:
                    time.sleep(1)
                    response = self.session.get(url, timeout=30)
                    
                    if response.status_code == 200 and response.text.startswith('>'):
                        content = response.text
                        lines = content.split('\n')
                        header = lines[0]
                        sequence = ''.join(lines[1:])
                        
                        if sequence and self.is_valid_dna_sequence(sequence):
                            return {
                                'accession': accession,
                                'url': f"{self.base_url}/nuccore/{accession}",
                                'success': True,
                                'cds_sequences': [{
                                    'accession': accession,
                                    'protein_name': f"Full_sequence_{accession}",
                                    'gene_name': '',
                                    'product': 'Complete sequence',
                                    'locus_tag': '',
                                    'start_position': 1,
                                    'end_position': len(sequence),
                                    'header': header,
                                    'sequence': sequence,
                                    'length': len(sequence),
                                    'url': f"{self.base_url}/nuccore/{accession}",
                                    'valid_dna': True
                                }],
                                'organism': '',
                                'definition': header,
                                'length': len(sequence)
                            }
                            
                except Exception as e:
                    continue
            
            return {
                'accession': accession,
                'success': False,
                'error': 'Both GenBank and FASTA approaches failed',
                'cds_sequences': []
            }
            
        except Exception as e:
            return {
                'accession': accession,
                'success': False,
                'error': str(e),
                'cds_sequences': []
            }
    
    def is_valid_dna_sequence(self, sequence: str) -> bool:
        """Check if sequence contains only valid DNA bases"""
        if not sequence:
            return False
        return all(base.upper() in 'ATGCN' for base in sequence)
    
    # Keep the existing AI ranking and download methods unchanged
    def ai_select_best_sequences(self, query: str, sequences_with_cds: List[Dict]) -> List[Dict]:
        """Use AI to select the most relevant sequences based on the query"""
        if not self.anthropic or not sequences_with_cds:
            return sequences_with_cds
        
        sequence_summaries = []
        for i, seq_data in enumerate(sequences_with_cds):
            cds_info = []
            for cds in seq_data.get('cds_sequences', []):
                cds_info.append(f"  - {cds.get('protein_name', 'Unknown')} ({cds.get('start_position', 0)}-{cds.get('end_position', 0)}, {cds.get('length', 0)} bp)")
            
            summary = f"""
Sequence {i+1}:
- Accession: {seq_data['accession']}
- Title: {seq_data.get('title', 'N/A')}
- Definition: {seq_data.get('definition', 'N/A')}
- Organism: {seq_data.get('organism', 'N/A')}
- Length: {seq_data.get('length', 0)} bp
- CDS Count: {len(seq_data.get('cds_sequences', []))}
- CDS Details:
{chr(10).join(cds_info) if cds_info else '  - No CDS found'}
"""
            sequence_summaries.append(summary)
        
        context = "\n".join(sequence_summaries)
        
        prompt = f"""
You are a bioinformatics expert. A user is searching for: "{query}"

Below are the available sequences with their CDS information:

{context}

Please analyze these sequences and rank them by relevance to the user's query. Consider:
1. How well the organism/title matches the query
2. The presence and quality of CDS sequences
3. The biological relevance to the query
4. The completeness of the sequence data

Return your response as a JSON list of accession numbers in order of relevance (most relevant first), with a brief explanation for each.

Format: {{"rankings": [{{"accession": "XXX", "rank": 1, "reason": "explanation"}}, ...]}}
"""

        try:
            message = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            
            import json
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                rankings_data = json.loads(json_match.group())
                rankings = rankings_data.get('rankings', [])
                
                ordered_sequences = []
                for ranking in rankings:
                    accession = ranking['accession']
                    for seq_data in sequences_with_cds:
                        if seq_data['accession'] == accession:
                            seq_data['ai_rank'] = ranking['rank']
                            seq_data['ai_reason'] = ranking['reason']
                            ordered_sequences.append(seq_data)
                            break
                
                ranked_accessions = [r['accession'] for r in rankings]
                for seq_data in sequences_with_cds:
                    if seq_data['accession'] not in ranked_accessions:
                        ordered_sequences.append(seq_data)
                
                return ordered_sequences
            
        except Exception as e:
            logger.error(f"Error in AI ranking: {e}")
        
        return sequences_with_cds
    
    def create_cds_download_data(self, sequences_with_cds: List[Dict]) -> pd.DataFrame:
        """Create downloadable DataFrame with CDS sequences"""
        download_data = []
        
        for seq_data in sequences_with_cds:
            base_info = {
                'Accession': seq_data.get('accession', ''),
                'Title': seq_data.get('title', ''),
                'Definition': seq_data.get('definition', ''),
                'Organism': seq_data.get('organism', ''),
                'Sequence_Length_bp': seq_data.get('length', 0),
                'NCBI_URL': seq_data.get('url', ''),
                'Filtered_Terms': ', '.join(seq_data.get('filtered_terms', [])),
                'AI_Rank': seq_data.get('ai_rank', ''),
                'AI_Reason': seq_data.get('ai_reason', '')
            }
            
            if seq_data.get('cds_sequences'):
                for i, cds in enumerate(seq_data['cds_sequences']):
                    row = base_info.copy()
                    row.update({
                        'CDS_Number': i + 1,
                        'Gene_Name': cds.get('gene_name', ''),
                        'Protein_Name': cds.get('protein_name', ''),
                        'Product': cds.get('product', ''),
                        'Locus_Tag': cds.get('locus_tag', ''),
                        'Start_Position': cds.get('start_position', 0),
                        'End_Position': cds.get('end_position', 0),
                        'CDS_Header': cds.get('header', ''),
                        'CDS_Sequence': cds.get('sequence', ''),
                        'CDS_Length_bp': cds.get('length', 0),
                        'Valid_DNA': cds.get('valid_dna', False),
                        'CDS_URL': cds.get('url', '')
                    })
                    download_data.append(row)
            else:
                row = base_info.copy()
                row.update({
                    'CDS_Number': 0,
                    'Gene_Name': 'No matching CDS found',
                    'Protein_Name': 'No matching CDS found',
                    'Product': '',
                    'Locus_Tag': '',
                    'Start_Position': 0,
                    'End_Position': 0,
                    'CDS_Header': '',
                    'CDS_Sequence': '',
                    'CDS_Length_bp': 0,
                    'Valid_DNA': False,
                    'CDS_URL': ''
                })
                download_data.append(row)
        
        return pd.DataFrame(download_data)


class UniProtSearchEngine:
    """Enhanced UniProt search engine for protein sequences with CDS data"""
    
    def __init__(self):
        self.base_url = "https://www.uniprot.org"
        self.api_url = "https://rest.uniprot.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.anthropic_api_key = os.getenv('ANTHROPIC_API')
        self.anthropic = Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None
    
    def search_protein_sequences(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search UniProt for protein sequences using REST API"""
        try:
            # Prepare search parameters
            params = {
                'query': query,
                'format': 'json',
                'size': max_results,
                'fields': 'accession,id,protein_name,gene_names,organism_name,length,reviewed,xref_refseq,xref_embl,sequence'
            }
            
            url = f"{self.api_url}/uniprotkb/search"
            
            st.write(f"🔍 **Searching UniProt for:** {query}")
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                st.error(f"UniProt API error: {response.status_code}")
                return []
            
            data = response.json()
            results = data.get('results', [])
            
            st.write(f"✅ **Found {len(results)} UniProt entries**")
            
            # Process results
            processed_results = []
            for result in results:
                try:
                    # Extract basic information
                    accession = result.get('primaryAccession', '')
                    protein_name = result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
                    if not protein_name:
                        protein_name = result.get('proteinDescription', {}).get('submissionNames', [{}])[0].get('fullName', {}).get('value', '')
                    
                    gene_names = []
                    if 'genes' in result:
                        for gene in result['genes']:
                            if 'geneName' in gene:
                                gene_names.append(gene['geneName']['value'])
                    
                    organism = result.get('organism', {}).get('scientificName', '')
                    sequence = result.get('sequence', {}).get('value', '')
                    length = result.get('sequence', {}).get('length', 0)
                    reviewed = result.get('entryType', '') == 'UniProtKB reviewed (Swiss-Prot)'
                    
                    # Extract cross-references to nucleotide databases
                    nucleotide_refs = []
                    if 'uniProtKBCrossReferences' in result:
                        for xref in result['uniProtKBCrossReferences']:
                            db_name = xref.get('database', '')
                            if db_name in ['EMBL', 'RefSeq']:
                                nucleotide_refs.append({
                                    'database': db_name,
                                    'id': xref.get('id', ''),
                                    'properties': xref.get('properties', [])
                                })
                    
                    processed_results.append({
                        'accession': accession,
                        'protein_name': protein_name,
                        'gene_names': ', '.join(gene_names),
                        'organism': organism,
                        'protein_sequence': sequence,
                        'length': length,
                        'reviewed': reviewed,
                        'nucleotide_refs': nucleotide_refs,
                        'uniprot_url': f"{self.base_url}/uniprotkb/{accession}"
                    })
                    
                except Exception as e:
                    st.write(f"❌ Error processing UniProt entry: {str(e)}")
                    continue
            
            return processed_results
            
        except Exception as e:
            st.error(f"Error searching UniProt: {str(e)}")
            return []
    
    def get_nucleotide_sequences_from_uniprot(self, uniprot_results: List[Dict], quoted_terms: List[str] = None) -> List[Dict]:
        """Extract nucleotide sequences from UniProt cross-references"""
        sequences_with_cds = []
        
        for uniprot_entry in uniprot_results:
            try:
                # Check if this entry matches quoted terms
                if quoted_terms and not self.matches_quoted_terms_uniprot(uniprot_entry, quoted_terms):
                    st.write(f"  ⏭️ **Skipped UniProt {uniprot_entry['accession']}:** No match for quoted terms")
                    continue
                
                st.write(f"🧬 **Processing UniProt {uniprot_entry['accession']}:** {uniprot_entry['protein_name']}")
                
                # Try to get nucleotide sequences from cross-references
                cds_sequences = []
                
                for nucleotide_ref in uniprot_entry.get('nucleotide_refs', []):
                    try:
                        db_name = nucleotide_ref['database']
                        nucleotide_id = nucleotide_ref['id']
                        
                        st.write(f"  🔗 **Checking {db_name} reference:** {nucleotide_id}")
                        
                        # Try to get nucleotide sequence
                        if db_name == 'EMBL':
                            nucleotide_seq = self.get_embl_sequence(nucleotide_id)
                        elif db_name == 'RefSeq':
                            nucleotide_seq = self.get_refseq_sequence(nucleotide_id)
                        else:
                            continue
                        
                        if nucleotide_seq:
                            cds_info = {
                                'accession': nucleotide_id,
                                'protein_name': uniprot_entry['protein_name'],
                                'gene_name': uniprot_entry['gene_names'],
                                'product': uniprot_entry['protein_name'],
                                'locus_tag': uniprot_entry['accession'],
                                'start_position': 1,
                                'end_position': len(nucleotide_seq),
                                'header': f">{nucleotide_id} {uniprot_entry['protein_name']}",
                                'sequence': nucleotide_seq,
                                'length': len(nucleotide_seq),
                                'url': f"https://www.ncbi.nlm.nih.gov/nuccore/{nucleotide_id}",
                                'valid_dna': self.is_valid_dna_sequence(nucleotide_seq),
                                'source_database': db_name,
                                'uniprot_accession': uniprot_entry['accession']
                            }
                            cds_sequences.append(cds_info)
                            st.write(f"    ✅ **Retrieved {db_name} sequence:** {len(nucleotide_seq)} bp")
                        else:
                            st.write(f"    ❌ **Failed to retrieve {db_name} sequence**")
                        
                        time.sleep(0.5)  # Rate limiting
                        
                    except Exception as e:
                        st.write(f"    ❌ **Error processing {db_name} reference:** {str(e)}")
                        continue
                
                # If we found nucleotide sequences, add this entry
                if cds_sequences:
                    sequences_with_cds.append({
                        'accession': uniprot_entry['accession'],
                        'title': uniprot_entry['protein_name'],
                        'definition': f"{uniprot_entry['protein_name']} [{uniprot_entry['organism']}]",
                        'organism': uniprot_entry['organism'],
                        'length': max(cds['length'] for cds in cds_sequences),
                        'url': uniprot_entry['uniprot_url'],
                        'success': True,
                        'cds_sequences': cds_sequences,
                        'filtered_terms': quoted_terms or [],
                        'source': 'UniProt'
                    })
                    st.write(f"  ✅ **Added {len(cds_sequences)} CDS sequences from UniProt entry**")
                else:
                    st.write(f"  ❌ **No nucleotide sequences found for UniProt entry**")
                    
            except Exception as e:
                st.write(f"❌ **Error processing UniProt entry {uniprot_entry.get('accession', 'Unknown')}:** {str(e)}")
                continue
        
        return sequences_with_cds
    
    def get_embl_sequence(self, embl_id: str) -> str:
        """Get nucleotide sequence from EMBL database"""
        try:
            # Try NCBI E-utilities (EMBL records are often in NCBI)
            urls = [
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={embl_id}&rettype=fasta&retmode=text",
                f"https://www.ncbi.nlm.nih.gov/nuccore/{embl_id}?report=fasta&format=text"
            ]
            
            for url in urls:
                try:
                    response = self.session.get(url, timeout=30)
                    if response.status_code == 200 and response.text.startswith('>'):
                        lines = response.text.strip().split('\n')
                        sequence = ''.join(lines[1:]).upper()
                        sequence = re.sub(r'[^ATGCN]', '', sequence)
                        if sequence:
                            return sequence
                except Exception:
                    continue
            
            return ""
            
        except Exception as e:
            return ""
    
    def get_refseq_sequence(self, refseq_id: str) -> str:
        """Get nucleotide sequence from RefSeq database"""
        try:
            # RefSeq is available through NCBI
            urls = [
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={refseq_id}&rettype=fasta&retmode=text",
                f"https://www.ncbi.nlm.nih.gov/nuccore/{refseq_id}?report=fasta&format=text"
            ]
            
            for url in urls:
                try:
                    response = self.session.get(url, timeout=30)
                    if response.status_code == 200 and response.text.startswith('>'):
                        lines = response.text.strip().split('\n')
                        sequence = ''.join(lines[1:]).upper()
                        sequence = re.sub(r'[^ATGCN]', '', sequence)
                        if sequence:
                            return sequence
                except Exception:
                    continue
            
            return ""
            
        except Exception as e:
            return ""
    
    def matches_quoted_terms_uniprot(self, uniprot_entry: Dict, quoted_terms: List[str]) -> bool:
        """Check if a UniProt entry matches quoted terms"""
        if not quoted_terms:
            return True
        
        try:
            searchable_fields = [
                uniprot_entry.get('protein_name', '').lower(),
                uniprot_entry.get('gene_names', '').lower(),
                uniprot_entry.get('organism', '').lower(),
                uniprot_entry.get('accession', '').lower()
            ]
            
            for term in quoted_terms:
                for field in searchable_fields:
                    if field and term in field:
                        return True
            
            return False
            
        except Exception:
            return True
    
    def is_valid_dna_sequence(self, sequence: str) -> bool:
        """Check if sequence contains only valid DNA bases"""
        if not sequence:
            return False
        return all(base.upper() in 'ATGCN' for base in sequence)
    
    def extract_quoted_terms(self, query: str) -> List[str]:
        """Extract terms in quotes from the search query"""
        try:
            quoted_terms = re.findall(r'"([^"]+)"', query)
            if not quoted_terms:
                quoted_terms = re.findall(r"'([^']+)'", query)
            
            cleaned_terms = []
            for term in quoted_terms:
                cleaned_term = term.strip().lower()
                if cleaned_term:
                    cleaned_terms.append(cleaned_term)
            
            return cleaned_terms
            
        except Exception as e:
            st.write(f"❌ **Error extracting quoted terms:** {str(e)}")
            return []
    
    def ai_select_best_sequences(self, query: str, sequences_with_cds: List[Dict]) -> List[Dict]:
        """Use AI to select the most relevant sequences based on the query"""
        if not self.anthropic or not sequences_with_cds:
            return sequences_with_cds
        
        sequence_summaries = []
        for i, seq_data in enumerate(sequences_with_cds):
            cds_info = []
            for cds in seq_data.get('cds_sequences', []):
                cds_info.append(f"  - {cds.get('protein_name', 'Unknown')} ({cds.get('start_position', 0)}-{cds.get('end_position', 0)}, {cds.get('length', 0)} bp)")
            
            summary = f"""
Sequence {i+1}:
- Accession: {seq_data['accession']}
- Title: {seq_data.get('title', 'N/A')}
- Definition: {seq_data.get('definition', 'N/A')}
- Organism: {seq_data.get('organism', 'N/A')}
- Source: {seq_data.get('source', 'Unknown')}
- Length: {seq_data.get('length', 0)} bp
- CDS Count: {len(seq_data.get('cds_sequences', []))}
- CDS Details:
{chr(10).join(cds_info) if cds_info else '  - No CDS found'}
"""
            sequence_summaries.append(summary)
        
        context = "\n".join(sequence_summaries)
        
        prompt = f"""
You are a bioinformatics expert. A user is searching for: "{query}"

Below are the available sequences with their CDS information from both NCBI and UniProt databases:

{context}

Please analyze these sequences and rank them by relevance to the user's query. Consider:
1. How well the organism/title matches the query
2. The presence and quality of CDS sequences
3. The biological relevance to the query
4. The completeness of the sequence data
5. The source database (NCBI vs UniProt)

Return your response as a JSON list of accession numbers in order of relevance (most relevant first), with a brief explanation for each.

Format: {{"rankings": [{{"accession": "XXX", "rank": 1, "reason": "explanation"}}, ...]}}
"""

        try:
            message = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            
            import json
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                rankings_data = json.loads(json_match.group())
                rankings = rankings_data.get('rankings', [])
                
                ordered_sequences = []
                for ranking in rankings:
                    accession = ranking['accession']
                    for seq_data in sequences_with_cds:
                        if seq_data['accession'] == accession:
                            seq_data['ai_rank'] = ranking['rank']
                            seq_data['ai_reason'] = ranking['reason']
                            ordered_sequences.append(seq_data)
                            break
                
                ranked_accessions = [r['accession'] for r in rankings]
                for seq_data in sequences_with_cds:
                    if seq_data['accession'] not in ranked_accessions:
                        ordered_sequences.append(seq_data)
                
                return ordered_sequences
            
        except Exception as e:
            logger.error(f"Error in AI ranking: {e}")
        
        return sequences_with_cds
    
    def create_cds_download_data(self, sequences_with_cds: List[Dict]) -> pd.DataFrame:
        """Create downloadable DataFrame with CDS sequences"""
        download_data = []
        
        for seq_data in sequences_with_cds:
            base_info = {
                'Source_Database': seq_data.get('source', 'Unknown'),
                'Accession': seq_data.get('accession', ''),
                'Title': seq_data.get('title', ''),
                'Definition': seq_data.get('definition', ''),
                'Organism': seq_data.get('organism', ''),
                'Sequence_Length_bp': seq_data.get('length', 0),
                'Database_URL': seq_data.get('url', ''),
                'Filtered_Terms': ', '.join(seq_data.get('filtered_terms', [])),
                'AI_Rank': seq_data.get('ai_rank', ''),
                'AI_Reason': seq_data.get('ai_reason', '')
            }
            
            if seq_data.get('cds_sequences'):
                for i, cds in enumerate(seq_data['cds_sequences']):
                    row = base_info.copy()
                    row.update({
                        'CDS_Number': i + 1,
                        'Gene_Name': cds.get('gene_name', ''),
                        'Protein_Name': cds.get('protein_name', ''),
                        'Product': cds.get('product', ''),
                        'Locus_Tag': cds.get('locus_tag', ''),
                        'Start_Position': cds.get('start_position', 0),
                        'End_Position': cds.get('end_position', 0),
                        'CDS_Header': cds.get('header', ''),
                        'CDS_Sequence': cds.get('sequence', ''),
                        'CDS_Length_bp': cds.get('length', 0),
                        'Valid_DNA': cds.get('valid_dna', False),
                        'Nucleotide_URL': cds.get('url', ''),
                        'Source_DB': cds.get('source_database', ''),
                        'UniProt_Accession': cds.get('uniprot_accession', '')
                    })
                    download_data.append(row)
            else:
                row = base_info.copy()
                row.update({
                    'CDS_Number': 0,
                    'Gene_Name': 'No matching CDS found',
                    'Protein_Name': 'No matching CDS found',
                    'Product': '',
                    'Locus_Tag': '',
                    'Start_Position': 0,
                    'End_Position': 0,
                    'CDS_Header': '',
                    'CDS_Sequence': '',
                    'CDS_Length_bp': 0,
                    'Valid_DNA': False,
                    'Nucleotide_URL': '',
                    'Source_DB': '',
                    'UniProt_Accession': ''
                })
                download_data.append(row)
        
        return pd.DataFrame(download_data)


def test_serper_connection(api_key: str) -> Dict:
    """Test SERPER API connection with a simple search"""
    try:
        url = "https://google.serper.dev/search"
        payload = {"q": "test", "num": 1}
        headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return {"success": True, "message": "Connection successful"}
        elif response.status_code == 403:
            return {"success": False, "error": "Invalid API key or insufficient permissions"}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "error": "Connection error - unable to reach google.serper.dev"}
    except requests.exceptions.Timeout as e:
        return {"success": False, "error": "Connection timeout"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def test_uniprot_connection() -> Dict:
    """Test UniProt API connection"""
    try:
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {"query": "insulin", "format": "json", "size": 1}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                return {"success": True, "message": "UniProt connection successful"}
            else:
                return {"success": False, "error": "Unexpected response format"}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "error": "Connection error - unable to reach UniProt API"}
    except requests.exceptions.Timeout as e:
        return {"success": False, "error": "Connection timeout"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def create_download_link(df, filename):
    """Create download link for DataFrame as Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    processed_data = output.getvalue()
    return processed_data

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
        elif method == "In-Frame Analysis":  # Updated from "CAI Weight Analysis"
            weights, codons_list = get_codon_weights_row(clean_seq)
            result = {
                'Position': list(range(1, len(codons_list) + 1)),
                'DNA_Codon': codons_list,
                'CAI_Weight': weights,
                'Amino_Acid': [st.session_state.genetic_code.get(c, '?') for c in codons_list],
                'Method': method
            }
        elif method == "Balanced Optimization":
            optimized = balanced_optimisation(clean_seq, bias_weight)
            weights, _ = get_codon_weights_row(optimized)
            result = {
                'Original_DNA': clean_seq,
                'Protein': protein_seq,
                'Optimized_DNA': optimized,
                'CAI_Weights': ','.join(f"{w:.4f}" for w in weights),
                'Method': method
            }
        elif method == "NC Stop Codon Optimization":
            optimized = nc_stop_codon_optimisation(clean_seq)
            weights, _ = get_codon_weights_row(optimized)
            result = {
                'Original_DNA': clean_seq,
                'Protein': protein_seq,
                'Optimized_DNA': optimized,
                'CAI_Weights': ','.join(f"{w:.4f}" for w in weights),
                'Method': method
            }
        elif method == "JT Plus1 Stop Optimization":
            optimized = JT_Plus1_Stop_Optimized(clean_seq)
            weights, _ = get_codon_weights_row(optimized)
            result = {
                'Original_DNA': clean_seq,
                'Protein': protein_seq,
                'Optimized_DNA': optimized,
                'CAI_Weights': ','.join(f"{w:.4f}" for w in weights),
                'Method': method
            }
        elif method == "+1 Frame Analysis":  # Updated from "Sequence Analysis"
            plus1_stop_counts = number_of_plus1_stops(clean_seq)
            start_pos, end_pos = find_coding_sequence_bounds(clean_seq)
            slippery_count = number_of_slippery_motifs(clean_seq)
            gc_content = calculate_gc_content(clean_seq)
            minus1_stop_counts = number_of_minus1_stops(clean_seq)
            
            if start_pos is not None and end_pos is not None:
                coding_length = end_pos - start_pos
                plus1_len = coding_length // 3
                coding_info = f"{start_pos}-{end_pos} ({coding_length} bp)"
            elif start_pos is not None:
                coding_length = len(clean_seq) - start_pos
                plus1_len = coding_length // 3
                coding_info = f"{start_pos}-end ({coding_length} bp, no stop found)"
            else:
                plus1_len = 0
                coding_info = "No valid coding sequence found"
                coding_length = 0
            
            result = {
                'Sequence_Length': len(clean_seq),
                'Protein_Length': len(protein_seq),
                'GC_Content': gc_content,
                'Coding_Info': coding_info,
                'Plus1_TAA_Count': plus1_stop_counts['TAA'],
                'Plus1_TAG_Count': plus1_stop_counts['TAG'],
                'Plus1_TGA_Count': plus1_stop_counts['TGA'],
                'Plus1_Total_Stops': plus1_stop_counts['total'],
                'minus1_TAA_Count': minus1_stop_counts['TAA'],
                'minus1_TAG_Count': minus1_stop_counts['TAG'],
                'minus1_TGA_Count': minus1_stop_counts['TGA'],
                'minus1_Total_Stops': minus1_stop_counts['total'],
                'Slippery_Motifs': slippery_count,
                'Stop_Density': plus1_stop_counts['total']/max(1, plus1_len) if plus1_len > 0 else 0,
                'Method': method
            }
        
        return result, None
    except Exception as e:
        return None, str(e)

def main():
    """Main Streamlit application"""
    st.sidebar.write("🚀 DEBUG: App is starting...")  # ADD THIS LINE

    # Apply the selected theme CSS
    inject_app_theme()
    # Initialize research engines
    if 'patent_engine' not in st.session_state:
        st.session_state.patent_engine = PatentSearchEngine()
    if 'ncbi_engine' not in st.session_state:
        st.session_state.ncbi_engine = NCBISearchEngine()
    if 'uniprot_engine' not in st.session_state:
        st.session_state.uniprot_engine = UniProtSearchEngine()
        
    st.title("DNA Codon Optimization and Analysis Tool")
    st.markdown("DNA sequence optimization and analysis")
    
    # Sidebar for settings and configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Auto-load default codon file if available and not already loaded
        if not st.session_state.genetic_code and 'codon_data_loaded' not in st.session_state:
            default_codon_file = "HumanCodons.xlsx"
            if os.path.exists(default_codon_file):
                try:
                    with open(default_codon_file, 'rb') as f:
                        file_content = f.read()
                    genetic_code, codon_weights, preferred_codons, human_codon_usage, aa_to_codons, codon_df = load_codon_data_from_file(file_content)
                    st.session_state.genetic_code = genetic_code
                    st.session_state.codon_weights = codon_weights
                    st.session_state.preferred_codons = preferred_codons
                    st.session_state.human_codon_usage = human_codon_usage
                    st.session_state.aa_to_codons = aa_to_codons
                    st.session_state.codon_data_loaded = True
                    st.session_state.codon_file_source = "Default (HumanCodons.xlsx)"
                    st.success(f"Auto-loaded {len(codon_df)} codon entries from HumanCodons.xlsx")
                except Exception as e:
                    st.warning(f"Could not auto-load HumanCodons.xlsx: {e}")
        
        # Display current codon file status
        if st.session_state.genetic_code:
            codon_source = st.session_state.get('codon_file_source', 'Unknown')
            st.info(f"**Codon Data Loaded:** {codon_source}")
            if st.button("Clear Codon Data", help="Clear current codon data to upload a different file"):
                # Clear all codon-related session state
                st.session_state.genetic_code = {}
                st.session_state.codon_weights = {}
                st.session_state.preferred_codons = {}
                st.session_state.human_codon_usage = {}
                st.session_state.aa_to_codons = defaultdict(list)
                if 'codon_data_loaded' in st.session_state:
                    del st.session_state.codon_data_loaded
                if 'codon_file_source' in st.session_state:
                    del st.session_state.codon_file_source
                st.rerun()
        
        # Codon usage file upload
        uploaded_file = st.file_uploader(
            "Upload Different Codon Usage File", 
            type=['xlsx'],
            help="Upload a different codon usage frequency file to override the default",
            key="codon_uploader"
        )
        
        if uploaded_file is not None:
            try:
                file_content = uploaded_file.read()
                genetic_code, codon_weights, preferred_codons, human_codon_usage, aa_to_codons, codon_df = load_codon_data_from_file(file_content)
                st.session_state.genetic_code = genetic_code
                st.session_state.codon_weights = codon_weights
                st.session_state.preferred_codons = preferred_codons
                st.session_state.human_codon_usage = human_codon_usage
                st.session_state.aa_to_codons = aa_to_codons
                st.session_state.codon_data_loaded = True
                st.session_state.codon_file_source = f"Uploaded ({uploaded_file.name})"
                st.success(f"Loaded {len(codon_df)} codon entries from {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading codon file: {e}")
        elif not st.session_state.genetic_code:
            st.warning("No codon usage file found. Please upload HumanCodons.xlsx or place it in the application directory.")
            #st.stop()
        
        st.divider()
        
        # Algorithm settings
        st.subheader("Algorithm Settings")
        bias_weight = st.slider(
            "Bias Weight (Balanced Optimization)", 
            min_value=0.1, 
            max_value=5.0, 
            value=float(st.session_state.config.get("bias_weight", BIAS_WEIGHT_DEFAULT)),
            step=0.1,
            help="Weight for +1 frame stop codon bias in balanced optimization"
        )
        st.session_state.config["bias_weight"] = bias_weight
        
        st.divider()
        
        # Theme selection
        st.subheader("Appearance")
        theme_name = st.selectbox(
            "Select Theme",
            options=list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.active_theme),
            help="Change the color scheme of the application."
        )
        if theme_name != st.session_state.active_theme:
            st.session_state.active_theme = theme_name
            st.rerun()
        
        st.info(THEMES[st.session_state.active_theme]["info"])
        
        # Accumulation settings
        st.subheader("Result Management")
        accumulate_results = st.checkbox(
            "Accumulate Results", 
            help="Collect multiple single-sequence results before download"
        )
        
        if st.session_state.accumulated_results:
            st.info(f"Accumulated: {len(st.session_state.accumulated_results)} results")
            if st.button("Clear Accumulated Results"):
                st.session_state.accumulated_results = []
                st.session_state.run_counter = 0
                st.rerun()
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Single Sequence", "Batch Optimization", "CDS Database Search", "Patent Search", "About", "mRNA Design", "Cancer Vaccine Design"])

    with tab1:
        st.header("Single Sequence Optimization")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Check for and handle transferred sequence
            if 'transfer_sequence' in st.session_state and st.session_state.transfer_sequence:
                transfer_info = st.session_state.get('transfer_sequence_info', {})
                source_info = transfer_info.get('source', 'another tab')
                st.success(f"🎯 Sequence from **{source_info}** has been loaded below.")
                
                st.session_state.sequence_input_area = st.session_state.transfer_sequence
                
                # Clean up session state after using the transferred sequence
                del st.session_state.transfer_sequence
                if 'transfer_sequence_info' in st.session_state:
                    del st.session_state.transfer_sequence_info

            sequence_input = st.text_area(
                "DNA Sequence",
                height=150,
                placeholder="Enter DNA sequence (A, T, G, C only)... or transfer from CDS Database Search",
                help="Paste your DNA sequence here. Spaces and newlines will be removed automatically. You can also transfer sequences from the CDS Database Search tab.",
                key="sequence_input_area"
            )
        
        with col2:
            optimization_method = st.selectbox(
    "Optimization Method",
    [
        "In-Frame Analysis",           # 1st
        "+1 Frame Analysis",           # 2nd
        "Standard Codon Optimization", # 3rd
        "NC Stop Codon Optimization",  # 4th
        "Balanced Optimization",       # 5th
        "JT Plus1 Stop Optimization"   # 6th
    ],
    help="Choose the optimization algorithm to apply"
)
            
            # Accumulation settings moved here
            st.markdown("**Result Management:**")
            accumulate_results = st.checkbox(
                "Accumulate Results", 
                help="Collect multiple single-sequence results before download",
                key="accumulate_results_tab1"
            )
            
            if st.session_state.accumulated_results:
                st.info(f"Accumulated: {len(st.session_state.accumulated_results)} results")
                if st.button("Clear Accumulated Results", key="clear_accumulated_tab1"):
                    st.session_state.accumulated_results = []
                    st.session_state.run_counter = 0
                    st.rerun()
            
            run_optimization_button = st.button("Run Optimization", type="primary")
        
        # Results section - using full width outside of columns
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
                    
                    # Full-width results section
                    st.divider()
                    
                    # Display results using full page width
                    if optimization_method == "In-Frame Analysis":
                        df = pd.DataFrame(result)
                        st.subheader("In-Frame Analysis Results")
                        
                        # Create interactive In-Frame graph with GC content
                        if not df.empty and 'CAI_Weight' in df.columns:
                            st.subheader("📊 Interactive CAI Weights and 10bp GC Content")
                            
                            positions = df['Position'].tolist()
                            cai_weights = df['CAI_Weight'].tolist()
                            amino_acids = df['Amino_Acid'].tolist()
                            
                            # Create interactive plot
                            colors = get_consistent_color_palette(1, "optimization")
                            fig = create_interactive_cai_gc_plot(
                                positions, 
                                cai_weights, 
                                amino_acids, 
                                sequence_input, 
                                f"Sequence ({len(sequence_input)} bp)",
                                colors['optimized']
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Statistics including GC content
                            col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
                            with col_stat1:
                                st.metric("Average CAI", f"{np.mean(cai_weights):.3f}")
                            with col_stat2:
                                st.metric("Min CAI", f"{np.min(cai_weights):.3f}")
                            with col_stat3:
                                st.metric("Max CAI", f"{np.max(cai_weights):.3f}")
                            with col_stat4:
                                low_cai_count = sum(1 for w in cai_weights if w < 0.5)
                                st.metric("Low CAI (<0.5)", f"{low_cai_count}/{len(cai_weights)}")
                            with col_stat5:
                                # Calculate GC content from the original sequence
                                gc_content = calculate_gc_content(sequence_input)
                                st.metric("GC Content", f"{gc_content:.1f}%")
                            
                            # Interactive data table
                            st.subheader("📋 Detailed Results")
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button
                            excel_data = create_download_link(df, f"InFrame_Analysis_{len(sequence_input)}bp.xlsx")
                            st.download_button(
                                label="Download Results (Excel)",
                                data=excel_data,
                                file_name=f"InFrame_Analysis_{len(sequence_input)}bp.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                    elif optimization_method == "+1 Frame Analysis":
                        st.subheader("+1 Frame Analysis Results")
                        
                        # Create metrics display using full width
                        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
                        with metric_col1:
                            st.metric("Sequence Length", f"{result['Sequence_Length']} bp")
                        with metric_col2:
                            st.metric("Protein Length", f"{result['Protein_Length']} aa")
                        with metric_col3:
                            st.metric("GC Content", f"{result['GC_Content']:.1f}%")
                        with metric_col4:
                            st.metric("Total +1 Stops", result['Plus1_Total_Stops'])
                        with metric_col5:
                            st.metric("Slippery Motifs", result['Slippery_Motifs'])
                        with metric_col6:
                            st.metric("Total -1 Stops", result['minus1_Total_Stops'])
                        
                        # Visualization of +1 frame stop codons
                        if result['Plus1_Total_Stops'] > 0:
                            st.subheader("Stop Codon Distribution (+1 Frame)")
                            
                            # Create chart layout
                            chart_col, table_col = st.columns([2, 1])
                            
                            with chart_col:
                                stop_data = {
                                    'Codon': ['TAA', 'TAG', 'TGA'],
                                    'Count': [result['Plus1_TAA_Count'], result['Plus1_TAG_Count'], result['Plus1_TGA_Count']]
                                }
                                stop_df = pd.DataFrame(stop_data)
                                stop_df = stop_df[stop_df['Count'] > 0]  # Only show non-zero counts
                                
                                if not stop_df.empty:
                                    # Create a pie chart with different colors
                                    fig, ax = plt.subplots(figsize=(8, 8))
                                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(stop_df)]  # Different colors
                                    
                                    wedges, texts, autotexts = ax.pie(
                                        stop_df['Count'], 
                                        labels=stop_df['Codon'], 
                                        colors=colors, 
                                        autopct='%1.1f%%', 
                                        startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
                                    )
                                    
                                    # Enhance text appearance
                                    for autotext in autotexts:
                                        autotext.set_color('white')
                                        autotext.set_fontweight('bold')
                                        autotext.set_fontsize(11)
                                    
                                    ax.set_title('Stop Codon Distribution in +1 Frame', fontsize=14, fontweight='bold', pad=20)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                            
                            with table_col:
                                st.markdown("**📊 Stop Codon Summary:**")
                                
                                # Calculate stops per 100bp
                                sequence_length = result.get('Sequence_Length', 1)
                                stops_per_100bp = (result['Plus1_Total_Stops'] / sequence_length) * 100
                                
                                stop_summary = pd.DataFrame({
                                    'Metric': ['TAA Count', 'TAG Count', 'TGA Count', 'Total Stops', 'Stops per 100bp'],
                                    'Value': [
                                        result['Plus1_TAA_Count'], 
                                        result['Plus1_TAG_Count'], 
                                        result['Plus1_TGA_Count'], 
                                        result['Plus1_Total_Stops'],
                                        f"{stops_per_100bp:.2f}"
                                    ]
                                })
                                st.dataframe(stop_summary, use_container_width=True, hide_index=True)
                            
                            # Summary table
                            sequence_length = result.get('Sequence_Length', 1)
                            stops_per_100bp = (result['Plus1_Total_Stops'] / sequence_length) * 100
                            
                            stop_summary = pd.DataFrame({
                                'Metric': ['TAA Count', 'TAG Count', 'TGA Count', 'Total Stops', 'Stops per 100bp'],
                                'Value': [
                                    result['Plus1_TAA_Count'], 
                                    result['Plus1_TAG_Count'], 
                                    result['Plus1_TGA_Count'], 
                                    result['Plus1_Total_Stops'],
                                    f"{stops_per_100bp:.2f}"
                                ]
                            })
                            st.dataframe(stop_summary, use_container_width=True, hide_index=True)
                            
                        # Visualization of -1 frame stop codons
                        if result['minus1_Total_Stops'] > 0:
                            st.subheader("Stop Codon Distribution (-1 Frame)")
                            
                            # Create chart layout
                            chart_col, table_col = st.columns([2, 1])
                            
                            with chart_col:
                                stop_data = {
                                    'Codon': ['TAA', 'TAG', 'TGA'],
                                    'Count': [result['minus1_TAA_Count'], result['minus1_TAG_Count'], result['minus1_TGA_Count']]
                                }
                                stop_df = pd.DataFrame(stop_data)
                                stop_df = stop_df[stop_df['Count'] > 0]  # Only show non-zero counts
                                
                                if not stop_df.empty:
                                    # Create a pie chart with different colors
                                    fig, ax = plt.subplots(figsize=(8, 8))
                                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(stop_df)]  # Different colors
                                    
                                    wedges, texts, autotexts = ax.pie(
                                        stop_df['Count'], 
                                        labels=stop_df['Codon'], 
                                        colors=colors, 
                                        autopct='%1.1f%%', 
                                        startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
                                    )
                                    
                                    # Enhance text appearance
                                    for autotext in autotexts:
                                        autotext.set_color('white')
                                        autotext.set_fontweight('bold')
                                        autotext.set_fontsize(11)
                                    
                                    ax.set_title('Stop Codon Distribution in -1 Frame', fontsize=14, fontweight='bold', pad=20)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                            
                            with table_col:
                                st.markdown("**📊 Stop Codon Summary:**")
                                
                                # Calculate stops per 100bp
                                sequence_length = result.get('Sequence_Length', 1)
                                stops_per_100bp = (result['minus1_Total_Stops'] / sequence_length) * 100
                                
                                stop_summary = pd.DataFrame({
                                    'Metric': ['TAA Count', 'TAG Count', 'TGA Count', 'Total Stops', 'Stops per 100bp'],
                                    'Value': [
                                        result['minus1_TAA_Count'], 
                                        result['minus1_TAG_Count'], 
                                        result['minus1_TGA_Count'], 
                                        result['minus1_Total_Stops'],
                                        f"{stops_per_100bp:.2f}"
                                    ]
                                })
                                st.dataframe(stop_summary, use_container_width=True, hide_index=True)
                            
                            # Summary table
                            sequence_length = result.get('Sequence_Length', 1)
                            stops_per_100bp = (result['minus1_Total_Stops'] / sequence_length) * 100
                            
                            stop_summary = pd.DataFrame({
                                'Metric': ['TAA Count', 'TAG Count', 'TGA Count', 'Total Stops', 'Stops per 100bp'],
                                'Value': [
                                    result['minus1_TAA_Count'], 
                                    result['minus1_TAG_Count'], 
                                    result['minus1_TGA_Count'], 
                                    result['minus1_Total_Stops'],
                                    f"{stops_per_100bp:.2f}"
                                ]
                            })
                            st.dataframe(stop_summary, use_container_width=True, hide_index=True)

                        # Add the new graphs
                        st.subheader("CAI and Stop Codon Analysis")

                        # Get CAI data
                        cai_result, cai_error = run_single_optimization(sequence_input, "In-Frame Analysis")
                        if not cai_error and cai_result:
                            cai_df = pd.DataFrame(cai_result)
                            positions = cai_df['Position'].tolist()
                            cai_weights = cai_df['CAI_Weight'].tolist()
                            amino_acids = cai_df['Amino_Acid'].tolist()

                            # Get stop codon positions
                            plus1_stop_positions = get_plus1_stop_positions(sequence_input)
                            minus1_stop_positions = get_minus1_stop_positions(sequence_input)

                            # Create +1 stop codon plot
                            if plus1_stop_positions:
                                fig_plus1 = create_interactive_cai_stop_codon_plot(
                                    positions,
                                    cai_weights,
                                    amino_acids,
                                    plus1_stop_positions,
                                    f"Sequence ({len(sequence_input)} bp)",
                                    "+1 Frame"
                                )
                                st.plotly_chart(fig_plus1, use_container_width=True)
                            else:
                                st.info("No +1 stop codons found to plot against CAI.")

                            # Create -1 stop codon plot
                            if minus1_stop_positions:
                                fig_minus1 = create_interactive_cai_stop_codon_plot(
                                    positions,
                                    cai_weights,
                                    amino_acids,
                                    minus1_stop_positions,
                                    f"Sequence ({len(sequence_input)} bp)",
                                    "-1 Frame"
                                )
                                st.plotly_chart(fig_minus1, use_container_width=True)
                            else:
                                st.info("No -1 stop codons found to plot against CAI.")
                        else:
                            st.warning("Could not generate CAI data for stop codon plots.")
                        
                    else:
                        # Standard optimization results
                        st.subheader("Optimization Results")
                        
                        # Show sequence comparison for optimization methods using full width
                        if 'Optimized_DNA' in result:
                            st.subheader("Sequence Comparison")
                            seq_col1, seq_col2 = st.columns(2)
                            
                            with seq_col1:
                                display_copyable_sequence(result['Original_DNA'], "Original Sequence", "orig")
                            with seq_col2:
                                display_copyable_sequence(result['Optimized_DNA'], "Optimized Sequence", "opt")
                            
                            # Results summary table
                            st.subheader("Results Summary")
                            result_data = []
                            for key, value in result.items():
                                if key != 'Method' and key not in ['Original_DNA', 'Optimized_DNA']:
                                    result_data.append({'Field': key.replace('_', ' ').title(), 'Value': str(value)})
                            
                            result_df = pd.DataFrame(result_data)
                            st.dataframe(result_df, use_container_width=True)
                        else:
                            # For methods without optimization (like pure analysis)
                            result_data = []
                            for key, value in result.items():
                                if key != 'Method':
                                    result_data.append({'Field': key.replace('_', ' ').title(), 'Value': str(value)})
                            
                            result_df = pd.DataFrame(result_data)
                            st.dataframe(result_df, use_container_width=True)
                    
                    # Accumulation option
                    if accumulate_results:
                        st.session_state.run_counter += 1
                        result_with_id = result.copy()
                        result_with_id['Run_ID'] = st.session_state.run_counter
                        st.session_state.accumulated_results.append(result_with_id)
                        st.info(f"Result added to accumulation buffer (Total: {len(st.session_state.accumulated_results)})")
        
        # Display accumulated results if any exist
        if st.session_state.accumulated_results:
            st.divider()
            st.subheader("📚 Accumulated Results")
            
            with st.expander(f"View Accumulated Results ({len(st.session_state.accumulated_results)} total)", expanded=False):
                # Convert accumulated results to DataFrame
                acc_df = pd.DataFrame(st.session_state.accumulated_results)
                
                # Reorder columns
                if 'Run_ID' in acc_df.columns:
                    cols = ['Run_ID'] + [col for col in acc_df.columns if col != 'Run_ID']
                    acc_df = acc_df[cols]
                
                st.dataframe(acc_df, use_container_width=True)
                
                # Download accumulated results
                excel_data = create_download_link(acc_df, f"Accumulated_Results_{len(st.session_state.accumulated_results)}_runs.xlsx")
                st.download_button(
                    label="Download Accumulated Results (Excel)",
                    data=excel_data,
                    file_name=f"Accumulated_Results_{len(st.session_state.accumulated_results)}_runs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                
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
                # Process uploaded file
                content = batch_file.read()
                
                # Handle different content types
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                elif content is None:
                    st.error("Failed to read file content")
                    st.stop()
                
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
                    # Text format - one sequence per line
                    lines = [line.strip() for line in content.splitlines() if line.strip()]
                    for i, line in enumerate(lines):
                        sequences.append((f"Sequence_{i+1}", line.upper()))
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                sequences = []
            
            if sequences:
                st.success(f"Loaded {len(sequences)} sequences")
                
                batch_method = st.selectbox(
    "Batch Optimization Method",
    [
        "In-Frame Analysis",           # 1st
        "+1 Frame Analysis",           # 2nd
        "Standard Codon Optimization", # 3rd
        "NC Stop Codon Optimization",  # 4th
        "Balanced Optimization",       # 5th
        "JT Plus1 Stop Optimization"   # 6th
    ]
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
                        # Convert to DataFrame
                        batch_df = pd.DataFrame(results)
                        
                        # Reorder columns to put Sequence_Name first
                        cols = ['Sequence_Name'] + [col for col in batch_df.columns if col != 'Sequence_Name']
                        batch_df = batch_df[cols]
                        
                        # In-Frame Analysis - Individual Interactive Charts for Each Sequence
                        if batch_method == "In-Frame Analysis" and not batch_df.empty:
                            st.subheader("📊 Interactive Individual In-Frame Analysis")
                            
                            # Create a unique key for this batch session
                            batch_key = f"batch_{len(sequences)}_{hash(str([name for name, _ in sequences]))}"
                            cai_data_key = f'batch_cai_data_{batch_key}'
                            
                            # Initialize cai_sequences
                            cai_sequences = []
                            
                            # Process sequences if not already cached
                            if cai_data_key not in st.session_state:
                                with st.spinner("Processing In-Frame data for all sequences..."):
                                    st.session_state[cai_data_key] = []
                                    
                                    progress_cai = st.progress(0)
                                    status_cai = st.empty()
                                    
                                    for i, (name, seq) in enumerate(sequences):
                                        status_cai.text(f"Processing {name}... ({i+1}/{len(sequences)})")
                                        try:
                                            result, error = run_single_optimization(seq, batch_method, bias_weight)
                                            if not error and isinstance(result, dict) and 'Position' in result:
                                                st.session_state[cai_data_key].append({
                                                    'name': name,
                                                    'sequence': seq,
                                                    'cai_data': pd.DataFrame(result)
                                                })
                                            progress_cai.progress((i + 1) / len(sequences))
                                        except Exception as e:
                                            continue
                                    
                                    # Clear progress indicators after processing is complete
                                    progress_cai.empty()
                                    status_cai.empty()
                            
                            # Get the processed sequences from session state
                            cai_sequences = st.session_state.get(cai_data_key, [])
                            
                            # Display results
                            if cai_sequences:
                                # Display all In-Frame interactive graphs
                                colors = get_consistent_color_palette(len(cai_sequences), "analysis")
                                for i, selected_data in enumerate(cai_sequences):
                                    df = selected_data['cai_data']
                                    seq_name = selected_data['name']
                                    seq_sequence = selected_data['sequence']
                                    
                                    st.markdown(f"### 📊 Interactive In-Frame Analysis for: {seq_name}")
                                    
                                    if not df.empty and 'CAI_Weight' in df.columns:
                                        positions = df['Position'].tolist()
                                        cai_weights = df['CAI_Weight'].tolist()
                                        amino_acids = df['Amino_Acid'].tolist()
                                        
                                        # Create interactive plot with GC content
                                        color = colors[i % len(colors)]
                                        fig = create_interactive_cai_gc_plot(
                                            positions, 
                                            cai_weights, 
                                            amino_acids, 
                                            seq_sequence, 
                                            seq_name,
                                            color
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Statistics including GC content
                                        col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
                                        with col_stat1:
                                            st.metric("Average CAI", f"{np.mean(cai_weights):.3f}")
                                        with col_stat2:
                                            st.metric("Min CAI", f"{np.min(cai_weights):.3f}")
                                        with col_stat3:
                                            st.metric("Max CAI", f"{np.max(cai_weights):.3f}")
                                        with col_stat4:
                                            low_cai_count = sum(1 for w in cai_weights if w < 0.5)
                                            st.metric("Low CAI (<0.5)", f"{low_cai_count}/{len(cai_weights)}")
                                        with col_stat5:
                                            # Calculate GC content for this sequence
                                            gc_content = calculate_gc_content(seq_sequence)
                                            st.metric("GC Content", f"{gc_content:.1f}%")
                                        
                                        # Data table in expandable section
                                        with st.expander(f"📋 View detailed In-Frame data for {seq_name}"):
                                            st.dataframe(df, use_container_width=True)
                                        
                                        st.divider()  # Add separator between sequences
                                    else:
                                        st.warning(f"No In-Frame data available for {seq_name}")
                            else:
                                st.warning("No valid In-Frame data found for any sequences")

                        # +1 Frame Analysis visualization with interactive charts
                        elif batch_method == "+1 Frame Analysis" and not batch_df.empty:
                            st.subheader("📊 Interactive Batch +1 Frame Analysis")
                            
                            # Check if we have the required columns and valid data
                            required_cols = ['Plus1_TAA_Count', 'Plus1_TAG_Count', 'Plus1_TGA_Count']
                            gc_available = 'GC_Content' in batch_df.columns
                            
                            if all(col in batch_df.columns for col in required_cols):
                                
                                # Overall statistics first
                                total_taa = batch_df['Plus1_TAA_Count'].sum()
                                total_tag = batch_df['Plus1_TAG_Count'].sum()
                                total_tga = batch_df['Plus1_TGA_Count'].sum()
                                total_stops = total_taa + total_tag + total_tga
                                
                                # Summary statistics
                                st.markdown("#### 📈 Overall Statistics")
                                if gc_available:
                                    avg_gc = batch_df['GC_Content'].mean()
                                    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
                                    with col_stat2:
                                        st.metric("Avg GC Content", f"{avg_gc:.1f}%")
                                else:
                                    col_stat1, col_stat3, col_stat4, col_stat5 = st.columns(4)
                                
                                with col_stat1:
                                    st.metric("Total Sequences", len(sequences))
                                with col_stat3:
                                    st.metric("Total +1 Stops", total_stops)
                                with col_stat4:
                                    avg_stops = total_stops / len(sequences) if len(sequences) > 0 else 0
                                    st.metric("Avg Stops/Seq", f"{avg_stops:.1f}")
                                with col_stat5:
                                    sequences_with_stops = len(batch_df[batch_df['Plus1_Total_Stops'] > 0])
                                    st.metric("Seqs with Stops", f"{sequences_with_stops}/{len(sequences)}")
                                
                                
                                    sequences_with_stops = len(batch_df[batch_df['Plus1_Total_Stops'] > 0])
                                    st.metric("Seqs with Stops", f"{sequences_with_stops}/{len(sequences)}")
                                
                                
                                # Individual sequence pie charts
                                if total_stops > 0:
                                    st.markdown("#### 🥧 Individual Sequence Stop Codon Distribution")
                                    
                                    # Create pie charts for each sequence that has stops
                                    sequences_with_stops_data = batch_df[batch_df['Plus1_Total_Stops'] > 0]
                                    
                                    if not sequences_with_stops_data.empty:
                                        # Create columns for pie charts (2 per row)
                                        cols_per_row = 2
                                        num_sequences = len(sequences_with_stops_data)
                                        
                                        for i in range(0, num_sequences, cols_per_row):
                                            cols = st.columns(cols_per_row)
                                            
                                            for j in range(cols_per_row):
                                                if i + j < num_sequences:
                                                    seq_data = sequences_with_stops_data.iloc[i + j]
                                                    seq_name = seq_data['Sequence_Name']
                                                    
                                                    taa_count = seq_data['Plus1_TAA_Count']
                                                    tag_count = seq_data['Plus1_TAG_Count']
                                                    tga_count = seq_data['Plus1_TGA_Count']
                                                    total_seq_stops = seq_data['Plus1_Total_Stops']
                                                    
                                                    if total_seq_stops > 0:
                                                        with cols[j]:
                                                            # Filter out zero values
                                                            pie_data = []
                                                            pie_labels = []
                                                            pie_colors = []
                                                            color_map = {'TAA': '#FF6B6B', 'TAG': '#4ECDC4', 'TGA': '#45B7D1'}
                                                            
                                                            for codon, count in [('TAA', taa_count), ('TAG', tag_count), ('TGA', tga_count)]:
                                                                if count > 0:
                                                                    pie_data.append(count)
                                                                    pie_labels.append(codon)
                                                                    pie_colors.append(color_map[codon])
                                                            
                                                            if pie_data:
                                                                fig_individual = go.Figure(data=[go.Pie(
                                                                    labels=pie_labels,
                                                                    values=pie_data,
                                                                    hole=.4,
                                                                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                                                                    textinfo='label+value',
                                                                    textfont_size=10,
                                                                    marker=dict(
                                                                        colors=pie_colors,
                                                                        line=dict(color='#FFFFFF', width=2)
                                                                    )
                                                                )])
                                                                
                                                                fig_individual.update_layout(
                                                                    title={
                                                                        'text': f'{seq_name[:20]}{"..." if len(seq_name) > 20 else ""}<br><sub>+1 Frame Stops</sub>',
                                                                        'x': 0.5,
                                                                        'font': {'size': 12}
                                                                    },
                                                                    annotations=[dict(
                                                                        text=f'{total_seq_stops}<br>Stops', 
                                                                        x=0.5, y=0.5, 
                                                                        font_size=11, 
                                                                        showarrow=False,
                                                                        font=dict(color="#2C3E50", weight="bold")
                                                                    )],
                                                                    height=300,
                                                                    showlegend=False,
                                                                    margin=dict(t=50, b=10, l=10, r=10)
                                                                )
                                                                
                                                                st.plotly_chart(fig_individual, use_container_width=True)
                                    else:
                                        st.info("No sequences with +1 frame stops found for individual visualization.")
                                
                                
                                
                                if total_stops > 0:
                                    # Interactive summary charts with breakdown by stop codon type
                                    st.markdown("#### 📊 Interactive Summary Charts")

                                    # Chart 1: +1 Stops per 100bp broken down by TAA, TAG, TGA
                                    sequence_names = batch_df['Sequence_Name'].tolist()
                                    sequence_lengths = batch_df['Sequence_Length'].tolist() if 'Sequence_Length' in batch_df.columns else [1] * len(sequence_names)
                                    
                                    # Calculate stops per 100bp for each type
                                    taa_per_100bp = [(batch_df.iloc[i]['Plus1_TAA_Count'] / sequence_lengths[i]) * 100 if sequence_lengths[i] > 0 else 0 for i in range(len(sequence_names))]
                                    tag_per_100bp = [(batch_df.iloc[i]['Plus1_TAG_Count'] / sequence_lengths[i]) * 100 if sequence_lengths[i] > 0 else 0 for i in range(len(sequence_names))]
                                    tga_per_100bp = [(batch_df.iloc[i]['Plus1_TGA_Count'] / sequence_lengths[i]) * 100 if sequence_lengths[i] > 0 else 0 for i in range(len(sequence_names))]

                                    # Create interactive stacked bar chart
                                    stops_data = {
                                        'TAA': taa_per_100bp,
                                        'TAG': tag_per_100bp,
                                        'TGA': tga_per_100bp
                                    }
                                    
                                    stops_fig = create_interactive_stacked_bar_chart(
                                        sequence_names,
                                        stops_data,
                                        '+1 Frame Stops per 100bp by Type',
                                        '+1 Frame Stops per 100bp'
                                    )
                                    st.plotly_chart(stops_fig, use_container_width=True)

                                    # Chart 2: Slippery Sites per 100bp broken down by TTTT and TTTC
                                    tttt_counts = []
                                    tttc_counts = []

                                    # Calculate specific slippery motifs for each sequence
                                    for i, (name, seq) in enumerate(sequences):
                                        slippery_breakdown = count_specific_slippery_motifs(seq)
                                        seq_length = len(seq) if len(seq) > 0 else 1
                                        tttt_per_100bp = (slippery_breakdown['TTTT'] / seq_length) * 100
                                        tttc_per_100bp = (slippery_breakdown['TTTC'] / seq_length) * 100
                                        tttt_counts.append(tttt_per_100bp)
                                        tttc_counts.append(tttc_per_100bp)

                                    # Create interactive stacked bar chart for slippery motifs
                                    slippery_data = {
                                        'TTTT': tttt_counts,
                                        'TTTC': tttc_counts
                                    }
                                    
                                    slippery_fig = create_interactive_stacked_bar_chart(
                                        sequence_names,
                                        slippery_data,
                                        'Slippery Sites per 100bp by Type',
                                        'Slippery Sites per 100bp'
                                    )
                                    st.plotly_chart(slippery_fig, use_container_width=True)

                                else:
                                    st.info("No +1 frame stop codons found in any sequence.")

                                # -1 Frame Analysis visualization
                                st.subheader("📊 Interactive Batch -1 Frame Analysis")

                                # Check if we have the required columns and valid data
                                required_cols = ['minus1_TAA_Count', 'minus1_TAG_Count', 'minus1_TGA_Count', 'minus1_Total_Stops']
                                if all(col in batch_df.columns for col in required_cols):

                                    # Overall statistics first
                                    total_taa = batch_df['minus1_TAA_Count'].sum()
                                    total_tag = batch_df['minus1_TAG_Count'].sum()
                                    total_tga = batch_df['minus1_TGA_Count'].sum()
                                    total_stops = total_taa + total_tag + total_tga

                                    # Summary statistics
                                    st.markdown("#### 📈 Overall Statistics")
                                    if gc_available:
                                        avg_gc = batch_df['GC_Content'].mean()
                                        col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
                                        with col_stat2:
                                            st.metric("Avg GC Content", f"{avg_gc:.1f}%")
                                    else:
                                        col_stat1, col_stat3, col_stat4, col_stat5 = st.columns(4)

                                    with col_stat1:
                                        st.metric("Total Sequences", len(sequences))
                                    with col_stat3:
                                        st.metric("Total -1 Stops", total_stops)
                                    with col_stat4:
                                        avg_stops = total_stops / len(sequences) if len(sequences) > 0 else 0
                                        st.metric("Avg Stops/Seq", f"{avg_stops:.1f}")
                                    with col_stat5:
                                        sequences_with_stops = len(batch_df[batch_df['minus1_Total_Stops'] > 0])
                                        st.metric("Seqs with Stops", f"{sequences_with_stops}/{len(sequences)}")


                                    # Individual sequence pie charts
                                    if total_stops > 0:
                                        st.markdown("#### 🥧 Individual Sequence Stop Codon Distribution")

                                        # Create pie charts for each sequence that has stops
                                        sequences_with_stops_data = batch_df[batch_df['minus1_Total_Stops'] > 0]

                                        if not sequences_with_stops_data.empty:
                                            # Create columns for pie charts (2 per row)
                                            cols_per_row = 2
                                            num_sequences = len(sequences_with_stops_data)

                                            for i in range(0, num_sequences, cols_per_row):
                                                cols = st.columns(cols_per_row)

                                                for j in range(cols_per_row):
                                                    if i + j < num_sequences:
                                                        seq_data = sequences_with_stops_data.iloc[i + j]
                                                        seq_name = seq_data['Sequence_Name']

                                                        taa_count = seq_data['minus1_TAA_Count']
                                                        tag_count = seq_data['minus1_TAG_Count']
                                                        tga_count = seq_data['minus1_TGA_Count']
                                                        total_seq_stops = seq_data['minus1_Total_Stops']

                                                        if total_seq_stops > 0:
                                                            with cols[j]:
                                                                # Filter out zero values
                                                                pie_data = []
                                                                pie_labels = []
                                                                pie_colors = []
                                                                color_map = {'TAA': '#FF6B6B', 'TAG': '#4ECDC4', 'TGA': '#45B7D1'}

                                                                for codon, count in [('TAA', taa_count), ('TAG', tag_count), ('TGA', tga_count)]:
                                                                    if count > 0:
                                                                        pie_data.append(count)
                                                                        pie_labels.append(codon)
                                                                        pie_colors.append(color_map[codon])

                                                                if pie_data:
                                                                    fig_individual = go.Figure(data=[go.Pie(
                                                                        labels=pie_labels,
                                                                        values=pie_data,
                                                                        hole=.4,
                                                                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                                                                        textinfo='label+value',
                                                                        textfont_size=10,
                                                                        marker=dict(
                                                                            colors=pie_colors,
                                                                            line=dict(color='#FFFFFF', width=2)
                                                                        )
                                                                    )])

                                                                    fig_individual.update_layout(
                                                                        title={
                                                                            'text': f'{seq_name[:20]}{"..." if len(seq_name) > 20 else ""}<br><sub>-1 Frame Stops</sub>',
                                                                            'x': 0.5,
                                                                            'font': {'size': 12}
                                                                        },
                                                                        annotations=[dict(
                                                                            text=f'{total_seq_stops}<br>Stops', 
                                                                            x=0.5, y=0.5, 
                                                                            font_size=11, 
                                                                            showarrow=False,
                                                                            font=dict(color="#2C3E50", weight="bold")
                                                                        )],
                                                                        height=300,
                                                                        showlegend=False,
                                                                        margin=dict(t=50, b=10, l=10, r=10)
                                                                    )

                                                                    st.plotly_chart(fig_individual, use_container_width=True)
                                        else:
                                            st.info("No sequences with -1 frame stops found for individual visualization.")

                                    if total_stops > 0:
                                        # Interactive summary charts with breakdown by stop codon type
                                        st.markdown("#### 📊 Interactive Summary Charts")

                                        # Chart 1: -1 Stops per 100bp broken down by TAA, TAG, TGA
                                        sequence_names = batch_df['Sequence_Name'].tolist()
                                        sequence_lengths = batch_df['Sequence_Length'].tolist() if 'Sequence_Length' in batch_df.columns else [1] * len(sequence_names)

                                        # Calculate stops per 100bp for each type
                                        taa_per_100bp = [(batch_df.iloc[i]['minus1_TAA_Count'] / sequence_lengths[i]) * 100 if sequence_lengths[i] > 0 else 0 for i in range(len(sequence_names))]
                                        tag_per_100bp = [(batch_df.iloc[i]['minus1_TAG_Count'] / sequence_lengths[i]) * 100 if sequence_lengths[i] > 0 else 0 for i in range(len(sequence_names))]
                                        tga_per_100bp = [(batch_df.iloc[i]['minus1_TGA_Count'] / sequence_lengths[i]) * 100 if sequence_lengths[i] > 0 else 0 for i in range(len(sequence_names))]

                                        # Create interactive stacked bar chart
                                        stops_data = {
                                            'TAA': taa_per_100bp,
                                            'TAG': tag_per_100bp,
                                            'TGA': tga_per_100bp
                                        }

                                        stops_fig = create_interactive_stacked_bar_chart(
                                            sequence_names,
                                            stops_data,
                                            '-1 Frame Stops per 100bp by Type',
                                            '-1 Frame Stops per 100bp'
                                        )
                                        st.plotly_chart(stops_fig, use_container_width=True)

                                    else:
                                        st.info("No -1 frame stop codons found in any sequence.")


                                else:
                                    st.warning("Analysis data not available for visualization.")

                                # Add the new graphs for batch analysis
                                st.subheader("📊 Interactive CAI and Stop Codon Analysis (Batch)")

                                for i, (name, seq) in enumerate(sequences):
                                    st.markdown(f"### 🧬 Analysis for: {name}")
                                    
                                    # Get CAI data
                                    cai_result, cai_error = run_single_optimization(seq, "In-Frame Analysis")
                                    if not cai_error and isinstance(cai_result, dict) and 'Position' in cai_result:
                                        cai_df = pd.DataFrame(cai_result)
                                        positions = cai_df['Position'].tolist()
                                        cai_weights = cai_df['CAI_Weight'].tolist()
                                        amino_acids = cai_df['Amino_Acid'].tolist()

                                        # Get stop codon positions
                                        plus1_stop_positions = get_plus1_stop_positions(seq)
                                        minus1_stop_positions = get_minus1_stop_positions(seq)

                                        # Create +1 stop codon plot
                                        if plus1_stop_positions:
                                            fig_plus1 = create_interactive_cai_stop_codon_plot(
                                                positions,
                                                cai_weights,
                                                amino_acids,
                                                plus1_stop_positions,
                                                name,
                                                "+1 Frame"
                                            )
                                            st.plotly_chart(fig_plus1, use_container_width=True)
                                        else:
                                            st.info(f"No +1 stop codons found in {name} to plot against CAI.")

                                        # Create -1 stop codon plot
                                        if minus1_stop_positions:
                                            fig_minus1 = create_interactive_cai_stop_codon_plot(
                                                positions,
                                                cai_weights,
                                                amino_acids,
                                                minus1_stop_positions,
                                                name,
                                                "-1 Frame"
                                            )
                                            st.plotly_chart(fig_minus1, use_container_width=True)
                                        else:
                                            st.info(f"No -1 stop codons found in {name} to plot against CAI.")
                                        
                                        st.divider()

                                    else:
                                        st.warning(f"Could not generate CAI data for {name}.")
                            else:
                                st.warning("Analysis data not available for visualization.")
                                
                                    
                    
                                
                        # Display results for other optimization methods with interactive charts
                        elif batch_method in ["Standard Codon Optimization", "Balanced Optimization", 
                                              "NC Stop Codon Optimization", "JT Plus1 Stop Optimization"]:
                            st.subheader(f"📊 Interactive Batch {batch_method} Results")
                            
                            # Check if we have optimization results
                            if 'Optimized_DNA' in batch_df.columns:
                                # Summary statistics
                                st.markdown("#### 📈 Optimization Summary")
                                
                                total_sequences = len(batch_df)
                                successful_optimizations = len(batch_df[batch_df['Optimized_DNA'].notna()])
                                
                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                with col_stat1:
                                    st.metric("Total Sequences", total_sequences)
                                with col_stat2:
                                    st.metric("Successful Optimizations", successful_optimizations)
                                with col_stat3:
                                    success_rate = (successful_optimizations / total_sequences * 100) if total_sequences > 0 else 0
                                    st.metric("Success Rate", f"{success_rate:.1f}%")
                                
                                # Display individual sequence results
                                st.markdown("#### 🧬 Individual Sequence Results")
                                
                                for idx, row in batch_df.iterrows():
                                    seq_name = row.get('Sequence_Name', f'Sequence_{idx+1}')
                                    
                                    with st.expander(f"📄 {seq_name}", expanded=False):
                                        if pd.notna(row.get('Optimized_DNA')):
                                            # Show sequence comparison
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                display_copyable_sequence(
                                                    row.get('Original_DNA', ''), 
                                                    "Original Sequence", 
                                                    f"batch_orig_{idx}"
                                                )
                                            
                                            with col2:
                                                display_copyable_sequence(
                                                    row.get('Optimized_DNA', ''), 
                                                    "Optimized Sequence", 
                                                    f"batch_opt_{idx}"
                                                )
                                            
                                            # Show metrics
                                            st.markdown("**📊 Optimization Metrics:**")

                                            # Create three columns for metrics
                                            metric_col1, metric_col2, metric_col3 = st.columns(3)

                                            with metric_col1:
                                                st.markdown("**🧬 Sequence Properties**")
                                                orig_len = len(row.get('Original_DNA', ''))
                                                opt_len = len(row.get('Optimized_DNA', ''))
                                                st.metric("Sequence Length", f"{orig_len} bp", delta=f"{opt_len - orig_len} bp" if opt_len != orig_len else None)
                                                
                                                if 'Protein' in row:
                                                    st.metric("Protein Length", f"{len(row['Protein'])} aa")

                                            with metric_col2:
                                                st.markdown("**🧪 GC & CAI Analysis**")
                                                
                                                # GC Content
                                                orig_gc = calculate_gc_content(row.get('Original_DNA', ''))
                                                opt_gc = calculate_gc_content(row.get('Optimized_DNA', ''))
                                                gc_change = opt_gc - orig_gc
                                                st.metric(
                                                    "GC Content", 
                                                    f"{opt_gc:.1f}%", 
                                                    delta=f"{gc_change:+.1f}%",
                                                    delta_color="inverse"  # Red if too high, green if moderate
                                                )
                                                
                                                # CAI Analysis
                                                orig_seq = row.get('Original_DNA', '')
                                                opt_seq = row.get('Optimized_DNA', '')
                                                if orig_seq and opt_seq:
                                                    orig_weights, _ = get_codon_weights_row(orig_seq)
                                                    opt_weights, _ = get_codon_weights_row(opt_seq)
                                                    orig_cai = sum(orig_weights) / len(orig_weights) if orig_weights else 0
                                                    opt_cai = sum(opt_weights) / len(opt_weights) if opt_weights else 0
                                                    cai_change = opt_cai - orig_cai
                                                    
                                                    st.metric(
                                                        "CAI Score", 
                                                        f"{opt_cai:.3f}", 
                                                        delta=f"{cai_change:+.3f}",
                                                        delta_color="normal"  # Green is good for CAI
                                                    )

                                            with metric_col3:
                                                st.markdown("**🛑 Stop Codon Analysis**")
                                                
                                                # +1 Frame stops
                                                orig_stops = number_of_plus1_stops(row.get('Original_DNA', ''))
                                                opt_stops = number_of_plus1_stops(row.get('Optimized_DNA', ''))
                                                stops_change = opt_stops['total'] - orig_stops['total']
                                                
                                                st.metric(
                                                    "+1 Frame Stops", 
                                                    f"{opt_stops['total']}", 
                                                    delta=f"{stops_change:+d}",
                                                    delta_color="inverse"  # Red if increased, green if decreased
                                                )
                                                
                                                # Show stop codon breakdown if there are stops
                                                if opt_stops['total'] > 0:
                                                    st.caption(f"TAA: {opt_stops['TAA']}, TAG: {opt_stops['TAG']}, TGA: {opt_stops['TGA']}")
                                        
                                        else:
                                            if 'Error' in row and pd.notna(row['Error']):
                                                st.error(f"Error: {row['Error']}")
                                            else:
                                                st.warning("No optimization results available")
                                
                                # Interactive summary comparison charts
                                if successful_optimizations > 0:
                                    st.markdown("#### 📊 Interactive Optimization Impact Analysis")
                                    
                                    # Calculate metrics for all sequences
                                    metrics_data = []
                                    for idx, row in batch_df.iterrows():
                                        if pd.notna(row.get('Optimized_DNA')):
                                            orig_seq = row.get('Original_DNA', '')
                                            opt_seq = row.get('Optimized_DNA', '')
                                            
                                            if orig_seq and opt_seq:
                                                # Calculate all metrics
                                                orig_stops = number_of_plus1_stops(orig_seq)
                                                opt_stops = number_of_plus1_stops(opt_seq)
                                                
                                                # Calculate CAI
                                                orig_weights, _ = get_codon_weights_row(orig_seq)
                                                opt_weights, _ = get_codon_weights_row(opt_seq)
                                                orig_avg_cai = sum(orig_weights) / len(orig_weights) if orig_weights else 0
                                                opt_avg_cai = sum(opt_weights) / len(opt_weights) if opt_weights else 0
                                                
                                                metrics_data.append({
                                                    'Sequence': row.get('Sequence_Name', f'Seq_{idx+1}'),
                                                    'Original_Stops': orig_stops['total'],
                                                    'Optimized_Stops': opt_stops['total'],
                                                    'Stop_Change': opt_stops['total'] - orig_stops['total'],
                                                    'Original_GC': calculate_gc_content(orig_seq),
                                                    'Optimized_GC': calculate_gc_content(opt_seq),
                                                    'Original_CAI': orig_avg_cai,
                                                    'Optimized_CAI': opt_avg_cai,
                                                    'CAI_Change': opt_avg_cai - orig_avg_cai
                                                })
                                    
                                    if metrics_data:
                                        metrics_df = pd.DataFrame(metrics_data)
                                        
                                        # Create interactive comparison charts
                                        col_chart1, col_chart2 = st.columns(2)
                                        
                                        with col_chart1:
                                            # +1 Frame Stops Comparison
                                            stops_comparison_fig = create_interactive_comparison_chart(
                                                metrics_df['Sequence'].tolist(),
                                                metrics_df['Original_Stops'].tolist(),
                                                metrics_df['Optimized_Stops'].tolist(),
                                                '+1 Frame Stops',
                                                'Number of Stops'
                                            )
                                            st.plotly_chart(stops_comparison_fig, use_container_width=True)
                                        
                                        with col_chart2:
                                            # GC Content Comparison
                                            gc_comparison_fig = create_interactive_comparison_chart(
                                                metrics_df['Sequence'].tolist(),
                                                metrics_df['Original_GC'].tolist(),
                                                metrics_df['Optimized_GC'].tolist(),
                                                'GC Content',
                                                'GC Content (%)'
                                            )
                                            st.plotly_chart(gc_comparison_fig, use_container_width=True)
                                        
                                        # CAI Comparison
                                        if 'Original_CAI' in metrics_df.columns and 'Optimized_CAI' in metrics_df.columns:
                                            st.markdown("#### 📊 Interactive CAI Comparison")
                                            
                                            cai_comparison_fig = create_interactive_comparison_chart(
                                                metrics_df['Sequence'].tolist(),
                                                metrics_df['Original_CAI'].tolist(),
                                                metrics_df['Optimized_CAI'].tolist(),
                                                'CAI Score',
                                                'CAI (Codon Adaptation Index)'
                                            )
                                            st.plotly_chart(cai_comparison_fig, use_container_width=True)
                                        
                                        # Summary statistics table - UPDATED
                                        st.markdown("#### 📋 Optimization Summary Report")

                                        # Create summary metrics in a more visual way - UPDATED
                                        summary_col1, summary_col2, summary_col3 = st.columns(3)

                                        # Calculate all averages
                                        avg_orig_cai = metrics_df['Original_CAI'].mean()
                                        avg_opt_cai = metrics_df['Optimized_CAI'].mean()
                                        cai_improvement = ((avg_opt_cai - avg_orig_cai) / avg_orig_cai) * 100 if avg_orig_cai > 0 else 0
                                        avg_stops_change = metrics_df['Stop_Change'].mean()  # Changed from reduction to change
                                        total_stops_changed = metrics_df['Stop_Change'].sum()  # Changed from removed to changed
                                        avg_gc_change = (metrics_df['Optimized_GC'] - metrics_df['Original_GC']).mean()

                                        with summary_col1:
                                            st.markdown("**🎯 CAI Performance**")
                                            st.metric("Original Avg CAI", f"{avg_orig_cai:.3f}")
                                            st.metric("Optimized Avg CAI", f"{avg_opt_cai:.3f}")
                                            st.metric(
                                                "CAI Improvement", 
                                                f"{cai_improvement:.1f}%",
                                                delta=f"{cai_improvement:.1f}%",
                                                delta_color="normal"
                                            )

                                        with summary_col2:
                                            st.markdown("**🛑 Stop Codon Changes**")  # Updated label
                                            st.metric("Avg Stops Changed", f"{avg_stops_change:.1f}")  # Updated metric
                                            st.metric("Total Stops Changed", f"{total_stops_changed}")  # Updated metric

                                        with summary_col3:
                                            st.markdown("**🧬 GC Content Changes**")
                                            st.metric(
                                                "Avg GC Change", 
                                                f"{avg_gc_change:+.1f}%",
                                                delta=f"{avg_gc_change:+.1f}%",
                                                delta_color="inverse"
                                            )
                                            best_cai_seq = metrics_df.loc[metrics_df['CAI_Change'].idxmax(), 'Sequence']
                                            st.metric("Best CAI Improvement", f"{best_cai_seq[:15]}...")

                                        # Add a detailed breakdown table
                                        st.markdown("#### 📊 Detailed Sequence Metrics")
                                        display_df = metrics_df[['Sequence', 'Original_CAI', 'Optimized_CAI', 'CAI_Change', 
                                                                'Original_Stops', 'Optimized_Stops', 'Stop_Change',  # Updated column name
                                                                'Original_GC', 'Optimized_GC']].copy()

                                        # Format the dataframe for display
                                        display_df['CAI_Change'] = display_df['CAI_Change'].apply(lambda x: f"{x:+.3f}")
                                        display_df['Stop_Change'] = display_df['Stop_Change'].apply(lambda x: f"{x:+d}")  # Updated column formatting
                                        display_df['Original_GC'] = display_df['Original_GC'].apply(lambda x: f"{x:.1f}%")
                                        display_df['Optimized_GC'] = display_df['Optimized_GC'].apply(lambda x: f"{x:.1f}%")
                                        display_df['Original_CAI'] = display_df['Original_CAI'].apply(lambda x: f"{x:.3f}")
                                        display_df['Optimized_CAI'] = display_df['Optimized_CAI'].apply(lambda x: f"{x:.3f}")

                                        # Rename columns for better display
                                        display_df.columns = ['Sequence', 'Orig CAI', 'Opt CAI', 'CAI Δ', 
                                                            'Orig Stops', 'Opt Stops', 'Stops Δ',  # Updated column name
                                                            'Orig GC', 'Opt GC']

                                        st.dataframe(
                                            display_df,
                                            use_container_width=True,
                                            hide_index=True,
                                            column_config={
                                                "Sequence": st.column_config.TextColumn("Sequence", width="medium"),
                                                "CAI Δ": st.column_config.TextColumn("CAI Δ", help="Change in CAI score"),
                                                "Stops Δ": st.column_config.TextColumn("Stops Δ", help="Change in stop codons")  # Updated help text
                                            }
                                        )
                            
                            else:
                                st.warning("No optimization results found in the batch data.")
                            
                            # Display the data table at the end
                            st.markdown("#### 📋 Complete Results Table")
                            st.dataframe(batch_df, use_container_width=True)
                        
                        # Add accumulation option for batch results
                        st.divider()
                        accumulate_batch = st.checkbox("Accumulate Batch Results", help="Add these batch results to accumulated collection")

                        if accumulate_batch and results:
                            # Add batch ID and timestamp
                            batch_id = f"Batch_{len(st.session_state.batch_accumulated_results) + 1}"
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            for result in results:
                                result['Batch_ID'] = batch_id
                                result['Timestamp'] = timestamp
                                st.session_state.batch_accumulated_results.append(result)
                            
                            st.success(f"Batch results added to accumulation (Total batches: {len(set([r['Batch_ID'] for r in st.session_state.batch_accumulated_results]))})")

                        # Download button
                        excel_data = create_download_link(batch_df, f"Batch_{batch_method}_{len(sequences)}_sequences.xlsx")
                        st.download_button(
                            label="Download Batch Results (Excel)",
                            data=excel_data,
                            file_name=f"Batch_{batch_method}_{len(sequences)}_sequences.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.warning("No valid results generated from batch processing")
        
        # Display accumulated batch results
        if st.session_state.batch_accumulated_results:
            st.divider()
            st.subheader("📚 Accumulated Batch Results")
            
            with st.expander(f"View All Accumulated Results ({len(st.session_state.batch_accumulated_results)} sequences from {len(set([r['Batch_ID'] for r in st.session_state.batch_accumulated_results]))} batches)", expanded=False):
                acc_batch_df = pd.DataFrame(st.session_state.batch_accumulated_results)
                
                # Reorder columns
                priority_cols = ['Batch_ID', 'Timestamp', 'Sequence_Name', 'Method']
                other_cols = [col for col in acc_batch_df.columns if col not in priority_cols]
                acc_batch_df = acc_batch_df[priority_cols + other_cols]
                
                st.dataframe(acc_batch_df, use_container_width=True)
                
                # Download accumulated results
                excel_data = create_download_link(acc_batch_df, f"Accumulated_Batch_Results_{len(st.session_state.batch_accumulated_results)}_sequences.xlsx")
                st.download_button(
                    label="Download All Accumulated Results (Excel)",
                    data=excel_data,
                    file_name=f"Accumulated_Batch_Results_{len(st.session_state.batch_accumulated_results)}_sequences.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                if st.button("Clear Accumulated Batch Results"):
                    st.session_state.batch_accumulated_results = []
                    st.rerun()
                    
            if sequences and batch_file is None:  # File was uploaded but no sequences found
                st.warning("No valid sequences found in uploaded file. Please check the file format.")
                    
    with tab3:
        st.header("CDS Database Search")
        st.markdown("🎯 **Search for specific proteins and extract their CDS sequences**")
        
        # Step 1: UniProt Search
        st.markdown("#### Step 1: Search UniProt for Proteins")
        
        protein_query = st.text_input(
            "Enter protein search (e.g., 'SARS-CoV-2 spike protein', 'human insulin'):",
            placeholder="SARS-CoV-2 spike protein",
            key="protein_search_query"
        )
        
        col_search1, col_search2 = st.columns([2, 1])
        with col_search1:
            search_protein_btn = st.button("🔍 Search UniProt", type="primary")
        with col_search2:
            max_uniprot_results = st.slider("Max results", 5, 20, 10, key="max_uniprot_cds")
        
        # Initialize session state for CDS workflow
        if 'uniprot_results' not in st.session_state:
            st.session_state.uniprot_results = []
        if 'selected_uniprot_entry' not in st.session_state:
            st.session_state.selected_uniprot_entry = None
        if 'ncbi_details' not in st.session_state:
            st.session_state.ncbi_details = None
        if 'cds_options' not in st.session_state:
            st.session_state.cds_options = []
        
        # Step 1: UniProt Search Results
        if search_protein_btn and protein_query.strip():
            with st.spinner("Searching UniProt..."):
                try:
                    results = st.session_state.uniprot_engine.search_protein_sequences(protein_query, max_uniprot_results)
                    st.session_state.uniprot_results = results
                    # Reset other states
                    st.session_state.selected_uniprot_entry = None
                    st.session_state.ncbi_details = None
                    st.session_state.cds_options = []
                    
                    if results:
                        st.success(f"✅ Found {len(results)} UniProt entries")
                    else:
                        st.warning("No UniProt entries found. Try different search terms.")
                        
                except Exception as e:
                    st.error(f"Error searching UniProt: {str(e)}")
        
        # Step 2: Display UniProt Results and Selection
        if st.session_state.uniprot_results:
            st.markdown("#### Step 2: Select a Protein Entry")
            
            # Create selection dropdown
            uniprot_options = []
            for i, entry in enumerate(st.session_state.uniprot_results):
                option_text = f"{entry['accession']} - {entry['protein_name']} [{entry['organism']}]"
                uniprot_options.append(option_text)
            
            selected_uniprot_idx = st.selectbox(
                "Choose a UniProt entry:",
                range(len(uniprot_options)),
                format_func=lambda x: uniprot_options[x],
                key="uniprot_selection"
            )
            
            selected_entry = st.session_state.uniprot_results[selected_uniprot_idx]
            st.session_state.selected_uniprot_entry = selected_entry
            
            # Display selected entry details
            with st.expander("📋 Selected Entry Details", expanded=True):
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**UniProt ID:** {selected_entry['accession']}")
                    st.write(f"**Protein:** {selected_entry['protein_name']}")
                    st.write(f"**Genes:** {selected_entry['gene_names']}")
                with col_info2:
                    st.write(f"**Organism:** {selected_entry['organism']}")
                    st.write(f"**Length:** {selected_entry['length']} aa")
                    st.write(f"**Reviewed:** {'Yes' if selected_entry['reviewed'] else 'No'}")
                
                # Show nucleotide cross-references
                if selected_entry['nucleotide_refs']:
                    st.markdown("**🔗 Nucleotide Cross-References:**")
                    for ref in selected_entry['nucleotide_refs']:
                        st.write(f"- **{ref['database']}:** {ref['id']}")
            
            # Step 3: Get NCBI Details
            if selected_entry['nucleotide_refs']:
                st.markdown("#### Step 3: Retrieve NCBI CDS Information")
                
                get_ncbi_btn = st.button("🧬 Get CDS from NCBI", type="primary")
                
                if get_ncbi_btn:
                    # Try each nucleotide reference
                    for ref in selected_entry['nucleotide_refs']:
                        if ref['database'] in ['EMBL', 'RefSeq']:
                            accession = ref['id']
                            
                            with st.spinner(f"Retrieving CDS information for {accession}..."):
                                try:
                                    ncbi_details = st.session_state.ncbi_engine.scrape_ncbi_page(accession, protein_query)
                                    
                                    if ncbi_details['success'] and ncbi_details['cds_sequences']:
                                        st.session_state.ncbi_details = ncbi_details
                                        st.session_state.cds_options = ncbi_details['cds_sequences']
                                        st.success(f"✅ Retrieved {len(ncbi_details['cds_sequences'])} CDS features")
                                        break
                                    else:
                                        st.warning(f"⚠️ No CDS found in {accession}")
                                        
                                except Exception as e:
                                    st.error(f"Error retrieving {accession}: {str(e)}")
                                    continue
                    
                    if not st.session_state.cds_options:
                        st.error("❌ Could not retrieve CDS information from any cross-reference")
        
        # Step 4: CDS Selection and Extraction
        if st.session_state.cds_options:
            st.markdown("#### Step 4: Select CDS to Extract")
            
            # Create CDS dropdown options
            cds_dropdown_options = []
            for i, cds in enumerate(st.session_state.cds_options):
                gene_name = cds.get('gene_name', 'Unknown')
                product = cds.get('product', cds.get('protein_name', 'Unknown'))
                positions = f"{cds.get('start_position', 0)}-{cds.get('end_position', 0)}"
                length = cds.get('length', 0)
                
                option_text = f"{gene_name} | {product} | {positions} | {length} bp"
                cds_dropdown_options.append(option_text)
            
            selected_cds_idx = st.selectbox(
                "Choose a CDS feature:",
                range(len(cds_dropdown_options)),
                format_func=lambda x: cds_dropdown_options[x],
                key="cds_selection"
            )
            
            selected_cds = st.session_state.cds_options[selected_cds_idx]
            
            # Display selected CDS details
            with st.expander("🧬 Selected CDS Details", expanded=True):
                col_cds1, col_cds2, col_cds3 = st.columns(3)
                with col_cds1:
                    st.write(f"**Gene:** {selected_cds.get('gene_name', 'N/A')}")
                    st.write(f"**Product:** {selected_cds.get('product', 'N/A')}")
                    st.write(f"**Locus Tag:** {selected_cds.get('locus_tag', 'N/A')}")
                with col_cds2:
                    st.write(f"**Position:** {selected_cds.get('start_position', 0)}-{selected_cds.get('end_position', 0)}")
                    st.write(f"**Length:** {selected_cds.get('length', 0)} bp")
                    st.write(f"**Valid DNA:** {selected_cds.get('valid_dna', False)}")
                with col_cds3:
                    if selected_cds.get('url'):
                        st.markdown(f"**[🔗 View NCBI Record]({selected_cds['url']})**")
            
            # Step 5: Show extracted sequence
            if selected_cds.get('sequence') and selected_cds.get('valid_dna'):
                st.markdown("#### Step 5: Extracted CDS Sequence")
                
                sequence = selected_cds['sequence']
                header = selected_cds.get('header', f">{selected_cds.get('accession', 'Unknown')}")
                
            
                display_copyable_sequence(
                    sequence,
                    "CDS DNA Sequence:",
                    "extracted_cds"
                )
                
                # Analysis and action buttons
                col_action1, col_action2, col_action3 = st.columns(3)
                
                with col_action1:
                    if st.button("🔬 Quick Analysis", key="cds_quick_analysis"):
                        try:
                            protein = translate_dna(sequence)
                            plus1_stops = number_of_plus1_stops(sequence)
                            slippery = number_of_slippery_motifs(sequence)
                            weights, _ = get_codon_weights_row(sequence)
                            avg_cai = sum(weights) / len(weights) if weights else 0
                            
                            st.markdown("**🔬 CDS Analysis Results:**")
                            anal_col1, anal_col2, anal_col3, anal_col4 = st.columns(4)
                            with anal_col1:
                                st.metric("DNA Length", f"{len(sequence)} bp")
                            with anal_col2:
                                st.metric("Protein Length", f"{len(protein)} aa")
                            with anal_col3:
                                st.metric("+1 Frame Stops", plus1_stops['total'])
                            with anal_col4:
                                st.metric("Avg CAI", f"{avg_cai:.3f}")
                            
                            with st.expander("Translated Protein Sequence"):
                                st.text_area("Protein:", protein, height=100, key="cds_protein_result")
                                
                        except Exception as e:
                            st.error(f"Analysis error: {e}")
                
                with col_action2:
                    if st.button("⚡ Send to Optimizer", key="send_cds_to_optimizer"):
                        # Store the sequence for transfer to Single Sequence tab
                        st.session_state.transfer_sequence = sequence
                        st.session_state.transfer_sequence_info = {
                            'source': f"{selected_cds.get('gene_name', 'Unknown')} from {selected_cds.get('accession', 'Unknown')}",
                            'accession': selected_cds.get('accession', 'Unknown'),
                            'protein_name': selected_cds.get('product', 'Unknown'),
                            'length': len(sequence),
                            'positions': f"{selected_cds.get('start_position', 0)}-{selected_cds.get('end_position', 0)}"
                        }
                        st.success("✅ Sequence sent! Check the 'Single Sequence' tab.")
                        st.rerun()
                
                with col_action3:
                    # Download as FASTA
                    fasta_content = f"{header}\n{sequence}"
                    st.download_button(
                        "💾 Download FASTA",
                        fasta_content,
                        file_name=f"{selected_cds.get('gene_name', 'cds')}_{selected_cds.get('accession', 'unknown')}.fasta",
                        mime="text/plain",
                        help="Download CDS sequence in FASTA format"
                    )
                
                # Create comprehensive download data
                st.markdown("#### 📊 Complete Analysis Report")
                
                # Create detailed DataFrame for download
                cds_analysis_data = {
                    'UniProt_Accession': [st.session_state.selected_uniprot_entry['accession']],
                    'UniProt_Protein': [st.session_state.selected_uniprot_entry['protein_name']],
                    'UniProt_Organism': [st.session_state.selected_uniprot_entry['organism']],
                    'UniProt_Genes': [st.session_state.selected_uniprot_entry['gene_names']],
                    'NCBI_Accession': [selected_cds.get('accession', '')],
                    'CDS_Gene_Name': [selected_cds.get('gene_name', '')],
                    'CDS_Product': [selected_cds.get('product', '')],
                    'CDS_Locus_Tag': [selected_cds.get('locus_tag', '')],
                    'CDS_Start_Position': [selected_cds.get('start_position', 0)],
                    'CDS_End_Position': [selected_cds.get('end_position', 0)],
                    'CDS_Length_bp': [selected_cds.get('length', 0)],
                    'CDS_Sequence': [sequence],
                    'CDS_Header': [header],
                    'NCBI_URL': [selected_cds.get('url', '')],
                    'UniProt_URL': [st.session_state.selected_uniprot_entry['uniprot_url']],
                    'Valid_DNA': [selected_cds.get('valid_dna', False)],
                    'Search_Query': [protein_query]
                }
                
                analysis_df = pd.DataFrame(cds_analysis_data)
                
                # Download button for analysis
                excel_data = create_download_link(analysis_df, f"CDS_Analysis_{selected_cds.get('gene_name', 'unknown')}.xlsx")
                st.download_button(
                    label="📥 Download Complete Analysis (Excel)",
                    data=excel_data,
                    file_name=f"CDS_Analysis_{selected_cds.get('gene_name', 'unknown')}_{selected_cds.get('accession', 'unknown')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download complete analysis including UniProt and NCBI information"
                )
            
            elif selected_cds.get('sequence') and not selected_cds.get('valid_dna'):
                st.warning("⚠️ This CDS contains invalid DNA characters and cannot be used for codon optimization.")
                st.text_area("Raw sequence (for reference only):", selected_cds['sequence'], height=100)
            
            else:
                st.error("❌ No sequence data available for this CDS feature.")
        
        # Help section for new workflow
        with st.expander("ℹ️ How to use the CDS Search", expanded=False):
            st.markdown("""
            **🎯 Targeted CDS Extraction Workflow:**
            
            **Step 1 - UniProt Search:**
            - Search for proteins using descriptive terms
            - Examples: "SARS-CoV-2 spike protein", "human insulin", "green fluorescent protein"
            - UniProt provides comprehensive protein information and cross-references
            
            **Step 2 - Select Protein:**
            - Choose the most relevant protein from the search results
            - Review organism, gene names, and protein details
            - Check for nucleotide database cross-references (EMBL/RefSeq)
            
            **Step 3 - Retrieve NCBI Data:**
            - Automatically fetch the complete genomic/nucleotide record
            - Parse GenBank format to extract all CDS features
            - Identify coding sequences with their exact positions
            
            **Step 4 - Select Specific CDS:**
            - Choose from all available CDS features in the record
            - Each option shows: Gene name | Product | Positions | Length
            - Example: "S | spike glycoprotein | 21563-25384 | 3822 bp"
            
            **Step 5 - Extract & Analyze:**
            - Get the exact DNA sequence from the specified genomic positions
            - Perform quick analysis (CAI, protein translation, +1 frame analysis)
            - Transfer directly to Single Sequence Optimization
            - Download FASTA or complete Excel analysis report
            
            **Key Features:**
            - **Precise Extraction**: Gets exact CDS sequence from genomic positions
            - **Multiple CDS Options**: Choose from all coding sequences in the record
            - **Cross-Database Integration**: Links UniProt proteins to NCBI nucleotide data
            - **Quality Validation**: Checks for valid DNA sequences before optimization
            - **Seamless Transfer**: Direct integration with codon optimization tools
            """) 
    
    with tab4:
        st.header("Patent Search")
        st.markdown("Search for patents related to DNA, RNA, codon optimization, and molecular biology technologies")
        
        # Check API configuration
        col_status1, col_status2 = st.columns(2)
        with col_status1:
            serper_status = "✅ Connected" if st.session_state.patent_engine.serper_api_key else "❌ Not configured"
            st.info(f"**SERPER API:** {serper_status}")
        with col_status2:
            anthropic_status = "✅ Connected" if st.session_state.patent_engine.anthropic else "❌ Not configured"
            st.info(f"**Anthropic API:** {anthropic_status}")
        
        if not st.session_state.patent_engine.serper_api_key:
            st.warning("Please configure SERPER_API_KEY in your .env file to use patent search")
            st.code("SERPER_API_KEY=your_serper_api_key_here")
        else:
            # Add connection test button
            if st.button("Test SERPER API Connection", key="test_patent_connection"):
                with st.spinner("Testing connection..."):
                    test_result = test_serper_connection(st.session_state.patent_engine.serper_api_key)
                    if test_result["success"]:
                        st.success("✅ SERPER API connection successful!")
                    else:
                        st.error(f"❌ Connection failed: {test_result['error']}")
                        st.info("Troubleshooting:\n- Check your SERPER API key\n- Verify internet connection\n- Try again in a few moments")
        
        patent_query = st.text_area(
            "Enter your patent search query:",
            placeholder="""Examples:
    - Codon optimization methods for protein expression
    - mRNA vaccine delivery systems
    - CRISPR gene editing technologies
    - DNA sequence analysis algorithms
    - Protein folding prediction methods
    - Nucleotide sequence modifications""",
            height=100,
            key="patent_search_query"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_patents_btn = st.button("Search Patents", type="primary", disabled=not st.session_state.patent_engine.serper_api_key)
        with col2:
            num_patents = st.slider("Number of results", 5, 20, 10, key="patent_num_results")
        
        if search_patents_btn and patent_query.strip():
            with st.spinner("Searching patents..."):
                search_results = st.session_state.patent_engine.search_patents(patent_query, num_patents)
                
                if search_results:
                    patents = st.session_state.patent_engine.extract_patent_info(search_results)
                    
                    # Extract mRNA/nucleotide sequences from patents
                    sequence_findings = st.session_state.patent_engine.extract_mrna_sequences_from_patents(patents)
                    
                    # Prepare context for AI analysis
                    patent_context = "\n\n".join([
                        f"Patent: {p['title']}\nID: {p['patent_id']}\nSummary: {p['snippet']}\nLink: {p['link']}"
                        for p in patents
                    ])
                    
                    # AI Analysis
                    if st.session_state.patent_engine.anthropic:
                        st.subheader("AI Analysis")
                        with st.spinner("Generating AI analysis..."):
                            ai_response = st.session_state.patent_engine.generate_ai_analysis(patent_query, patent_context)
                            st.markdown(ai_response)
                    
                    # Sequence Analysis Section - NEW
                    if sequence_findings:
                        st.subheader("🧬 mRNA/Nucleotide Sequence Analysis")
                        
                        # Display sequence findings
                        st.markdown(f"**Found {len(sequence_findings)} patents with potential mRNA/nucleotide sequences:**")
                        
                        for i, finding in enumerate(sequence_findings[:5], 1):  # Show top 5
                            with st.expander(f"🔬 Patent {i}: {finding['title'][:60]}... (Score: {finding['confidence_score']})", expanded=i <= 2):
                                col_seq1, col_seq2 = st.columns([2, 1])
                                
                                with col_seq1:
                                    st.write(f"**Patent ID:** {finding['patent_id']}")
                                    st.write(f"**Title:** {finding['title']}")
                                    
                                    if finding['sequences_found']:
                                        st.markdown("**🧬 Sequences Found:**")
                                        for seq in finding['sequences_found']:
                                            st.write(f"- **Type:** {seq['type']} ({seq['confidence']} confidence)")
                                            st.write(f"- **Length:** {seq['length']} bp")
                                            
                                            # Display sequence with copy functionality
                                            if len(seq['sequence']) <= 200:
                                                display_copyable_sequence(
                                                    seq['sequence'], 
                                                    f"Sequence ({seq['type']})", 
                                                    f"patent_seq_{i}_{seq['type']}"
                                                )
                                            else:
                                                st.text_area(
                                                    f"Sequence ({seq['type']}) - First 200 characters:",
                                                    seq['sequence'][:200] + "...",
                                                    height=80,
                                                    key=f"patent_preview_{i}_{seq['type']}"
                                                )
                                                
                                                # Full sequence in expandable section
                                                with st.expander("View Full Sequence"):
                                                    display_copyable_sequence(
                                                        seq['sequence'], 
                                                        "Full Sequence", 
                                                        f"patent_full_{i}_{seq['type']}"
                                                    )
                                            
                                            # Transfer to optimizer button
                                            if st.button(f"🚀 Send to Optimizer", key=f"transfer_patent_{i}_{seq['type']}"):
                                                st.session_state.transfer_sequence = seq['sequence']
                                                st.session_state.transfer_sequence_info = {
                                                    'source': f"Patent {finding['patent_id']} ({seq['type']})",
                                                    'patent_id': finding['patent_id'],
                                                    'sequence_type': seq['type'],
                                                    'length': seq['length'],
                                                    'confidence': seq['confidence']
                                                }
                                                st.success("✅ Sequence sent! Check the 'Single Sequence' tab.")
                                                st.rerun()
                                
                                with col_seq2:
                                    st.markdown("**📊 Analysis Details:**")
                                    st.write(f"**Confidence Score:** {finding['confidence_score']}")
                                    st.write(f"**Sequences Found:** {len(finding['sequences_found'])}")
                                    
                                    if finding['mrna_indicators']:
                                        st.write("**mRNA Indicators:**")
                                        for indicator in finding['mrna_indicators'][:5]:  # Show first 5
                                            st.write(f"- {indicator}")
                                    
                                    st.link_button("🔗 View Patent", finding['link'], use_container_width=True)
                        
                        # Generate AI sequence analysis
                        if st.session_state.patent_engine.anthropic:
                            st.subheader("🤖 AI Sequence Analysis")
                            with st.spinner("Analyzing sequences with AI..."):
                                sequence_analysis = st.session_state.patent_engine.generate_sequence_analysis(patent_query, sequence_findings)
                                st.markdown(sequence_analysis)
                        
                        # Create downloadable summary
                        st.subheader("📥 Download Sequence Findings")
                        
                        # Prepare data for download
                        download_data = []
                        for finding in sequence_findings:
                            base_info = {
                                'Patent_ID': finding['patent_id'],
                                'Patent_Title': finding['title'],
                                'Patent_Link': finding['link'],
                                'Confidence_Score': finding['confidence_score'],
                                'mRNA_Indicators': ', '.join(finding['mrna_indicators']),
                                'Total_Sequences': len(finding['sequences_found'])
                            }
                            
                            if finding['sequences_found']:
                                for i, seq in enumerate(finding['sequences_found']):
                                    row = base_info.copy()
                                    row.update({
                                        'Sequence_Number': i + 1,
                                        'Sequence_Type': seq['type'],
                                        'Sequence_Length': seq['length'],
                                        'Sequence_Confidence': seq['confidence'],
                                        'Sequence_Data': seq['sequence']
                                    })
                                    download_data.append(row)
                            else:
                                row = base_info.copy()
                                row.update({
                                    'Sequence_Number': 0,
                                    'Sequence_Type': 'None',
                                    'Sequence_Length': 0,
                                    'Sequence_Confidence': 'N/A',
                                    'Sequence_Data': ''
                                })
                                download_data.append(row)
                        
                        if download_data:
                            download_df = pd.DataFrame(download_data)
                            excel_data = create_download_link(download_df, f"Patent_Sequences_{len(sequence_findings)}_patents.xlsx")
                            st.download_button(
                                label="📥 Download Sequence Analysis (Excel)",
                                data=excel_data,
                                file_name=f"Patent_Sequences_{len(sequence_findings)}_patents.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Download complete analysis of found sequences"
                            )
                    
                    # Patent Results
                    st.subheader(f"Patent Search Results ({len(patents)} found)")
                    
                    for i, patent in enumerate(patents, 1):
                        with st.expander(f"Patent {i}: {patent['title'][:80]}..."):
                            col_info, col_link = st.columns([3, 1])
                            
                            with col_info:
                                st.write(f"**Patent ID:** {patent['patent_id']}")
                                st.write(f"**Title:** {patent['title']}")
                                st.write(f"**Summary:** {patent['snippet']}")
                            
                            with col_link:
                                st.link_button("View Patent", patent['link'], use_container_width=True)
                else:
                    st.warning("No patents found for your query. Try different keywords.")
    
    with tab5:
        st.header("About")
        st.markdown("""
        ### DNA Codon Optimization Tool v2.5
        
        This professional bioinformatics application provides comprehensive DNA sequence optimization and analysis capabilities with enhanced multi-database search functionality.
        
        **Available Methods:**
        - **Standard Codon Optimization**: Uses most frequent codons for each amino acid
        - **In-Frame Analysis**: Calculates Codon Adaptation Index for sequence assessment with interactive 10bp GC content window
        - **Balanced Optimization**: Advanced algorithm considering codon usage and +1 frame effects
        - **NC Stop Codon Optimization**: Specialized for alternative reading frame stop codon creation
        - **JT Plus1 Stop Optimization**: Creates specific stop motifs in +1 frame
        - **+1 Frame Analysis**: Comprehensive analysis including slippery motifs and frame analysis with interactive visualizations
        
        **Enhanced Features in v2.5:**
        - **Interactive Visualizations**: All charts now use Plotly for interactive exploration
        - **10bp GC Content Window**: Added real-time GC content analysis in sliding windows for In-Frame Analysis
        - **Dual-Axis Plots**: CAI weights and GC content displayed simultaneously with hover details
        - **Interactive Batch Charts**: Hover, zoom, and explore batch analysis results
        - **Enhanced Copy Functionality**: Improved sequence copying with better UX
        - **Consistent Color Schemes**: Professional two-shade palette throughout the application
        
        **Research & Database Features:**
        - **Patent Search**: AI-powered search of Google Patents for molecular biology technologies
        - **NCBI CDS Extraction**: Automated search and extraction of coding sequences from NCBI
        - **UniProt Integration**: Search protein database and retrieve linked nucleotide sequences
        - **Multi-Database Search**: Simultaneous search across NCBI and UniProt databases
        - **AI-Powered Analysis**: Anthropic Claude integration for intelligent sequence ranking
        - **Cross-Reference Mining**: Automatic retrieval of nucleotide sequences from protein databases
        - **Comprehensive Export**: Professional Excel files with multiple sheets and metadata
        
        **Interactive Features:**
        - **Hover Information**: Detailed tooltips on all charts showing exact values
        - **Zoom and Pan**: Interactive exploration of large datasets
        - **Click to Select**: Interactive data point selection where applicable
        - **Responsive Design**: Charts adapt to different screen sizes
        - **Real-time Updates**: Interactive controls update visualizations instantly
        
        **Core Features:**
        - Single sequence and batch processing
        - Result accumulation and export
        - Professional Excel output with formatting
        - Real-time validation and feedback
        - Configurable algorithm parameters
        - Multi-source sequence integration
        - Quote-based filtering for precise searches
        
        **Database Coverage:**
        - **NCBI**: Direct access to nucleotide sequences and CDS features
        - **UniProt**: Protein sequences with nucleotide cross-references
        - **EMBL**: European nucleotide database integration
        - **RefSeq**: Reference sequence collection
        - **Google Patents**: Intellectual property and technology search
        
        **API Requirements:**
        - SERPER API key for Google-based searches (NCBI, Patents)
        - Anthropic API key for AI analysis (optional but recommended)
        - UniProt REST API (free, no key required)
        
        **API Configuration:**
        Create a `.env` file in your application directory:
        ```
        SERPER_API_KEY=your_serper_api_key_here
        ANTHROPIC_API=your_anthropic_api_key_here
        ```
        
        **Version History:**
        - **v2.5**: Interactive visualizations, 10bp GC window analysis, enhanced user experience
        - **v2.4**: Enhanced patent search with sequence extraction, breakdown charts, simplified stats
        - **v2.3**: Unified batch CAI display, enhanced GC integration, slippery sites visualization
        - **v2.2**: Two-pass search strategy, intelligent result caching, seamless sequence transfer
        - **v2.1**: Consolidated search interface, smart quote filtering, enhanced CDS annotation
        - **v2.0**: UniProt integration, multi-database support, AI ranking, cross-reference mining
        
        **Use Cases:**
        - **Targeted CDS Search**: Find specific proteins like "spike protein" or "insulin"
        - **Comparative Genomics**: Find homologous sequences across species
        - **Codon Optimization Projects**: Source sequences for optimization workflows
        - **Protein Expression Research**: Find coding sequences for cloning
        - **Patent Research**: Discover existing technologies and extract sequences
        - **Sequence Analysis Pipelines**: Integrate with existing bioinformatics workflows
        - **Educational Research**: Access curated biological sequence data
        - **Interactive Data Exploration**: Explore sequence properties with interactive tools
        
        **Version:** Streamlit v2.5 (Interactive Visualizations & Enhanced Analysis)
        """)

    with tab6:
        st.header("mRNA Design")
        st.markdown("Design a full-length mRNA sequence by providing a coding sequence (CDS) and adding UTRs.")

        # Define UTR constants
        JT_5_UTR = "TCGAGCTCGGTACCTAATACGACTCACTATAAGGGAATAAACTAGTATTCTTCTGGTCCCCACAGACTCAGAGAGAACCCGCCACC"
        JT_3_UTR = "CTCGAGCTGGTACTGCATGCACGCAATGCTAGCTGCCCCTTTCCCGTCCTGGGTACCCCGAGTCTCCCCCGACCTCGGGTCCCAGGTATGCTCCCACCTCCACCTGCCCCACTCACCACCTCTGCTAGTTCCAGACACCTCCCAAGCACGCAGCAATGCAGCTCAAAACGCTTAGCCTAGCCACACCCCCACGGGAAACAGCAGTGATTAACCTTTAGCAATAAACGAAAGTTTAACTAAGCTATACTAACCCCAGGGTTGGTCAATTTCGTGCCAGCCACACCCTGGAGCTAGCAAACTTGTTTATTGCAGCTTATAATGGTTACAAATAAAGCAATAGCATCACAAATTTCACAAATAAAGCATTTTTTTCACTGCATTCTAGTTGTGGTTTGTCCAAACTCATCAATGTATCTTATCATGTCTGGATC"

        SIGNAL_PEPTIDES_DATA = {
            "tPA Signal Peptide": {
                "common_use": "Directs proteins to secretory pathway",
                "sequence_aa": "MDAMKRGLCCVLLLCGAVFVS"
            },
            "IL-2 Signal Peptide": {
                "common_use": "Enhances secretion of cytokines, antigens",
                "sequence_aa": "MYRMQLLSCIALSLALVTNS"
            },
            "Ig κ-chain Signal Peptide": {
                "common_use": "Common in antibody or fusion protein expression",
                "sequence_aa": "METDTLLLWVLLLWVPGSTG"
            },
            "Albumin Signal Peptide": {
                "common_use": "Used for liver-targeted or plasma-secreted proteins",
                "sequence_aa": "MKWVTFISLLLLFSSAYSRGV"
            },
            "Gaussia Luciferase SP": {
                "common_use": "Used in reporter constructs for secreted luciferase",
                "sequence_aa": "MKTIIALSYIFCLVFA"
            },
            "BM40/SPARC Signal Peptide": {
                "common_use": "Common in mRNA vaccines for targeting secretory pathway",
                "sequence_aa": "MGSFSLWLLLLLQSLVAIQG"
            },
            "CD33 Signal Peptide": {
                "common_use": "Used in immune cell-targeted expression",
                "sequence_aa": "MDMVLKVAAVLAGLVSLLVRA"
            },
            "HSA Signal Peptide": {
                "common_use": "Used for hepatocyte-specific mRNA delivery",
                "sequence_aa": "MKWVTFISLLLLFSSAYSRGVFRR"
            },
            "EPO Signal Peptide": {
                "common_use": "Common for erythropoietin or glycoprotein secretion",
                "sequence_aa": "MGVHECPAWLWLLLSLLSLPLGL"
            },
            "Tissue Plasminogen Activator (tPA)": {
                "common_use": "Strong secretory signal; Frequently used in mRNA vaccines (e.g., for spike protein)",
                "sequence_aa": "MDAMKRGLCCVLLLCGAVFVS"
            }
        }

        st.subheader("1. Provide Coding Sequence (CDS)")

        # Initialize session state for mrna_cds_input_method if not already set
        if 'mrna_cds_input_method' not in st.session_state:
            st.session_state.mrna_cds_input_method = "Paste Sequence"

        # Determine the initial index for the radio button
        initial_index = 0 if st.session_state.mrna_cds_input_method == "Paste Sequence" else 1
        cds_input_method = st.radio("Choose CDS input method:", ("Paste Sequence", "Search Database"), key="mrna_cds_input_method", index=initial_index)
        
        # Ensure cds_sequence always reflects the session state for the text area
        cds_sequence = st.session_state.mrna_design_cds_paste

        if cds_input_method == "Paste Sequence":
            st.text_area("Paste CDS here:", value=st.session_state.mrna_design_cds_paste, height=150, key="mrna_design_cds_paste")
        else:
            st.markdown("#### Search UniProt for Proteins")

            protein_query_mrna = st.text_input(
                "Enter protein search (e.g., 'SARS-CoV-2 spike protein', 'human insulin'):",
                placeholder="SARS-CoV-2 spike protein",
                key="mrna_protein_search_query"
            )

            col_search1_mrna, col_search2_mrna = st.columns([2, 1])
            with col_search1_mrna:
                search_protein_btn_mrna = st.button("🔍 Search UniProt", type="primary", key="mrna_search_protein_btn")
            with col_search2_mrna:
                max_uniprot_results_mrna = st.slider("Max results", 5, 20, 10, key="mrna_max_uniprot_cds")

            # Initialize session state for this tab's CDS workflow
            if 'mrna_uniprot_results' not in st.session_state:
                st.session_state.mrna_uniprot_results = []
            if 'mrna_selected_uniprot_entry' not in st.session_state:
                st.session_state.mrna_selected_uniprot_entry = None
            if 'mrna_cds_options' not in st.session_state:
                st.session_state.mrna_cds_options = []

            if search_protein_btn_mrna and protein_query_mrna.strip():
                with st.spinner("Searching UniProt..."):
                    results = st.session_state.uniprot_engine.search_protein_sequences(protein_query_mrna, max_uniprot_results_mrna)
                    st.session_state.mrna_uniprot_results = results
                    st.session_state.mrna_selected_uniprot_entry = None
                    st.session_state.mrna_cds_options = []
                    if not results:
                        st.warning("No UniProt entries found.")

            if st.session_state.mrna_uniprot_results:
                st.markdown("#### Select a Protein Entry")
                uniprot_options_mrna = [f"{entry['accession']} - {entry['protein_name']} [{entry['organism']}]" for entry in st.session_state.mrna_uniprot_results]
                selected_uniprot_idx_mrna = st.selectbox(
                    "Choose a UniProt entry:",
                    range(len(uniprot_options_mrna)),
                    format_func=lambda x: uniprot_options_mrna[x],
                    key="mrna_uniprot_selection"
                )
                st.session_state.mrna_selected_uniprot_entry = st.session_state.mrna_uniprot_results[selected_uniprot_idx_mrna]

                if st.session_state.mrna_selected_uniprot_entry['nucleotide_refs']:
                    st.markdown("#### Retrieve NCBI CDS Information")
                    if st.button("🧬 Get CDS from NCBI", type="primary", key="mrna_get_ncbi_btn"):
                        st.session_state.mrna_cds_options = []
                        for ref in st.session_state.mrna_selected_uniprot_entry['nucleotide_refs']:
                            if ref['database'] in ['EMBL', 'RefSeq']:
                                with st.spinner(f"Retrieving CDS info for {ref['id']}..."):
                                    ncbi_details = st.session_state.ncbi_engine.scrape_ncbi_page(ref['id'], protein_query_mrna)
                                    if ncbi_details['success'] and ncbi_details['cds_sequences']:
                                        st.session_state.mrna_cds_options.extend(ncbi_details['cds_sequences'])
                        if not st.session_state.mrna_cds_options:
                            st.error("Could not retrieve any CDS information.")

            if st.session_state.mrna_cds_options:
                st.markdown("#### Select CDS to Use")
                cds_dropdown_options_mrna = [f"{cds.get('gene_name', 'Unknown')} | {cds.get('product', 'Unknown')} | {cds.get('length', 0)} bp" for cds in st.session_state.mrna_cds_options]
                selected_cds_idx_mrna = st.selectbox(
                    "Choose a CDS feature:",
                    range(len(cds_dropdown_options_mrna)),
                    format_func=lambda x: cds_dropdown_options_mrna[x],
                    key="mrna_cds_selection"
                )
                selected_cds_mrna = st.session_state.mrna_cds_options[selected_cds_idx_mrna]
                
                if st.button("Apply this CDS to mRNA Design", key="mrna_apply_cds"):
                    st.session_state.mrna_design_cds_paste = selected_cds_mrna['sequence']
                    st.success(f"✅ CDS from {selected_cds_mrna.get('gene_name', 'Unknown')} applied!")
                    st.rerun()

        st.subheader("2. Add Signal Peptide (Optional)")
        add_signal_peptide = st.checkbox("Add Signal Peptide to mRNA", key="add_signal_peptide_checkbox")

        if add_signal_peptide:
            signal_peptide_names = list(SIGNAL_PEPTIDES_DATA.keys())
            selected_signal_peptide_name = st.selectbox(
                "Select a Signal Peptide:",
                signal_peptide_names,
                key="signal_peptide_selection"
            )
            
            selected_sp_info = SIGNAL_PEPTIDES_DATA[selected_signal_peptide_name]
            st.info(f"**Common Use:** {selected_sp_info['common_use']}\n\n**Amino Acid Sequence:** {selected_sp_info['sequence_aa']}")

        st.subheader("2. Sequence Processing Options")

        # GC Content Correction
        st.markdown("**GC Content Correction**")
        gc_correction_enabled = st.checkbox("Enable GC content correction (if > 70%)")
        st.info("This will attempt to lower the GC content of the CDS to between 55-70% using synonymous codons.")
        
        # Codon Optimization
        st.markdown("**Codon Optimization**")
        optimization_method_mrna = st.selectbox(
            "Choose an optimization method for the CDS:",
            ["None", "Standard Codon Optimization", "Balanced Optimization", "NC Stop Codon Optimization", "JT Plus1 Stop Optimization"],
            key="mrna_design_optimization"
        )

        # Stop Codon Selection
        st.markdown("**Stop Codon Selection**")
        STOP_CODONS = ["TAA", "TAG", "TGA"]
        selected_stop_codon = st.selectbox(
            "Select Stop Codon to append:",
            STOP_CODONS,
            key="mrna_design_stop_codon"
        )

        st.subheader("3. Design mRNA")
        if st.button("Design mRNA Sequence", type="primary"):
            if not cds_sequence.strip():
                st.error("Please provide a CDS sequence first.")
            else:
                is_valid, clean_cds, error_msg = validate_dna_sequence(cds_sequence)
                if not is_valid:
                    st.error(error_msg)
                else:
                    processed_cds = clean_cds
                    
                    # Step 1: GC Content Correction
                    if gc_correction_enabled:
                        with st.spinner("Adjusting GC content..."):
                            initial_gc = calculate_gc_content(processed_cds)
                            if initial_gc > 70.0:
                                processed_cds = adjust_gc_content(processed_cds, max_gc=75.0, min_gc=55.0)
                            else:
                                st.info(f"Initial GC content ({initial_gc:.1f}%) is not above 70%. No correction applied.")
                            
                            with st.spinner("Enforcing local GC content (10bp windows)..."):
                                processed_cds = enforce_local_gc_content(processed_cds, target_max_gc=75.0, window_size=25, step_size=1)

                    # Step 2: Codon Optimization
                    if optimization_method_mrna != "None":
                        with st.spinner(f"Applying {optimization_method_mrna}..."):
                            protein_seq = translate_dna(processed_cds)
                            if optimization_method_mrna == "Standard Codon Optimization":
                                processed_cds = codon_optimize(protein_seq)
                            elif optimization_method_mrna == "Balanced Optimization":
                                processed_cds = balanced_optimisation(processed_cds)
                            elif optimization_method_mrna == "NC Stop Codon Optimization":
                                processed_cds = nc_stop_codon_optimisation(processed_cds)
                            elif optimization_method_mrna == "JT Plus1 Stop Optimization":
                                processed_cds = JT_Plus1_Stop_Optimized(processed_cds)
                            st.success(f"Successfully applied {optimization_method_mrna}.")

                    # Step 3: Assemble the full CDS, handling the signal peptide correctly
                    dna_signal_peptide = ""
                    main_cds = processed_cds

                    if add_signal_peptide:
                        selected_sp_info = SIGNAL_PEPTIDES_DATA[selected_signal_peptide_name]
                        selected_sp_aa_seq = selected_sp_info["sequence_aa"]
                        dna_signal_peptide = reverse_translate_highest_cai(selected_sp_aa_seq)
                        
                        # IMPORTANT: Remove the ATG from the main CDS only if a signal peptide is added
                        if main_cds.upper().startswith("ATG"):
                            main_cds = main_cds[3:]
                            st.info("Removed ATG start codon from main CDS because a signal peptide was added.")
                    
                    # The full coding sequence is the signal peptide followed by the main CDS
                    full_cds = dna_signal_peptide + main_cds
                    
                    # Add stop codons to the end of the full coding sequence
                    STANDARD_STOP_CODONS = {"TAA", "TAG", "TGA"}
                    last_codon = full_cds[-3:].upper() if len(full_cds) >= 3 else ""
                    if last_codon in STANDARD_STOP_CODONS:
                        # If the sequence already ends in a stop, replace it with the selected double stop
                        cds_with_stops = full_cds[:-3] + (selected_stop_codon * 2)
                    else:
                        # Otherwise, append the double stop codon
                        cds_with_stops = full_cds + (selected_stop_codon * 2)

                    # Step 4: Assemble and display the final mRNA sequence
                    final_mrna_sequence = JT_5_UTR + cds_with_stops + JT_3_UTR
                    st.subheader("✅ Final mRNA Sequence")
                    
                    # For display, we pass the components separately to be colored correctly
                    # The main CDS for display is the full CDS with stops, minus the signal peptide part
                    main_cds_for_display = cds_with_stops[len(dna_signal_peptide):]
                    
                    display_colored_mrna_sequence(
                        utr5_seq=JT_5_UTR, 
                        cds_seq=main_cds_for_display,
                        utr3_seq=JT_3_UTR, 
                        signal_peptide_seq=dna_signal_peptide, 
                        key_suffix="final_mrna"
                    )

                    # Step 5: Perform and display the final analysis
                    st.subheader("📊 Final Analysis")
                    
                    # The context for frame analysis must include the 5' UTR to find the junctional ACCATG
                    analysis_context_sequence = JT_5_UTR + full_cds
                    
                    # Detailed stats table (using the full, correct CDS)
                    summary_df = generate_detailed_mrna_summary(full_cds, final_mrna_sequence, JT_5_UTR, JT_3_UTR)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                    # Top row: CAI/GC and +1 Stop Pie Chart
                    col_chart1, col_chart2 = st.columns([3, 1])

                    with col_chart1:
                        st.markdown("##### CDS CAI and GC Content")
                        cai_result, cai_error = run_single_optimization(full_cds, "In-Frame Analysis")
                        if not cai_error and cai_result:
                            cai_df = pd.DataFrame(cai_result)
                            fig_cai_gc = create_interactive_cai_gc_plot(
                                cai_df['Position'].tolist(),
                                cai_df['CAI_Weight'].tolist(),
                                cai_df['Amino_Acid'].tolist(),
                                full_cds,
                                "Processed CDS"
                            )
                            st.plotly_chart(fig_cai_gc, use_container_width=True)
                        else:
                            st.warning("Could not generate CAI/GC plot.")

                    with col_chart2:
                        st.markdown("##### CDS +1 Stop Codons")
                        plus1_stops = number_of_plus1_stops(full_cds)
                        if plus1_stops['total'] > 0:
                            stop_labels = ['TAA', 'TAG', 'TGA']
                            stop_values = [plus1_stops['TAA'], plus1_stops['TAG'], plus1_stops['TGA']]
                            fig_pie = create_interactive_pie_chart(stop_values, stop_labels, "+1 Stop Codon Distribution")
                            st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            st.info("No +1 stop codons found in the processed CDS.")

                    # Bottom row: full-width visualization
                    st.markdown("---")
                    st.markdown("##### Final mRNA Visualisation")
                    create_geneious_like_visualization(
                        utr5_seq=JT_5_UTR, 
                        cds_seq=main_cds_for_display,
                        utr3_seq=JT_3_UTR, 
                        signal_peptide_seq=dna_signal_peptide, 
                        key_suffix="final_mrna"
                    )

                            
    with tab7:
            st.header("Cancer Vaccine Design")
            st.markdown("Design a personalized cancer vaccine by combining multiple peptides with appropriate linkers")
            
            # Step 1: Select number of peptides
            st.subheader("1. Define Vaccine Components")
            num_peptides = st.number_input("Number of peptides in vaccine", min_value=1, max_value=10, value=3)
            
            # Initialize peptide inputs dictionary if not in session state
            if 'cancer_vaccine_peptides' not in st.session_state:
                st.session_state.cancer_vaccine_peptides = {}
            
            # Step 2: Input peptide sequences
            st.markdown("#### Peptide Sequences")
            peptide_sequences = []
            
            for i in range(1, int(num_peptides) + 1):
                key = f"peptide_{i}"
                if key not in st.session_state.cancer_vaccine_peptides:
                    st.session_state.cancer_vaccine_peptides[key] = ""
                
                st.session_state.cancer_vaccine_peptides[key] = st.text_area(
                    f"Peptide {i} (amino acid sequence):",
                    value=st.session_state.cancer_vaccine_peptides[key],
                    height=80,
                    key=f"cancer_peptide_{i}"
                )
                
                if st.session_state.cancer_vaccine_peptides[key]:
                    peptide_sequences.append(st.session_state.cancer_vaccine_peptides[key])
            
            # Step 3: Select linker
            st.markdown("#### Linker Selection")
            
            LINKER_OPTIONS = {
                "(G₄S)n linker": {
                    "sequence_aa": "GGGGS",
                    "type": "Flexible",
                    "purpose": "Most widely used linker; adds flexibility between domains to reduce steric clash"
                },
                "EAAAK linker": {
                    "sequence_aa": "EAAAK",
                    "type": "Rigid (helical)",
                    "purpose": "Promotes α-helix formation; keeps domains structurally separate"
                },
                "HE Linker": {
                    "sequence_aa": "HEHEHE",
                    "type": "Rigid",
                    "purpose": "Promotes hydrophilic spacing; sometimes used for helical separation"
                },
                "AP linker": {
                    "sequence_aa": "AEAAAKA",
                    "type": "Rigid",
                    "purpose": "Engineered rigid helix; used for mechanical separation of domains"
                },
                "(XP)n linker": {
                    "sequence_aa": "GPGPG",
                    "type": "Flexible/Spacer",
                    "purpose": "T cell epitope spacers (seen in multi-epitope vaccines)"
                },
                "AAY linker": {
                    "sequence_aa": "AAY",
                    "type": "Cleavable",
                    "purpose": "Used in epitope fusion vaccines; recognized by immunoproteasome"
                },
                "GPGPG linker": {
                    "sequence_aa": "GPGPG",
                    "type": "Flexible",
                    "purpose": "Used in multi-epitope vaccine constructs; promotes better MHC presentation"
                },
                "RRRRRR linker": {
                    "sequence_aa": "RRRRRR",
                    "type": "Cell-penetrating",
                    "purpose": "Enhances delivery, e.g., in peptide-based vaccines or intracellular targeting"
                },
                "KFERQ linker": {
                    "sequence_aa": "KFERQ",
                    "type": "Degron motif",
                    "purpose": "Targets proteins for lysosomal degradation (used in autophagy or clearance therapy)"
                },
                "ENLYFQG (TEV site)": {
                    "sequence_aa": "ENLYFQG",
                    "type": "Protease site",
                    "purpose": "Cleavable linker for conditional release (TEV protease)"
                },
                "LVPRGS (Thrombin site)": {
                    "sequence_aa": "LVPRGS",
                    "type": "Protease site",
                    "purpose": "Used to cleave fusion tags (e.g., tag–protein constructs)"
                }
            }
            
            # Format linker options for display
            linker_names = list(LINKER_OPTIONS.keys())
            
            selected_linker_name = st.selectbox(
                "Select Linker:",
                linker_names,
                key="cancer_vaccine_linker_selection"
            )
            
            selected_linker_info = LINKER_OPTIONS[selected_linker_name]
            st.info(f"**Type:** {selected_linker_info['type']}\n\n**Purpose:** {selected_linker_info['purpose']}\n\n**Amino Acid Sequence:** {selected_linker_info['sequence_aa']}")
            
            linker_repeats = 1
            if selected_linker_name in ["(G₄S)n linker", "EAAAK linker", "(XP)n linker"]:
                linker_repeats = st.slider("Number of linker repeats:", 1, 5, 1, key="cancer_linker_repeats")
            
            # Step 4: Signal Peptide Selection
            st.markdown("#### Signal Peptide Selection")
            
            signal_peptide_names = list(SIGNAL_PEPTIDES_DATA.keys())
            selected_signal_peptide_name = st.selectbox(
                "Select a Signal Peptide:",
                signal_peptide_names,
                key="cancer_signal_peptide_selection"
            )
            
            selected_sp_info = SIGNAL_PEPTIDES_DATA[selected_signal_peptide_name]
            st.info(f"**Common Use:** {selected_sp_info['common_use']}\n\n**Amino Acid Sequence:** {selected_sp_info['sequence_aa']}")
            
            # Step 5: MITD Option
            st.markdown("#### MITD Option")
            add_mitd = st.checkbox("Add Membrane-Interacting Transport Domain (MITD)", key="cancer_add_mitd")
            if add_mitd:
                st.info("MITD sequence (STQALNTVYTKLNIRLRQGRTLYTILNLA) will be added after the last peptide")
            
            # Step 6: Optimization Options
            st.subheader("2. Sequence Processing Options")
            
            # GC Content Correction
            st.markdown("**GC Content Correction**")
            gc_correction_enabled = st.checkbox("Enable GC content correction (if > 70%)", key="cancer_gc_correction")
            
            # Codon Optimization
            st.markdown("**Codon Optimization**")
            optimization_method_cancer = st.selectbox(
                "Choose an optimization method for the CDS:",
                ["None", "Standard Codon Optimization", "Balanced Optimization", "NC Stop Codon Optimization", "JT Plus1 Stop Optimization"],
                key="cancer_optimization"
            )
            
            # Stop Codon Selection
            st.markdown("**Stop Codon Selection**")
            STOP_CODONS = ["TAA", "TAG", "TGA"]
            selected_stop_codon = st.selectbox(
                "Select Stop Codon to append:",
                STOP_CODONS,
                key="cancer_stop_codon"
            )
            
            # Step 7: Design Vaccine Button
            st.subheader("3. Design Cancer Vaccine")
            design_vaccine_btn = st.button("Design Cancer Vaccine", type="primary", key="design_cancer_vaccine_btn")
            
            if design_vaccine_btn:
                if len(peptide_sequences) < 1:
                    st.error("Please provide at least one peptide sequence.")
                else:
                    with st.spinner("Designing cancer vaccine..."):
                        # Generate full amino acid sequence
                        # 1. Start with signal peptide
                        full_aa_sequence = selected_sp_info['sequence_aa']
                        
                        # 2. Add peptides with linkers
                        linker_aa_sequence = selected_linker_info['sequence_aa'] * linker_repeats
                        
                        for i, peptide in enumerate(peptide_sequences):
                            # Clean peptide (remove spaces, etc.)
                            clean_peptide = peptide.strip().upper()
                            
                            # Add peptide
                            full_aa_sequence += clean_peptide
                            
                            # Add linker after all peptides except the last one
                            if i < len(peptide_sequences) - 1:
                                full_aa_sequence += linker_aa_sequence
                        
                        # 3. Add MITD if selected
                        if add_mitd:
                            mitd_aa_sequence = "STQALNTVYTKLNIRLRQGRTLYTILNLA"
                            full_aa_sequence += mitd_aa_sequence
                        
                        # Reverse translate to nucleotide sequence
                        full_cds = reverse_translate_highest_cai(full_aa_sequence)
                        
                        # Process CDS (GC content correction and optimization)
                        processed_cds = full_cds
                        
                        # Step 1: GC Content Correction
                        if gc_correction_enabled:
                            initial_gc = calculate_gc_content(processed_cds)
                            if initial_gc > 70.0:
                                processed_cds = adjust_gc_content(processed_cds, max_gc=75.0, min_gc=55.0)
                            else:
                                st.info(f"Initial GC content ({initial_gc:.1f}%) is not above 70%. No correction applied.")
                            
                            # Enforce local GC content
                            processed_cds = enforce_local_gc_content(processed_cds, target_max_gc=75.0, window_size=25, step_size=1)
                        
                        # Step 2: Codon Optimization
                        if optimization_method_cancer != "None":
                            protein_seq = translate_dna(processed_cds)
                            if optimization_method_cancer == "Standard Codon Optimization":
                                processed_cds = codon_optimize(protein_seq)
                            elif optimization_method_cancer == "Balanced Optimization":
                                processed_cds = balanced_optimisation(processed_cds)
                            elif optimization_method_cancer == "NC Stop Codon Optimization":
                                processed_cds = nc_stop_codon_optimisation(processed_cds)
                            elif optimization_method_cancer == "JT Plus1 Stop Optimization":
                                processed_cds = JT_Plus1_Stop_Optimized(processed_cds)
                        
                        # Add stop codons
                        STANDARD_STOP_CODONS = {"TAA", "TAG", "TGA"}
                        last_codon = processed_cds[-3:].upper() if len(processed_cds) >= 3 else ""
                        if last_codon in STANDARD_STOP_CODONS:
                            # If the sequence already ends in a stop, replace it with the selected double stop
                            cds_with_stops = processed_cds[:-3] + (selected_stop_codon * 2)
                        else:
                            # Otherwise, append the double stop codon
                            cds_with_stops = processed_cds + (selected_stop_codon * 2)
                        
                        # Assemble final mRNA sequence
                        final_mrna_sequence = JT_5_UTR + cds_with_stops + JT_3_UTR
                        
                        # Display the results
                        st.subheader("✅ Cancer Vaccine mRNA Sequence")
                        
                        # For display only - calculate the signal peptide DNA sequence
                        signal_peptide_dna = reverse_translate_highest_cai(selected_sp_info['sequence_aa'])
                        
                        # Display colored sequence
                        display_colored_mrna_sequence(
                            utr5_seq=JT_5_UTR,
                            cds_seq=cds_with_stops[len(signal_peptide_dna):],  # Main CDS without signal peptide
                            utr3_seq=JT_3_UTR,
                            signal_peptide_seq=signal_peptide_dna,
                            key_suffix="cancer_vaccine"
                        )
                        
                        # Display components
                        st.subheader("📋 Vaccine Components")
                        
                        components_data = {
                            "Component": ["5' UTR", "Signal Peptide"],
                            "Type": ["Regulatory Element", "Targeting Sequence"],
                            "Length (aa)": ["N/A", len(selected_sp_info['sequence_aa'])],
                            "Length (bp)": [len(JT_5_UTR), len(signal_peptide_dna)]
                        }
                        
                        # Add peptides and linkers
                        for i, peptide in enumerate(peptide_sequences):
                            clean_peptide = peptide.strip().upper()
                            components_data["Component"].append(f"Peptide {i+1}")
                            components_data["Type"].append("Antigen")
                            components_data["Length (aa)"].append(len(clean_peptide))
                            components_data["Length (bp)"].append(len(clean_peptide) * 3)  # Approximate
                            
                            if i < len(peptide_sequences) - 1:
                                components_data["Component"].append(f"Linker {i+1}")
                                components_data["Type"].append(selected_linker_info['type'])
                                components_data["Length (aa)"].append(len(selected_linker_info['sequence_aa']) * linker_repeats)
                                components_data["Length (bp)"].append(len(selected_linker_info['sequence_aa']) * linker_repeats * 3)  # Approximate
                        
                        # Add MITD if selected
                        if add_mitd:
                            components_data["Component"].append("MITD")
                            components_data["Type"].append("Transport Domain")
                            components_data["Length (aa)"].append(30)  # Length of MITD
                            components_data["Length (bp)"].append(90)  # 30 aa * 3 bp/aa
                        
                        # Add 3' UTR
                        components_data["Component"].append("3' UTR")
                        components_data["Type"].append("Regulatory Element")
                        components_data["Length (aa)"].append("N/A")
                        components_data["Length (bp)"].append(len(JT_3_UTR))
                        
                        # Display components table
                        components_df = pd.DataFrame(components_data)
                        st.dataframe(components_df, use_container_width=True, hide_index=True)
                        
                        # Perform final analysis
                        st.subheader("📊 Final Analysis")
                        
                        # The context for frame analysis must include the 5' UTR to find the junctional ACCATG
                        analysis_context_sequence = JT_5_UTR + processed_cds
                        
                        # Detailed stats table (using the full, correct CDS)
                        summary_df = generate_detailed_mrna_summary(processed_cds, final_mrna_sequence, JT_5_UTR, JT_3_UTR)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                        
                        # CAI/GC Plot and +1 Stop Pie Chart
                        col_chart1, col_chart2 = st.columns([3, 1])
                        
                        with col_chart1:
                            st.markdown("##### CDS CAI and GC Content")
                            # The CAI plot should analyze the full coding sequence (SP + main CDS)
                            cai_result, cai_error = run_single_optimization(processed_cds, "In-Frame Analysis")
                            if not cai_error and cai_result:
                                cai_df = pd.DataFrame(cai_result)
                                fig_cai_gc = create_interactive_cai_gc_plot(
                                    cai_df['Position'].tolist(),
                                    cai_df['CAI_Weight'].tolist(),
                                    cai_df['Amino_Acid'].tolist(),
                                    processed_cds,
                                    "Processed CDS"
                                )
                                st.plotly_chart(fig_cai_gc, use_container_width=True)
                            else:
                                st.warning("Could not generate CAI/GC plot.")
                        
                            
                        with col_chart2:
                            st.markdown("##### CDS +1 Stop Codons")
                            plus1_stops = number_of_plus1_stops(full_cds)
                            if plus1_stops['total'] > 0:
                                stop_labels = ['TAA', 'TAG', 'TGA']
                                stop_values = [plus1_stops['TAA'], plus1_stops['TAG'], plus1_stops['TGA']]
                                fig_pie = create_interactive_pie_chart(stop_values, stop_labels, "+1 Stop Codon Distribution")
                                st.plotly_chart(fig_pie, use_container_width=True)
                            else:
                                st.info("No +1 stop codons found in the processed CDS.")

                        # Bottom row: full-width visualization
                        st.markdown("---")
                        st.markdown("##### Final mRNA Visualisation")
                        create_geneious_like_visualization(
                                utr5_seq=JT_5_UTR,
                                cds_seq=cds_with_stops[len(signal_peptide_dna):],
                                utr3_seq=JT_3_UTR,
                                signal_peptide_seq=signal_peptide_dna,
                                key_suffix=f"cancer_vaccine_{id(cds_with_stops)}"  # Using a unique ID
                            ) 
                    

if __name__ == "__main__":
    main()
