# v1.7 - - 20250626 No more charts they were gitting in the way 
#Need to sort out the format of option 2 in batch optimisation 
#make the logo rounder like an app 
#get it to recognise Us

import os
import sys
import platform
import subprocess
import pandas as pd
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog, ttk
from collections import defaultdict, Counter
from Bio.Seq import Seq
import logging
import json
import math
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.drawing.image import Image

# -------------------
# Configuration and Constants
# -------------------
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
combined_df = pd.DataFrame()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('codon_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------
# Configuration Management
# -------------------
def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                logger.info(f"Configuration loaded from {CONFIG_FILE}")
                return {**DEFAULT_CONFIG, **loaded_config}
        else:
            logger.info("No config file found, using defaults")
            return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.warning(f"Error loading config: {e}. Using defaults.")
        return DEFAULT_CONFIG.copy()

def save_config(config_to_save):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")

config = load_config()

# --- Globals for Step 4 ---
accumulated_results_list = []
accumulation_run_id_counter = 0
combine_outputs_var = None 
root = None

# -------------------
# macOS Specific Helper Functions
# -------------------
def set_macos_app_icon(root_window, logger_instance):
    """Sets the application icon on macOS if the icon file exists."""
    if platform.system() == "Darwin":
        icon_path = os.path.join(os.getcwd(), "nucleic_acid_logo.png")
        if os.path.exists(icon_path):
            try:
                # Ensure root_window is not None and is a valid Tkinter window
                if root_window and hasattr(root_window, 'iconphoto'):
                    img = tk.PhotoImage(file=icon_path)
                    root_window.iconphoto(True, img)
                    logger_instance.info(f"Attempted to set macOS application icon from {icon_path}")
                else:
                    logger_instance.warning("Root window not valid for setting macOS icon.")
            except tk.TclError as e:
                logger_instance.warning(f"Failed to set macOS application icon from {icon_path}: {e}. Might be an invalid image format or other Tcl/Tk error.")
            except Exception as e:
                logger_instance.warning(f"An unexpected error occurred while setting macOS application icon: {e}")
        else:
            logger_instance.warning(f"macOS icon file not found at {icon_path}. Application will use default icon.")
    # else:
        # logger_instance.info("Not on macOS, skipping application icon setting.") # Optional: for non-macOS platforms

# -------------------
# Utility Functions
# -------------------
def open_file(filepath):
    global config, logger # Added globals
    if not config.get("auto_open_files", True):
        return
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(filepath)
        elif system == "Darwin":
            subprocess.call(["open", filepath])
        else:
            subprocess.call(["xdg-open", filepath])
        logger.info(f"Opened file: {filepath}")
    except Exception as e:
        logger.error(f"Could not open file {filepath}: {e}")
        messagebox.showwarning("File Open Error", f"Could not open file {filepath}: {e}")

def safe_exit(code=0):
    global root, logger # Added globals
    try:
        if root and root.winfo_exists(): 
            root.destroy()
        logger.info(f"Application exiting with code {code}")
    except tk.TclError: 
        logger.info("Root window already destroyed or not available.")
    except Exception as e:
        logger.error(f"Error during safe_exit: {e}")
    finally:
        os._exit(code)

def validate_dna_sequence(sequence):
    global logger # Added global
    if not sequence:
        return False, "", "No DNA sequence provided"
    cleaned = sequence.upper().replace('\n', '').replace(' ', '').replace('\t', '')
    invalid_bases = set(cleaned) - set(VALID_DNA_BASES)
    if invalid_bases:
        return False, "", f"Invalid characters found: {', '.join(invalid_bases)}. Only A, T, G, C allowed."
    if len(cleaned) % 3 != 0:
        logger.warning(f"Sequence length ({len(cleaned)}) is not a multiple of 3")
    return True, cleaned, ""

def get_codon_file_path():
    global config, logger, root # Added globals
    codon_file = config.get("codon_file_path", "HumanCodons.xlsx")
    if not os.path.isabs(codon_file):
        potential_paths = [
            codon_file, os.path.join(os.getcwd(), codon_file),
            os.path.expanduser(f"~/Desktop/{codon_file}"), os.path.expanduser(f"~/Documents/{codon_file}"),
        ]
        for path in potential_paths:
            if os.path.exists(path): return path
    elif os.path.exists(codon_file): return codon_file
    
    parent_window = root if root and root.winfo_exists() else tk.Toplevel()
    if parent_window is not root and hasattr(parent_window, 'withdraw'): parent_window.withdraw()

    messagebox.showwarning("Codon File Not Found", f"Could not find codon usage file: {codon_file}\nPlease select the file manually.", parent=parent_window)
    file_path = filedialog.askopenfilename(title="Select Human Codon Usage File", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")], parent=parent_window)
    
    if parent_window is not root and hasattr(parent_window, 'destroy'): parent_window.destroy()

    if file_path:
        config["codon_file_path"] = file_path
        save_config(config)
        return file_path
    return None

# -------------------
# Load Codon Usage Data
# -------------------
# Define globals that will be populated by load_codon_data
genetic_code = {}
codon_weights = {}
preferred_codons = {}
human_codon_usage = {}
aa_to_codons = defaultdict(list)

def load_codon_data():
    global logger, genetic_code, codon_weights, preferred_codons, human_codon_usage, aa_to_codons, root, config
    codon_file = get_codon_file_path()
    if not codon_file:
        logger.error("No codon file provided")
        parent_window = root if root and root.winfo_exists() else tk.Toplevel()
        if parent_window is not root and hasattr(parent_window, 'withdraw'): parent_window.withdraw()
        messagebox.showerror("Error", "Codon usage file is required.", parent=parent_window)
        if parent_window is not root and hasattr(parent_window, 'destroy'): parent_window.destroy()
        safe_exit(1)
    try:
        logger.info(f"Loading codon data from: {codon_file}")
        df = pd.read_excel(codon_file)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        required_columns = ['triplet', 'amino_acid', 'fraction']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns: raise ValueError(f"Missing required columns: {missing_columns}")
        
        df['triplet'] = df['triplet'].str.upper().str.strip()
        df['amino_acid'] = df['amino_acid'].str.upper().str.strip().replace({'*': 'X'})
        df = df.dropna(subset=['triplet', 'amino_acid', 'fraction'])
        logger.info(f"Successfully loaded {len(df)} codon entries")
        
        genetic_code.clear(); genetic_code.update(df.set_index('triplet')['amino_acid'].to_dict())
        max_fraction = df.groupby('amino_acid')['fraction'].transform('max')
        df['weight'] = df['fraction'] / max_fraction
        codon_weights.clear(); codon_weights.update(df.set_index('triplet')['weight'].to_dict())
        preferred_codons.clear(); preferred_codons.update(df.sort_values('fraction', ascending=False).drop_duplicates('amino_acid').set_index('amino_acid')['triplet'].to_dict())
        human_codon_usage.clear(); human_codon_usage.update(df.set_index('triplet')['fraction'].to_dict())
        
        aa_to_codons.clear()
        for codon_val, freq in human_codon_usage.items():
            aa = genetic_code.get(codon_val, None)
            if aa and aa != 'X': aa_to_codons[aa].append((codon_val, freq))
        return df 
    except FileNotFoundError:
        logger.error(f"Codon file not found: {codon_file}")
        parent_window = root if root and root.winfo_exists() else tk.Toplevel()
        if parent_window is not root and hasattr(parent_window, 'withdraw'): parent_window.withdraw()
        messagebox.showerror("File Error", f"Codon file not found: {codon_file}", parent=parent_window)
        if parent_window is not root and hasattr(parent_window, 'destroy'): parent_window.destroy()
        safe_exit(1)
    except Exception as e:
        logger.error(f"Error loading codon file: {e}", exc_info=True)
        parent_window = root if root and root.winfo_exists() else tk.Toplevel()
        if parent_window is not root and hasattr(parent_window, 'withdraw'): parent_window.withdraw()
        messagebox.showerror("Data Error", f"Error loading codon file: {e}", parent=parent_window)
        if parent_window is not root and hasattr(parent_window, 'destroy'): parent_window.destroy()
        safe_exit(1)

codon_df = load_codon_data()
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
for codon_val, aa_val in STANDARD_GENETIC_CODE.items(): synonymous_codons[aa_val].append(codon_val)
FIRST_AA_CANDIDATES = ['L', 'I', 'V']
SECOND_AA_CANDIDATES = ['V', 'I']

def third_aa_has_A_G_synonymous(aa):
    for codon_val in synonymous_codons.get(aa, []): # Added .get for safety
        if codon_val.startswith(('A', 'G')): return True
    return False

# -------------------
# Core Optimization Functions
# -------------------
def translate_dna(seq):
    global logger, genetic_code
    try:
        protein = ""
        for i in range(0, len(seq) - 2, 3):
            codon_val = seq[i:i+3].upper()
            aa = genetic_code.get(codon_val, '?')
            protein += aa
        return protein
    except Exception as e:
        logger.error(f"Error translating DNA sequence: {e}", exc_info=True)
        raise

def codon_optimize(protein_seq):
    global logger, preferred_codons
    try:
        optimized = ''.join(preferred_codons.get(aa, 'NNN') for aa in protein_seq if aa != 'X')
        logger.info(f"Standard codon optimization completed for {len(protein_seq)} amino acids")
        return optimized
    except Exception as e:
        logger.error(f"Error in codon optimization: {e}", exc_info=True)
        raise

def get_codon_weights_row(dna_seq):
    global logger, codon_weights
    try:
        codons_list = [dna_seq[i:i+3].upper() for i in range(0, len(dna_seq) - 2, 3)]
        weights = [codon_weights.get(c, 1e-6) for c in codons_list]
        return weights, codons_list
    except Exception as e:
        logger.error(f"Error calculating codon weights: {e}", exc_info=True)
        raise

def find_coding_sequence_bounds(dna_seq):
    global logger
    try:
        dna_seq_upper = dna_seq.upper().replace('U', 'T')
        stop_codons = {"TAA", "TAG", "TGA"}
        
        # Find start position
        start_pos = None
        if dna_seq_upper.startswith('ATG'):
            start_pos = 3  # Start after ATG
        else:
            # Look for ACCATG
            accatg_pos = dna_seq_upper.find('ACCATG')
            if accatg_pos != -1:
                start_pos = accatg_pos + 6  # Start after ACCATG
        
        if start_pos is None:
            return None, None
        
        # Find end position - first in-frame stop codon
        end_pos = None
        for i in range(start_pos, len(dna_seq_upper) - 2, 3):
            codon = dna_seq_upper[i:i+3]
            if len(codon) == 3 and codon in stop_codons:
                end_pos = i  # Position of the stop codon (not after it)
                break
        
        return start_pos, end_pos
    except Exception as e:
        logger.error(f"Error finding coding sequence bounds: {e}", exc_info=True)
        return None, None

def number_of_Slippery_motifs(dna_seq):
    global logger
    try:
        dna_seq_upper = dna_seq.upper().replace('U', 'T')
        start_pos, end_pos = find_coding_sequence_bounds(dna_seq_upper)
        if start_pos is None:
            logger.warning("No valid start position found for slippery motif analysis")
            return 0
        
        # Use end_pos if found, otherwise use full sequence
        search_end = end_pos if end_pos is not None else len(dna_seq_upper) - 3
         
        # Count slippery motifs within coding sequence only, in-frame
        slippery_count = sum(1 for i in range(start_pos, search_end, 3) if dna_seq_upper[i:i+4] in Slippery_Motifs and i+4 <= len(dna_seq_upper))
        logger.info(f"Slippery motif analysis completed: {slippery_count} motifs found in coding sequence")
        return slippery_count
    except Exception as e:
        logger.error(f"Error in slippery motif analysis: {e}", exc_info=True)
        raise

def number_of_plus1_stops(dna_seq):
    global logger
    try:
        dna_seq_upper = dna_seq.upper().replace('U', 'T')
        start_pos, end_pos = find_coding_sequence_bounds(dna_seq_upper)
        if start_pos is None:
            logger.warning("No valid start position found for plus1 stop analysis")
            return {'TAA': 0, 'TAG': 0, 'TGA': 0, 'total': 0}
         
        stop_codons_set = {"TAA", "TAG", "TGA"}
        # Calculate +1 frame starting from start_pos + 1
        plus1_start = start_pos + 1
        
        # Use end_pos if found, otherwise use full sequence  
        search_end = end_pos if end_pos is not None else len(dna_seq_upper) - 2
        
        plus1_codons_list = [dna_seq_upper[i:i+3] for i in range(plus1_start, search_end, 3) if i+3 <= len(dna_seq_upper)]
        counts = Counter(c for c in plus1_codons_list if c in stop_codons_set)
        counts['total'] = sum(counts.values())
        for stop_val in stop_codons_set: 
            counts.setdefault(stop_val, 0)
        logger.info(f"Plus1 stop analysis completed: {counts['total']} total stops found in coding sequence")
        return dict(counts)
    except Exception as e:
        logger.error(f"Error in plus1 stop analysis: {e}", exc_info=True)
        raise

def balanced_optimisation(dna_seq, bias_weight_input=None):
    global logger, config, aa_to_codons, PLUS1_STOP_CODONS, genetic_code # Added genetic_code
    bias_weight = bias_weight_input if bias_weight_input is not None else config.get("bias_weight", BIAS_WEIGHT_DEFAULT)
    try:
        dna_seq_upper = dna_seq.upper()
        # Protein translation using local genetic_code if available, or Bio.Seq default
        protein_str = ""
        for i in range(0, len(dna_seq_upper) -2, 3):
            codon = dna_seq_upper[i:i+3]
            protein_str += genetic_code.get(codon, Seq(codon).translate()) # Fallback if local genetic_code is incomplete
        
        optimised_seq = ""
        idx = 0 # Renamed i to idx
        while idx < len(dna_seq_upper) - 2:
            current_codon = dna_seq_upper[idx:idx+3]
            aa = genetic_code.get(current_codon, str(Seq(current_codon).translate()))

            if idx < len(dna_seq_upper) - 5: # Check for two-codon substitutions
                next_codon_val = dna_seq_upper[idx+3:idx+6]
                aa2 = genetic_code.get(next_codon_val, str(Seq(next_codon_val).translate()))
                candidates = []
                # Ensure aa and aa2 are valid keys in aa_to_codons
                if aa in aa_to_codons and aa2 in aa_to_codons:
                    for c1, f1 in aa_to_codons[aa]:
                        for c2, f2 in aa_to_codons[aa2]:
                            combined = c1 + c2
                            codon1_plus1 = combined[1:4]
                            codon2_plus1 = combined[4:7] # This is actually codon from +2 frame relative to original `idx`
                            bonus = 0
                            if codon1_plus1 in PLUS1_STOP_CODONS and combined[2:5] in PLUS1_STOP_CODONS : bonus +=2 # check overlapping too
                            elif codon1_plus1 in PLUS1_STOP_CODONS : bonus +=1
                            
                            score = (f1 * f2) + bias_weight * bonus # Consider normalizing f1,f2 or using log scores
                            candidates.append((score, c1, c2))
                if candidates:
                    _, best1, best2 = max(candidates)
                    optimised_seq += best1 + best2
                    idx += 6
                    continue
            
            # Single codon substitution
            best_codon_val = current_codon
            # Initialize best_score with the score of the current_codon
            current_codon_freq = 0
            for syn_c, freq_val in aa_to_codons.get(aa, []):
                 if syn_c == current_codon:
                      current_codon_freq = freq_val
                      break
            
            temp_seq_orig = optimised_seq + current_codon + dna_seq_upper[idx+3:]
            plus1_window_orig_start = len(optimised_seq) + 1 # Start of +1 frame codon corresponding to current_codon in temp_seq_orig
            bonus_orig = 0
            if plus1_window_orig_start < len(temp_seq_orig) -2 :
                codon_plus1_orig = temp_seq_orig[plus1_window_orig_start : plus1_window_orig_start+3]
                if codon_plus1_orig in PLUS1_STOP_CODONS: bonus_orig = bias_weight
            best_score = current_codon_freq + bonus_orig
            
            for syn_codon, freq in aa_to_codons.get(aa, []):
                temp_seq = optimised_seq + syn_codon + dna_seq_upper[idx+3:]
                # The +1 codon starts at the second base of syn_codon
                plus1_codon_start_in_temp = len(optimised_seq) + 1 
                
                bonus_val = 0
                if plus1_codon_start_in_temp < len(temp_seq) -2: # Ensure there are enough chars for a codon
                    codon_plus1 = temp_seq[plus1_codon_start_in_temp : plus1_codon_start_in_temp+3]
                    if codon_plus1 in PLUS1_STOP_CODONS:
                        bonus_val = bias_weight
                
                score = freq + bonus_val
                if score > best_score:
                    best_score = score
                    best_codon_val = syn_codon
            
            optimised_seq += best_codon_val
            idx += 3

        if idx < len(dna_seq_upper): optimised_seq += dna_seq_upper[idx:]
        
        final_protein_str = ""
        for i in range(0, len(optimised_seq) -2, 3):
            codon = optimised_seq[i:i+3]
            final_protein_str += genetic_code.get(codon, str(Seq(codon).translate()))

        if final_protein_str != protein_str:
            logger.error(f"Protein sequence changed in balanced_optimisation! Original: {protein_str}, New: {final_protein_str}")
            return dna_seq_upper # Fallback to original if protein changes
        logger.info("Balanced optimization completed successfully")
        return optimised_seq
    except Exception as e:
        logger.error(f"Error in balanced optimization: {e}", exc_info=True)
        raise

def nc_stop_codon_optimisation(dna_seq):
    global logger, genetic_code
    try:
        dna_seq_upper = dna_seq.upper()
        protein_str = ""
        for i in range(0, len(dna_seq_upper) -2, 3):
            codon = dna_seq_upper[i:i+3]
            protein_str += genetic_code.get(codon, str(Seq(codon).translate()))

        synonymous_codons_local = defaultdict(list)
        for c, aa_val in genetic_code.items(): synonymous_codons_local[aa_val].append(c)
        
        optimised_seq = ""
        idx = 0 # Renamed i to idx
        while idx < len(dna_seq_upper) - 2:
            codon_val = dna_seq_upper[idx:idx+3]
            aa = genetic_code.get(codon_val, str(Seq(codon_val).translate()))

            if idx < len(dna_seq_upper) - 5: # Try double substitution
                codon2 = dna_seq_upper[idx+3:idx+6]
                aa2 = genetic_code.get(codon2, str(Seq(codon2).translate()))
                if aa in synonymous_codons_local and aa2 in synonymous_codons_local:
                    double_subs = [(c1, c2) for c1 in synonymous_codons_local[aa] for c2 in synonymous_codons_local[aa2] if (c1[1:3] + c2[0:3] + c2[0:1]) in PLUS1_STOP_MOTIFS or (c1[2:3]+c2[0:3]+c2[0:2]) in PLUS1_STOP_MOTIFS] # Simplified motif check logic from original
                    if double_subs: # If any such pair creates a target +1 motif (e.g. TAATAA)
                        # This logic needs to be more specific about which motifs (TAATAA, TAGTAG) are formed by c1+c2 in +1 frame
                        # Original: if (c1 + c2)[1:7] in {"TAATAA", "TAGTAG"}
                        # This implies the 6 bases starting from the 2nd base of c1 form TAATAA or TAGTAG.
                        # c1 = N N N, c2 = N N N.  (c1+c2) = NNNNNN.  (c1+c2)[1:7] is NNNNNN (last 5 of c1, all of c2)
                        # This should be:  c1[1]c1[2]c2[0]c2[1]c2[2]c3[0] (if three codons involved)
                        # For two codons c1, c2: the +1 frame codons are c1[1]c1[2]c2[0] and c1[2]c2[0]c2[1] and c2[0]c2[1]c2[2]
                        # Let's use the original check for simplicity of this step:
                        double_subs_orig_check = [(c1,c2) for c1 in synonymous_codons_local[aa] for c2 in synonymous_codons_local[aa2] if (c1+c2)[1:7] in {"TAATAA", "TAGTAG"}]
                        if double_subs_orig_check:
                            best_c1, best_c2 = double_subs_orig_check[0] 
                            optimised_seq += best_c1 + best_c2
                            idx += 6
                            continue
            
            best_codon_val = codon_val
            # For single codon, check if any synonym creates TAA or TAG in +1 frame
            # The +1 codon is formed by last 2 bases of current codon and 1st of next
            if idx + 3 < len(dna_seq_upper): # if there is a next codon
                next_actual_codon = dna_seq_upper[idx+3:idx+6]
                for syn_c in synonymous_codons_local.get(aa,[]):
                    plus1_codon = syn_c[1:3] + next_actual_codon[0:1]
                    if plus1_codon in {"TAA", "TAG"}:
                        best_codon_val = syn_c
                        break 
            optimised_seq += best_codon_val
            idx += 3

        final_protein_str = ""
        for i in range(0, len(optimised_seq) -2, 3):
            codon = optimised_seq[i:i+3]
            final_protein_str += genetic_code.get(codon, str(Seq(codon).translate()))

        if final_protein_str != protein_str:
            logger.error(f"Protein sequence changed in nc_stop_codon_optimisation! Original: {protein_str}, New: {final_protein_str}")
            return dna_seq_upper # Fallback
        logger.info("NC stop codon optimization completed successfully")
        return optimised_seq
    except Exception as e:
        logger.error(f"Error in NC stop codon optimization: {e}", exc_info=True)
        raise

def JT_Plus1_Stop_Optimized(seq_input):
    global logger, STANDARD_GENETIC_CODE, synonymous_codons, FIRST_AA_CANDIDATES, SECOND_AA_CANDIDATES, PLUS1_STOP_MOTIFS
    try:
        seq = seq_input.upper()
        out_seq = ''
        idx = 0 # Renamed i to idx
        while idx <= len(seq) - 9:
            c1, c2, c3 = seq[idx:idx+3], seq[idx+3:idx+6], seq[idx+6:idx+9]
            aa1 = STANDARD_GENETIC_CODE.get(c1, '?')
            aa2 = STANDARD_GENETIC_CODE.get(c2, '?')
            aa3 = STANDARD_GENETIC_CODE.get(c3, '?')

            if (aa1 in FIRST_AA_CANDIDATES and aa2 in SECOND_AA_CANDIDATES and
                aa3 in synonymous_codons and third_aa_has_A_G_synonymous(aa3)):
                found_motif = False
                for syn1 in synonymous_codons.get(aa1,[]):
                    if not syn1.endswith('TA'): continue
                    for syn2 in synonymous_codons.get(aa2,[]):
                        if not syn2.startswith(('A', 'G')): continue
                        for syn3 in synonymous_codons.get(aa3,[]):
                            if not syn3.startswith(('A', 'G')): continue
                            motif_check = syn1[1:] + syn2 + syn3[:1] 
                            if motif_check in PLUS1_STOP_MOTIFS:
                                out_seq += syn1 + syn2 + syn3
                                idx += 9
                                found_motif = True; break
                        if found_motif: break
                    if found_motif: break
                if not found_motif: out_seq += c1; idx += 3
            else: out_seq += c1; idx += 3
        out_seq += seq[idx:]
        logger.info("JT Plus1 stop optimization completed successfully")
        return out_seq
    except Exception as e:
        logger.error(f"Error in JT Plus1 stop optimization: {e}", exc_info=True)
        raise

# -------------------
# Output Functions (Modified for Step 4)
# -------------------
def save_output_to_excel(data_dict, filename, auto_open=True, chart_image_path=None):
    global logger, config, root
    try:
        output_dir = config.get("default_output_dir", ".")
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
        df = pd.DataFrame(data_dict)
        df.to_excel(full_path, index=False)
        logger.info(f"Results saved to {full_path}")

        if chart_image_path and os.path.exists(chart_image_path):
            try:
                wb = load_workbook(full_path)
                ws = wb.active
                img = Image(chart_image_path)
                # Attempt to anchor below data, adjust if needed
                # Max_row might not be enough if data is very wide, consider a fixed cell like 'A20' or more dynamic
                img.anchor = ws.cell(row=ws.max_row + 2, column=1).coordinate 
                ws.add_image(img)
                wb.save(full_path)
                logger.info(f'Chart {chart_image_path} embedded into {full_path}')
            except Exception as chart_e:
                logger.error(f'Failed to embed chart {chart_image_path}: {chart_e}')

        if auto_open and config.get("auto_open_files", True): open_file(full_path)
        return full_path
    except Exception as e:
        logger.error(f"Error saving to Excel: {e}", exc_info=True)
        parent_window = root if root and root.winfo_exists() else None
        messagebox.showerror("Save Error", f"Could not save file: {e}", parent=parent_window)
        return None

# -------------------
# GUI Functions
# -------------------
def get_user_input(): # Full version
    global logger, root
    parent_dialog = root if root and root.winfo_exists() else tk.Toplevel()
    if parent_dialog is not root : parent_dialog.withdraw()

    while True:
        seq_input = simpledialog.askstring("Input DNA Sequence", "Please enter your DNA sequence...", parent=parent_dialog)
        if parent_dialog is not root and not parent_dialog.winfo_exists(): return None, None, None # Window closed
        if seq_input is None: # User cancelled
            if parent_dialog is not root: parent_dialog.destroy()
            return None, None, None
        is_valid, clean_seq, error_msg = validate_dna_sequence(seq_input)
        if is_valid: break
        else: messagebox.showerror("Input Error", error_msg, parent=parent_dialog)
    
    if parent_dialog is not root and parent_dialog.winfo_exists() : parent_dialog.destroy() # Clean up if temporary

    parent_dialog_choice = root if root and root.winfo_exists() else tk.Toplevel()
    if parent_dialog_choice is not root : parent_dialog_choice.withdraw()
    
    prompt_message = """Choose optimization method:
1: Standard Codon Optimization
2: CAI Weight Analysis (Original)
3: Balanced Optimization
4: NC Stop Codon Optimization
5: JT Plus1 Stop Optimization
6: Sequence Analysis"""
    choice = simpledialog.askinteger("Choose Optimization Method", prompt_message, minvalue=1, maxvalue=6, parent=parent_dialog_choice)
    
    if parent_dialog_choice is not root and not parent_dialog_choice.winfo_exists(): return None, None, None # Window closed
    if parent_dialog_choice is not root: parent_dialog_choice.destroy()

    if choice is None: return None, None, None
    try:
        protein_seq = translate_dna(clean_seq)
        logger.info(f"Input sequence: {len(clean_seq)} bp, Protein: {len(protein_seq)} aa")
        return clean_seq, protein_seq, choice
    except Exception as e:
        logger.error(f"Translation error in get_user_input: {e}", exc_info=True)
        parent_window = root if root and root.winfo_exists() else None
        messagebox.showerror("Translation Error", f"Error translating DNA: {e}", parent=parent_window)
        return None, None, None

# --- run_optimization (Modified for Step 4) ---
def run_optimization(clean_seq, protein_seq, choice): # Full version
    global combine_outputs_var, accumulated_results_list, accumulation_run_id_counter, logger, genetic_code, root, FRAME_OFFSET

    filename_template_for_individual_save = ""
    data = {}
    analysis_type_description = ""
    parent_window = root if root and root.winfo_exists() else None
    chart_image_to_embed_in_excel = None  # Initialize this variable
    
    try:
        if choice == 1:
            optimized = codon_optimize(protein_seq)
            weights, _ = get_codon_weights_row(optimized)
            data = {'Original_DNA': [clean_seq], 'Protein': [protein_seq], 'Optimized_DNA': [optimized], 'CAI_Weights': [','.join(f"{w:.4f}" for w in weights)]}
            filename_template_for_individual_save = "Codon_Optimized_Output.xlsx"
            analysis_type_description = "Standard Codon Optimization"
        elif choice == 2:
            weights, codons_list = get_codon_weights_row(clean_seq)
            data = {'Position': list(range(1, len(codons_list) + 1)), 'DNA_Codon': codons_list, 'CAI_Weight': weights, 'Amino_Acid': [genetic_code.get(c, '?') for c in codons_list]}
            filename_template_for_individual_save = "Original_Sequence_CAI_Analysis.xlsx"
            analysis_type_description = "CAI Weight Analysis"
        elif choice == 3:
            balanced_seq = balanced_optimisation(clean_seq)
            weights, _ = get_codon_weights_row(balanced_seq)
            data = {'Original_DNA': [clean_seq], 'Protein': [protein_seq], 'Balanced_Optimized_DNA': [balanced_seq], 'CAI_Weights': [','.join(f"{w:.4f}" for w in weights)]}
            filename_template_for_individual_save = "Balanced_Optimized_Output.xlsx"
            analysis_type_description = "Balanced Optimization"
        elif choice == 4:
            nc_stop_seq = nc_stop_codon_optimisation(clean_seq)
            weights, _ = get_codon_weights_row(nc_stop_seq)
            data = {'Original_DNA': [clean_seq], 'Protein': [protein_seq], 'NC_Stop_Optimized_DNA': [nc_stop_seq], 'CAI_Weights': [','.join(f"{w:.4f}" for w in weights)]}
            filename_template_for_individual_save = "NC_Stop_Optimized_Output.xlsx"
            analysis_type_description = "NC Stop Codon Optimization"
        elif choice == 5:
            plus1_optimized = JT_Plus1_Stop_Optimized(clean_seq)
            weights, _ = get_codon_weights_row(plus1_optimized)
            data = {'Original_DNA': [clean_seq], 'Protein': [protein_seq], 'JT_Plus1_Optimized_DNA': [plus1_optimized], 'CAI_Weights': [','.join(f"{w:.4f}" for w in weights)]}
            filename_template_for_individual_save = "JT_Plus1_Optimized_Output.xlsx"
            analysis_type_description = "JT Plus1 Stop Optimization"
        elif choice == 6:
            plus1_stop_counts = number_of_plus1_stops(clean_seq)
            start_pos, end_pos = find_coding_sequence_bounds(clean_seq)
            
            if start_pos is not None and end_pos is not None:
                coding_length = end_pos - start_pos
                plus1_len = coding_length // 3
                coding_info = f"Coding seq: {start_pos}-{end_pos} ({coding_length} bp)"
            elif start_pos is not None:
                # No stop codon found, use full sequence from start
                coding_length = len(clean_seq) - start_pos
                plus1_len = coding_length // 3
                coding_info = f"Coding seq: {start_pos}-end ({coding_length} bp, no stop found)"
            else:
                plus1_len = 0
                coding_info = "No valid coding sequence found"
            
            slippery_count = number_of_Slippery_motifs(clean_seq)
            data = {
                'Stop_Codon_Type': ['TAA', 'TAG', 'TGA', 'Total', 'Slippery_Motifs'],
                'Count_in_Plus1_Frame': [plus1_stop_counts['TAA'], plus1_stop_counts['TAG'], plus1_stop_counts['TGA'], plus1_stop_counts['total'], slippery_count],
                'Sequence_Info': [f"Seq len: {len(clean_seq)} bp", f"Prot len: {len(protein_seq)} aa", coding_info, f"Stop density: {plus1_stop_counts['total']/max(1,plus1_len):.3f}", f"Slippery density: {slippery_count}"]
            }
            filename_template_for_individual_save = "Sequence_Analysis.xlsx"
            analysis_type_description = "Plus1 Frame Stop Count Analysis"
        else:
            logger.error(f"Invalid choice: {choice}")
            messagebox.showerror("Invalid Choice", f"Choice {choice} is not recognized.", parent=parent_window)
            return
            
        # Check if we're in accumulation mode
        if combine_outputs_var is not None and combine_outputs_var.get():
            # Accumulation mode
            accumulation_run_id_counter += 1
            current_run_rows = []
            num_rows = 0
            
            if data:
                list_lengths = [len(v) for v in data.values() if isinstance(v, list)]
                num_rows = max(list_lengths) if list_lengths else 1
                
            for i in range(num_rows):
                row_dict = {'Accumulation_Run_ID': accumulation_run_id_counter, 'Analysis_Type_In_Run': analysis_type_description}
                for key, val_list_or_scalar in data.items():
                    if isinstance(val_list_or_scalar, list):
                        row_dict[key] = val_list_or_scalar[i] if i < len(val_list_or_scalar) else None
                    elif num_rows == 1:
                        row_dict[key] = val_list_or_scalar
                    else:
                        row_dict[key] = None
                current_run_rows.append(row_dict)
                
            accumulated_results_list.append(current_run_rows)
            logger.info(f"Results from run {accumulation_run_id_counter} ({analysis_type_description}) added.")
            messagebox.showinfo("Accumulated", f"Results added. Buffer: {len(accumulated_results_list)} run(s).", parent=parent_window)
            
        else:
            # Not in accumulation mode (combine_outputs_var is false)
            if not filename_template_for_individual_save:
                logger.error("Filename template not set.")
                messagebox.showerror("Internal Error", "Filename template not set.", parent=parent_window)
                return
            
            # For other choices that might generate charts (currently none do, but for future proofing):
            # if choice == X and needs_chart_for_excel:
            #   chart_image_to_embed_in_excel = ... save fig ...
            
            output_filename = save_output_to_excel(data, filename_template_for_individual_save, chart_image_path=chart_image_to_embed_in_excel)
            
            if output_filename:
                messagebox.showinfo("Complete", f"Operation completed.\nSaved to: {output_filename}", parent=parent_window)
            
            if chart_image_to_embed_in_excel and os.path.exists(chart_image_to_embed_in_excel):
                # Cleanup if any chart was saved for excel
                try:
                    os.remove(chart_image_to_embed_in_excel)
                    logger.info(f"Temporary chart file {chart_image_to_embed_in_excel} removed.")
                except Exception as e_remove:
                    logger.error(f"Failed to remove temporary chart file {chart_image_to_embed_in_excel}: {e_remove}")
                    
    except Exception as e:
        logger.error(f"Error during optimization choice {choice}: {e}", exc_info=True)
        messagebox.showerror("Optimization Error", f"An error occurred: {e}", parent=parent_window)

# --- Batch Processing (Modified for Step 1) ---
def process_batch_sequences(): # Full version
    global logger, json, root
    parent_window = root if root and root.winfo_exists() else None
    try:
        dialog_parent = tk.Toplevel(parent_window) if parent_window else tk.Toplevel()
        dialog_parent.withdraw()
        file_path = filedialog.askopenfilename(title="Select sequence file", filetypes=[("Text files", "*.txt"),("FASTA files", "*.fasta *.fa")], parent=dialog_parent)
        dialog_parent.destroy()
        if not file_path: return

        with open(file_path, 'r') as f: content = f.read()
        sequences = []
        if content.startswith('>'):
            lines = content.strip().splitlines(); current_seq, current_name = "", ""
            for line in lines:
                if line.startswith('>'):
                    if current_seq: sequences.append((current_name, current_seq))
                    current_name, current_seq = line[1:].strip(), ""
                else: current_seq += line.strip().upper()
            if current_seq: sequences.append((current_name, current_seq))
        else:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            for i, line in enumerate(lines): sequences.append((f"Sequence_{i+1}", line.upper()))
        
        if not sequences: messagebox.showwarning("No Sequences", "No valid sequences found.", parent=parent_window); return
        
        dialog_parent_choice = tk.Toplevel(parent_window) if parent_window else tk.Toplevel()
        dialog_parent_choice.withdraw()
        prompt_message = f"""Found {len(sequences)} sequences.
Choose optimization method for batch processing:
1: Standard Codon Optimization
2: CAI Weight Analysis (Original)
3: Balanced Optimization
4: NC Stop Codon Optimization
5: JT Plus1 Stop Optimization
6: Sequence Analysis"""
        choice = simpledialog.askinteger("Batch Processing", prompt_message, minvalue=1, maxvalue=6, parent=dialog_parent_choice)
        dialog_parent_choice.destroy()
        if choice is None: return
        
        results = []
        num_sequences = len(sequences)
        combined_df = pd.DataFrame()

        for name, seq_val in sequences:
            try:
                is_valid, clean_seq, error_msg = validate_dna_sequence(seq_val)
                if not is_valid:
                    logger.warning(f"Skipping invalid: {name}: {error_msg}"); results.append({'Sequence_Name': name, 'Error': error_msg}); continue
                protein_seq = translate_dna(clean_seq)
                result_data = {'Sequence_Name': name, 'Original_DNA': clean_seq, 'Protein': protein_seq}
                # Logic from Step 1 for choice handling
                if choice == 1:
                    optimized_seq = codon_optimize(protein_seq); weights, _ = get_codon_weights_row(optimized_seq)
                    result_data.update({'Optimized_DNA': optimized_seq, 'CAI_Weights': ','.join(f"{w:.4f}" for w in weights), 'Analysis_Type': 'Standard Codon Optimization'})
                elif choice == 2:
                    weights, codons_list = get_codon_weights_row(clean_seq)

                    # Build individual DataFrame for the sequence
                    df = pd.DataFrame({
                        'DNA_Codon': codons_list,
                        'CAI_Weight': weights,
                        'Amino_Acid': [genetic_code.get(c, '?') for c in codons_list]
                    })

                    # Add MultiIndex column header: top = sequence name, bottom = data type
                    df.columns = pd.MultiIndex.from_product([[name], df.columns])

                    # Add position column separately only once
                    if 'combined_df' not in globals() or combined_df.empty:
                        combined_df = pd.DataFrame({'Position': list(range(1, len(codons_list) + 1))})

                    # Concatenate side-by-side
                    combined_df = pd.concat([combined_df, df], axis=1)

                    # Add summary info too (optional)
                    result_data.update({
                        'Optimized_DNA': clean_seq,
                        'CAI_Weights': ','.join(f"{w:.6f}" for w in weights),
                        'Analysis_Type': 'CAI Weight Analysis (Detailed)'
                    })

                    # Save Excel only after all sequences are processed
                    if len(results) + 1 == len(sequences):  # Add +1 because this iteration hasn't appended yet
                        output_filename = "Codon_CAI_SideBySide.xlsx"
                        with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
                            combined_df.to_excel(writer, index=False)

                        import os
                        os.startfile(output_filename)


                elif choice == 3:
                    optimized_seq = balanced_optimisation(clean_seq); weights, _ = get_codon_weights_row(optimized_seq)
                    result_data.update({'Optimized_DNA': optimized_seq, 'CAI_Weights': ','.join(f"{w:.4f}" for w in weights), 'Analysis_Type': 'Balanced Optimization'})
                elif choice == 4:
                    optimized_seq = nc_stop_codon_optimisation(clean_seq); weights, _ = get_codon_weights_row(optimized_seq)
                    result_data.update({'Optimized_DNA': optimized_seq, 'CAI_Weights': ','.join(f"{w:.4f}" for w in weights), 'Analysis_Type': 'NC Stop Codon Optimization'})
                elif choice == 5:
                    optimized_seq = JT_Plus1_Stop_Optimized(clean_seq); weights, _ = get_codon_weights_row(optimized_seq)
                    result_data.update({'Optimized_DNA': optimized_seq, 'CAI_Weights': ','.join(f"{w:.4f}" for w in weights), 'Analysis_Type': 'JT Plus1 Stop Optimization'})
                elif choice == 6: 
                    stop_counts = number_of_plus1_stops(clean_seq)
                    start_pos, end_pos = find_coding_sequence_bounds(clean_seq)
                    slippery_count = number_of_Slippery_motifs(clean_seq)
                    
                    # Calculate coding sequence info
                    if start_pos is not None and end_pos is not None:
                        coding_length = end_pos - start_pos
                        plus1_len = coding_length // 3
                        coding_info = f"{start_pos}-{end_pos}"
                        coding_status = "Complete CDS"
                    elif start_pos is not None:
                        # No stop codon found, use full sequence from start
                        coding_length = len(clean_seq) - start_pos
                        plus1_len = coding_length // 3
                        coding_info = f"{start_pos}-end"
                        coding_status = "No stop found"
                    else:
                        plus1_len = 0
                        coding_info = "No start found"
                        coding_status = "No valid CDS"
                        coding_length = 0
                    
                    # Calculate densities
                    stop_density = stop_counts.get('total', 0) / max(1, plus1_len) if plus1_len > 0 else 0
                    slippery_density = slippery_count / max(1, plus1_len) if plus1_len > 0 else 0
                    
                    result_data.update({
                        'Plus1_TAA_Count': stop_counts.get('TAA', 0),
                        'Plus1_TAG_Count': stop_counts.get('TAG', 0),
                        'Plus1_TGA_Count': stop_counts.get('TGA', 0),
                        'Plus1_Total_Stops': stop_counts.get('total', 0),
                        'Slippery_Motifs': slippery_count,
                        'Coding_Sequence_Range': coding_info,
                        'Coding_Length_bp': coding_length,
                        
                    })

                results.append(result_data)
            except Exception as e:
                logger.error(f"Error processing {name}: {e}", exc_info=True); results.append({'Sequence_Name': name, 'Error': str(e)})
        
        if results:
            all_keys_initial = set().union(*(d.keys() for d in results))
            
            final_ordered_keys = []
            if choice == 6:
                # Define the specific order for choice 6
                preferred_order = [
                    'Sequence_Name', 'Plus1_TGA_Count', 'Plus1_TAA_Count', 'Plus1_TAG_Count', 
                    'Plus1_Total_Stops'
                ]
                # Start with preferred keys that are actually present
                final_ordered_keys.extend([key for key in preferred_order if key in all_keys_initial])
                # Add remaining keys, sorted, ensuring no duplicates from preferred_order
                remaining_keys = sorted([key for key in all_keys_initial if key not in final_ordered_keys])
                final_ordered_keys.extend(remaining_keys)
            else:
                # Default sorting for other choices (e.g., alphabetical)
                # Or maintain a specific default order if needed for other choices.
                # For now, using sorted list of all keys as a general default.
                final_ordered_keys = sorted(list(all_keys_initial))

            # Create standardized_results with the desired key order for each dictionary
            # This helps ensure pandas DataFrame respects this order (especially Python 3.7+)
            standardized_results_ordered = []
            for original_dict in results: # 'results' is the list of dicts from processing
                new_ordered_dict = {}
                for key in final_ordered_keys:
                    new_ordered_dict[key] = original_dict.get(key, 'N/A')
                standardized_results_ordered.append(new_ordered_dict)
            
            # batch_chart_path = None # Comment from previous logic, not directly relevant here
            if choice == 6: 
                sequences_with_stops_data = []
                for res_dict in results: # Iterate over original results which have richer data types
                    if res_dict.get('Analysis_Type') == 'Plus1 Frame Stop Count Analysis' and 'Plus1_Total_Stops' in res_dict:
                        try:
                            total_stops = int(res_dict.get('Plus1_Total_Stops', 0))
                            if total_stops > 0:
                                sequences_with_stops_data.append({
                                    'name': res_dict.get('Sequence_Name', 'Unknown'),
                                    'taa': int(res_dict.get('Plus1_TAA_Count', 0)),
                                    'tag': int(res_dict.get('Plus1_TAG_Count', 0)),
                                    'tga': int(res_dict.get('Plus1_TGA_Count', 0)),
                                    'total': total_stops
                                })
                        except ValueError:
                            logger.warning(f"Could not parse stop counts for {res_dict.get('Sequence_Name', 'Unknown')} in batch chart prep.")
                            continue
                
                charts_shown = False
                if sequences_with_stops_data:
                    num_sequences_to_chart = len(sequences_with_stops_data)

                    if num_sequences_to_chart == 1:
                        seq_data = sequences_with_stops_data[0]
                        single_plt_title = f"+1 Stop Codon Distribution for {seq_data['name']}"
                        single_temp_labels = ['TAA', 'TAG', 'TGA']
                        single_temp_sizes = [seq_data['taa'], seq_data['tag'], seq_data['tga']]
                        
                        single_chart_labels = [l for i, l in enumerate(single_temp_labels) if single_temp_sizes[i] > 0]
                        single_chart_sizes = [s for s in single_temp_sizes if s > 0]

                        # Define specific colors for each codon
                        color_map = {
                            'TAA': '#111d94',  # Blue
                            'TAG': '#752db0',  # Purpe
                            'TGA': '#b34c9d'   # Pink
                        }

                        valid_colorsII = [color_map[label] for label in single_chart_labels]
                        valid_colors = [color_map[label] for label in valid_chart_labels]

                        if not single_chart_sizes:
                            logger.info(f"Sequence {seq_data['name']} has no +1 stop codons (TAA, TAG, TGA) to display.")
                        else:
                            try:
                                fig, ax = plt.subplots()
                                
                                wedges, texts, autotexts = ax.pie(current_sizes, colors=current_colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
                                ax.legend(wedges, current_labels, title="+1 Stop Codons", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, title_fontsize=9)
                                ax.axis('equal')
                                plt.title(single_plt_title)
                                plt.pie(valid_chart_sizes, labels=valid_chart_labels, colors=valid_colors, autopct='%1.1f%%', startangle=90)
                                plt.show()
                                plt.close(fig)
                                logger.info(f"Batch chart displayed for single sequence: {single_plt_title}")
                                charts_shown = True
                            except Exception as e_single_chart:
                                logger.error(f"Single batch chart display failed: {e_single_chart}")
                    
                    elif num_sequences_to_chart > 1:
                        num_cols = 4  # Or min(3, num_sequences_to_chart)
                        num_rows = math.ceil(num_sequences_to_chart / num_cols)
                        fig_height = 4 * num_rows 
                        fig_width = 5 * num_cols

                        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
                        axes = axes.flatten()

                        for i, seq_data in enumerate(sequences_with_stops_data):
                            ax = axes[i]
                            current_title = f"{seq_data['name']}"
                            current_labels_all = ['TAA', 'TAG', 'TGA']
                            current_sizes_all = [seq_data['taa'], seq_data['tag'], seq_data['tga']]
                            
                            color_map = {
                                'TAA': '#111d94',  # Blue
                                'TAG': '#752db0',  # Purple
                                'TGA': '#b34c9d'   # Pink
                            }

                            # Legend: always include all codons
                            legend_labels = ['TAA', 'TAG', 'TGA']
                            legend_colors = [color_map[label] for label in legend_labels]
                            legend_patches = [plt.matplotlib.patches.Patch(color=color, label=label) for label, color in zip(legend_labels, legend_colors)]

                            # Pie chart: only include non-zero codons
                            current_labels = [l for idx, l in enumerate(legend_labels) if current_sizes_all[idx] > 0]
                            current_sizes = [s for s in current_sizes_all if s > 0]
                            valid_colors = [color_map[label] for label in current_labels]


                            if not current_sizes:
                                ax.text(0.5, 0.5, 'No TAA/TAG/TGA\n+1 stop codons', ha='center', va='center', fontsize=5)
                                ax.set_title(current_title, fontsize=5)
                                ax.axis('off')
                                continue

                            wedges, _ = ax.pie(current_sizes, colors=valid_colors, startangle=90, textprops={'fontsize': 7.5})
                            ax.legend(handles=legend_patches, title="+1 Stops", loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=5, title_fontsize=6)
                            ax.axis('equal')
                            ax.set_title(current_title, fontsize=5)
                        
                        for j in range(num_sequences_to_chart, len(axes)):
                            axes[j].axis('off') # Hide unused subplots

                        fig.suptitle("", fontsize=10, y=0.5) # Adjusted y for suptitle
                        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust rect to make space for suptitle
                        try:
                            plt.show()
                            plt.close(fig)
                            logger.info("Composite batch chart displayed for multiple sequences.")
                            charts_shown = True
                        except Exception as e_composite_chart:
                            logger.error(f"Composite batch chart display failed: {e_composite_chart}")
                else:
                    logger.info("No sequences with +1 stop data to chart in batch mode.")
                    
                if charts_shown: # Only reset icon if a chart was actually shown
                    set_macos_app_icon(root, logger)
            
            # For choice 6, chart_image_path will be None, so no chart is passed to Excel.
            filename = save_output_to_excel(standardized_results_ordered, "Batch_Optimization_Results.xlsx", chart_image_path=None) # Pass None for choice 6 chart
            
            # No batch_chart_path to clean up as it's not created for choice 6 for file saving

            if filename: messagebox.showinfo("Batch Complete", f"Processed {len(results)} sequences.\nSaved to: {filename}", parent=parent_window)
        else: messagebox.showwarning("Batch Failed", "No sequences processed.", parent=parent_window)
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        messagebox.showerror("Batch Error", f"Error: {e}", parent=parent_window)

# --- Settings GUI ---
def show_settings(): # Full version
    global root, config, logger, DEFAULT_CONFIG
    settings_window = tk.Toplevel(root if root and root.winfo_exists() else None)
    settings_window.title("Codon Optimizer Settings"); settings_window.geometry("500x400"); settings_window.resizable(True, True)
    notebook = ttk.Notebook(settings_window); notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    general_frame, paths_frame, algo_frame = ttk.Frame(notebook), ttk.Frame(notebook), ttk.Frame(notebook)
    notebook.add(general_frame, text="General"); notebook.add(paths_frame, text="File Paths"); notebook.add(algo_frame, text="Algorithms")
    
    auto_open_var = tk.BooleanVar(value=config.get("auto_open_files", True))
    ttk.Label(general_frame, text="Auto-open output files:").pack(anchor=tk.W, pady=5); ttk.Checkbutton(general_frame, variable=auto_open_var).pack(anchor=tk.W)
    
    codon_file_var = tk.StringVar(value=config.get("codon_file_path", "")); ttk.Label(paths_frame, text="Codon usage file:").pack(anchor=tk.W, pady=5)
    ttk.Entry(paths_frame, textvariable=codon_file_var, width=50).pack(anchor=tk.W, pady=2)
    def browse_codon_file():
        file_path = filedialog.askopenfilename(title="Select codon file", filetypes=[("Excel","*.xlsx")], parent=settings_window)
        if file_path: codon_file_var.set(file_path)
    ttk.Button(paths_frame, text="Browse...", command=browse_codon_file).pack(anchor=tk.W, pady=2)
    
    output_dir_var = tk.StringVar(value=config.get("default_output_dir", ".")); ttk.Label(paths_frame, text="Default output directory:").pack(anchor=tk.W, pady=5)
    ttk.Entry(paths_frame, textvariable=output_dir_var, width=50).pack(anchor=tk.W, pady=2)
    def browse_output_dir():
        dir_path = filedialog.askdirectory(title="Select output directory", parent=settings_window)
        if dir_path: output_dir_var.set(dir_path)
    ttk.Button(paths_frame, text="Browse...", command=browse_output_dir).pack(anchor=tk.W, pady=2)
    
    bias_weight_var = tk.DoubleVar(value=config.get("bias_weight", BIAS_WEIGHT_DEFAULT))
    ttk.Label(algo_frame, text="Bias weight for balanced optimization:").pack(anchor=tk.W, pady=5)
    ttk.Scale(algo_frame, from_=0.1, to=5.0, variable=bias_weight_var, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
    current_bias_label = ttk.Label(algo_frame, text=f"{bias_weight_var.get():.2f}"); current_bias_label.pack(anchor=tk.W)
    def update_bias_label(val): current_bias_label.config(text=f"{float(val):.2f}")
    bias_weight_var.trace_add("write", lambda n,i,m,v=bias_weight_var: update_bias_label(v.get())); update_bias_label(bias_weight_var.get())

    button_frame = ttk.Frame(settings_window); button_frame.pack(fill=tk.X, padx=10, pady=10)
    def save_settings_action(): 
        global config; config.update({
            "auto_open_files": auto_open_var.get(), "codon_file_path": codon_file_var.get(),
            "default_output_dir": output_dir_var.get(), "bias_weight": bias_weight_var.get()})
        save_config(config); messagebox.showinfo("Settings", "Settings saved!", parent=settings_window); settings_window.destroy()
    def reset_settings_action():
        global config; config = DEFAULT_CONFIG.copy(); save_config(config)
        auto_open_var.set(config["auto_open_files"]); codon_file_var.set(config["codon_file_path"])
        output_dir_var.set(config["default_output_dir"]); bias_weight_var.set(config["bias_weight"])
        messagebox.showinfo("Settings", "Settings reset to defaults!", parent=settings_window)
    ttk.Button(button_frame, text="Save", command=save_settings_action).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Reset", command=reset_settings_action).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    settings_window.transient(root if root and root.winfo_exists() else None); settings_window.grab_set(); settings_window.wait_window()

# --- save_accumulated_results_action (New for Step 4) ---
def save_accumulated_results_action(): # Full version
    global accumulated_results_list, accumulation_run_id_counter, config, logger, root
    parent_window = root if root and root.winfo_exists() else None
    if not accumulated_results_list:
        messagebox.showwarning("No Results", "No results accumulated.", parent=parent_window); return
    output_dir = config.get("default_output_dir", ".")
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    file_path = filedialog.asksaveasfilename(initialdir=output_dir, title="Save Combined As", defaultextension=".xlsx", filetypes=[("Excel","*.xlsx")], parent=parent_window)
    if not file_path: return
    flat_list_of_rows = [row for run_list in accumulated_results_list for row in run_list]
    if not flat_list_of_rows:
        messagebox.showwarning("Empty Results", "Accumulated results are empty.", parent=parent_window); return
    try:
        all_keys = set().union(*(d.keys() for d in flat_list_of_rows))
        standardized_list = [{key: d.get(key, None) for key in all_keys} for d in flat_list_of_rows]
        df = pd.DataFrame(standardized_list)
        preferred_order = ['Accumulation_Run_ID', 'Analysis_Type_In_Run', 'Sequence_Name', 'Original_DNA', 'Protein', 'Optimized_DNA', 'Balanced_Optimized_DNA', 'NC_Stop_Optimized_DNA', 'JT_Plus1_Optimized_DNA', 'CAI_Weights', 'Plus1_Stop_Counts', 'Position', 'DNA_Codon', 'CAI_Weight', 'Amino_Acid', 'Stop_Codon_Type', 'Count_in_Plus1_Frame', 'Sequence_Info', 'Error']
        final_columns = [col for col in preferred_order if col in df.columns] + [col for col in df.columns if col not in preferred_order]
        df = df[final_columns]
        df.to_excel(file_path, index=False)
        logger.info(f"Combined results saved to {file_path}")
        messagebox.showinfo("Save Complete", f"Saved to {file_path}", parent=parent_window)
        if messagebox.askyesno("Clear Accumulated Results", "Clear buffer?", parent=parent_window):
            accumulated_results_list, accumulation_run_id_counter = [], 0; logger.info("Accumulated results cleared.")
    except Exception as e:
        logger.error(f"Error saving combined: {e}", exc_info=True)
        messagebox.showerror("Save Error", f"Could not save: {e}", parent=parent_window)

# --- Main Application (Modified for Step 4) ---
def create_main_menu(): # Full version
    global root, combine_outputs_var, config, logger
    root = tk.Tk()
    set_macos_app_icon(root, logger) # Set initial icon
    
    root.title("DNA Codon Optimization Tool - v1"); root.geometry("600x700")
    main_frame = ttk.Frame(root, padding="20"); main_frame.pack(fill=tk.BOTH, expand=True)
    ttk.Label(main_frame, text="JT mRNA Tool", font=("Arial", 16, "bold")).pack(pady=10)
    desc_text = "* Standard optimization\n* CAI analysis\n* Sequence Analysis\n* Specialized +1 stops\n* Batch processing" # Corrected
    ttk.Label(main_frame, text=desc_text, justify=tk.LEFT).pack(pady=10)
    buttons_frame = ttk.Frame(main_frame); buttons_frame.pack(pady=20)
    def start_single_optimization():
        clean_seq, protein_seq, choice_val = get_user_input()
        if clean_seq and protein_seq and choice_val: run_optimization(clean_seq, protein_seq, choice_val)
    ttk.Button(buttons_frame, text="Single Sequence Optimization", command=start_single_optimization, width=30).pack(pady=5)
    combine_outputs_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(buttons_frame, text="Accumulate single optimization results", variable=combine_outputs_var, width=30).pack(pady=5)
    ttk.Button(buttons_frame, text="Save Accumulated Results", command=save_accumulated_results_action, width=30).pack(pady=5)
    ttk.Button(buttons_frame, text="Batch Processing", command=process_batch_sequences, width=30).pack(pady=5)
    ttk.Button(buttons_frame, text="Settings", command=show_settings, width=30).pack(pady=5)
    info_frame = ttk.LabelFrame(main_frame, text="Information", padding="10"); info_frame.pack(fill=tk.X, pady=10)
    info_text_content = f"Codon file: {config.get('codon_file_path', 'N/S')}\nOutput dir: {config.get('default_output_dir', '.')}\nLog: codon_optimizer.log"
    ttk.Label(info_frame, text=info_text_content, justify=tk.LEFT).pack()
    ttk.Button(main_frame, text="Exit", command=lambda: safe_exit(0)).pack(pady=10)
    root.protocol("WM_DELETE_WINDOW", lambda: safe_exit(0))
    return root

def main(): # Full version
    global logger, root
    try:
        logger.info("Starting Full DNA Codon Optimization Tool")
        if len(sys.argv) > 1: print("No CLI mode."); sys.exit(1)
        create_main_menu()
        if root: root.mainloop()
        else: logger.error("Failed to create root window.")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        parent_error_dialog = None
        try:
            if 'root' in globals() and root and root.winfo_exists(): parent_error_dialog = root
            else: temp_err_root = tk.Tk(); temp_err_root.withdraw(); parent_error_dialog = temp_err_root
            messagebox.showerror("Fatal Error", f"A fatal error occurred: {e}", parent=parent_error_dialog)
            if parent_error_dialog is not root and hasattr(parent_error_dialog, 'destroy') and parent_error_dialog.winfo_exists(): parent_error_dialog.destroy()
        except Exception as me: print(f"FATAL ERROR (messagebox failed): {e}, secondary: {me}")
        sys.exit(1)

if __name__ == "__main__":
    main()
