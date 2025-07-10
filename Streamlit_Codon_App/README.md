# DNA Codon Optimization Tool - Streamlit Version

A professional bioinformatics application for DNA sequence optimization and analysis, featuring multiple optimization algorithms and comprehensive analytical capabilities.

## Features

### Optimization Methods
- **Standard Codon Optimization**: Uses most frequent codons for each amino acid
- **CAI Weight Analysis**: Calculates Codon Adaptation Index for sequence quality assessment
- **Balanced Optimization**: Advanced algorithm considering codon usage and +1 frame effects
- **NC Stop Codon Optimization**: Specialized for alternative reading frame stop codon creation
- **JT Plus1 Stop Optimization**: Creates specific stop motifs in +1 frame
- **Sequence Analysis**: Comprehensive analysis including slippery motifs and frame analysis

### Research & Patents (NEW!)
- **Patent Search**: AI-powered search of Google Patents for molecular biology technologies
- **NCBI Sequence Search**: Google-powered search of NCBI nucleotide database
- **AI-Powered Analysis**: Anthropic Claude integration for intelligent sequence and patent analysis
- **Research Integration**: Analyze NCBI sequences with codon optimization tools
- **Unified API**: Both searches use SERPER API for consistent, reliable results

### Capabilities
- Single sequence processing with real-time validation
- Batch processing for multiple sequences (FASTA or text format)
- Result accumulation and management
- Professional Excel export functionality
- Interactive visualizations for analysis results
- Configurable algorithm parameters

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup
1. Clone or download the application files
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### API Configuration (Optional - for Research & Patents)

To use the new Research & Patents functionality, set up API keys:

1. **Create a `.env` file** in the project directory:
   ```bash
   # SERPER API Key for patent search (free tier: 2,500 searches)
   SERPER_API_KEY=your_serper_api_key_here

   # Anthropic API Key for AI analysis
   ANTHROPIC_API=your_anthropic_api_key_here
   ```

2. **Get API Keys:**
   - **SERPER API**: Free account at https://serper.dev/
   - **Anthropic API**: Account at https://console.anthropic.com/

3. **See `RESEARCH_SETUP.md`** for detailed setup instructions

## Usage

### Starting the Application
```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Required Files
Before using the application, you'll need a codon usage frequency file:
- Format: Excel (.xlsx)
- Required columns: `triplet`, `amino_acid`, `fraction`
- Example: `HumanCodons.xlsx`

Upload this file using the "Upload Codon Usage File" option in the sidebar.

### Basic Workflow

#### Single Sequence Optimization
1. Upload your codon usage file in the sidebar
2. Navigate to the "Single Sequence" tab
3. Paste your DNA sequence in the text area
4. Select an optimization method
5. Click "Run Optimization"
6. View results and download if needed

#### Batch Processing
1. Navigate to the "Batch Processing" tab
2. Upload a sequence file (FASTA or text format)
3. Select optimization method
4. Click "Process Batch"
5. Download batch results

#### Result Management
- Enable "Accumulate Results" in sidebar to collect multiple single-sequence results
- View accumulated results in the "Accumulated Results" tab
- Download combined results as Excel files

### Input Requirements

#### DNA Sequences
- Valid bases: A, T, G, C (U will be converted to T)
- Spaces and newlines are automatically removed
- Sequence length should be a multiple of 3 for optimal results

#### Batch Files
- **FASTA format**: Standard FASTA with `>header` lines
- **Text format**: One sequence per line

## Configuration

### Algorithm Settings
- **Bias Weight**: Adjusts the weight for +1 frame stop codon bias in balanced optimization (0.1 - 5.0)

### Output Options
- Results are provided as Excel files with professional formatting
- All downloads include comprehensive metadata and analysis details

## Technical Details

### Optimization Algorithms

#### Standard Codon Optimization
Uses the most frequently used codon for each amino acid based on the provided codon usage table.

#### Balanced Optimization
Advanced algorithm that considers:
- Codon usage frequency
- +1 frame stop codon creation
- Two-codon and single-codon substitution strategies

#### Sequence Analysis
Provides comprehensive analysis including:
- Coding sequence identification (ATG start, stop codons)
- +1 frame stop codon counting
- Slippery motif detection (TTTT, TTTC)
- Statistical metrics and visualizations

### Performance
- Real-time sequence validation
- Efficient batch processing with progress tracking
- Caching for improved performance on repeated operations

## File Structure
```
├── streamlit_app.py          # Main application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── v1.8.py                  # Original Tkinter version (reference)
└── HumanCodons.xlsx         # Codon usage data (required)
```

## Dependencies
- streamlit
- pandas
- numpy
- matplotlib
- biopython
- openpyxl
- xlsxwriter

## Version History
- **Streamlit v1.0**: Complete conversion from Tkinter to web-based interface
- **Tkinter v1.8**: Original desktop application version

## Support
For issues or questions, refer to the "About" tab within the application for detailed method descriptions and requirements.

## License
This tool is designed for academic and research use in bioinformatics and molecular biology applications. 