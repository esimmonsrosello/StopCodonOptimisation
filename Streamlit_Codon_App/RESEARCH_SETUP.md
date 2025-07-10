# Research & Patents Setup Guide

## Overview

Your DNA Codon Optimization Tool now includes a powerful **Research & Patents** tab that integrates:

1. **Patent Search**: AI-powered search of Google Patents for molecular biology technologies
2. **NCBI Sequence Search**: Search NCBI nucleotide database via Google for DNA/RNA sequences
3. **AI Analysis**: Anthropic Claude integration for intelligent analysis
4. **Sequence Integration**: Analyze NCBI sequences with your codon optimization tools

**Note**: Both patent and NCBI searches use the same SERPER API for consistent, reliable results.

## API Setup Required

To use the new research functionality, you need to set up API keys in a `.env` file:

### 1. Create `.env` file

Create a file named `.env` in your project directory with the following content:

```bash
# SERPER API Key for patent search functionality
# Get your free API key from https://serper.dev/
SERPER_API_KEY=your_serper_api_key_here

# Anthropic API Key for AI-powered analysis  
# Get your API key from https://console.anthropic.com/
ANTHROPIC_API=your_anthropic_api_key_here
```

### 2. Get SERPER API Key (Free)

1. Go to [https://serper.dev/](https://serper.dev/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Replace `your_serper_api_key_here` in your `.env` file

**Free tier includes**: 2,500 free searches

### 3. Get Anthropic API Key

1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Create an account and verify your phone number
3. Get your API key from the dashboard
4. Replace `your_anthropic_api_key_here` in your `.env` file

**Note**: Anthropic requires payment for API usage

## Features

### Patent Search Tab

- **AI-Powered Search**: Search Google Patents with intelligent analysis
- **Molecular Biology Focus**: Specialized for DNA, RNA, and biotechnology patents
- **Comprehensive Results**: Patent IDs, titles, summaries, and direct links
- **Claude Analysis**: AI-generated insights about patent technologies

**Example Searches**:
- "Codon optimization methods for protein expression"
- "mRNA vaccine delivery systems"
- "CRISPR gene editing technologies"
- "DNA sequence analysis algorithms"

### NCBI Sequence Search Tab

- **Google-Powered Search**: Search NCBI nucleotide database via Google (site:ncbi.nlm.nih.gov)
- **Full Sequence Retrieval**: Automatically fetch complete DNA/RNA sequences
- **AI Sequence Analysis**: Claude-powered analysis of sequence features
- **Codon Integration**: Analyze sequences with your optimization tools
- **Same API as Patents**: Uses SERPER API for consistent search experience

**Example Searches**:
- "COVID-19 spike protein mRNA"
- "Human insulin gene"
- "BRCA1 tumor suppressor"
- "16S ribosomal RNA"

### Integration Features

- **Sequence Analysis**: Run codon optimization on NCBI sequences
- **Patent Context**: Connect sequences to relevant patents
- **Export Options**: Download results for further analysis
- **Professional Interface**: Clean, scientific presentation

## Usage Examples

### 1. Research mRNA Vaccines

1. **Patent Search**: Search for "mRNA vaccine codon optimization"
2. **Review AI Analysis**: Get insights about optimization techniques
3. **NCBI Search**: Find actual mRNA sequences (e.g., "SARS-CoV-2 spike mRNA")
4. **Analyze Sequences**: Use codon optimizer on retrieved sequences
5. **Compare**: See how patents relate to actual sequence data

### 2. Gene Therapy Research

1. **Patent Search**: "Gene therapy vector optimization"
2. **NCBI Search**: Find target gene sequences
3. **Optimization**: Run balanced optimization on sequences
4. **Patent Context**: Understand existing IP landscape

### 3. Protein Expression

1. **NCBI Search**: Find gene of interest
2. **Codon Analysis**: Analyze existing codon usage
3. **Optimization**: Apply optimization algorithms
4. **Patent Research**: Check for related expression patents

## Troubleshooting

### No Patent Results
- Check SERPER API key is correct
- Try different search terms
- Verify internet connection

### No NCBI Results  
- Check SERPER API key is correct (same as patent search)
- Try broader search terms
- Check sequence names and spellings
- Try different search keywords (e.g., "gene name mRNA" or "protein sequence")

### AI Analysis Not Working
- Verify Anthropic API key is correct
- Check you have API credits available
- Try shorter sequences for analysis

### Application Won't Start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check `.env` file format (no quotes around values)
- Verify API keys are active

## Dependencies

The following new packages were added:

```
requests>=2.28.0
beautifulsoup4>=4.11.0  
python-dotenv>=0.19.0
anthropic>=0.3.0
```

Install with: `pip install -r requirements.txt`

## Privacy & Ethics

- **Google Search**: Both patent and NCBI searches use Google via SERPER API
- **NCBI**: Sequence retrieval respects NCBI terms of service
- **Patents**: Uses public patent information only
- **API Keys**: Stored locally, never transmitted except to respective services
- **AI Analysis**: Sent to Anthropic for processing (check their privacy policy)

## Support

If you encounter issues:

1. Check API key configuration
2. Verify internet connectivity
3. Review application logs
4. Try with simpler search terms
5. Check API service status pages

## Future Enhancements

Planned improvements:
- PubMed literature search integration
- Patent PDF download and analysis
- Sequence similarity search
- Custom patent filtering
- Bulk sequence processing 