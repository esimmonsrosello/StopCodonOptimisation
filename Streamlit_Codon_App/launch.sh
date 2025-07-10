#!/bin/bash

# DNA Codon Optimization Tool - Launch Script
echo "Starting DNA Codon Optimization Tool..."
echo "================================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run setup first:"
    echo "python3 -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Launch the application
echo "Launching DNA Codon Optimization Tool..."
echo "The application will open in your default web browser."
echo "Use Ctrl+C to stop the application."
echo "================================================"

streamlit run streamlit_app.py 