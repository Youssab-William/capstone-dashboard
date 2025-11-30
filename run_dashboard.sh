#!/bin/bash
# Script to run the Streamlit dashboard

cd "$(dirname "$0")"
echo "Starting LLM Tone Effects Dashboard..."
echo "The dashboard will open in your browser automatically."
echo "If it doesn't, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Use Python module approach to avoid CLI issues
python -m streamlit run dashboard/streamlit_app.py

