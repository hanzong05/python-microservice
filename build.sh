#!/bin/bash
# Build script for Render deployment

# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p data/input data/output data/backups logs

# Copy cleaned data to input directory
cp cleaned_geotechnical_data.csv data/input/

# Start the geotechnical processor
python process_geotechnical_data.py
