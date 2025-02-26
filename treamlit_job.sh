#!/bin/bash
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --job-name=streamlit

# Load any necessary modules if required (e.g., Python)
# module load python/3.11

# Navigate to your project folder
cd ~/Assignment_3_SA

# Activate your virtual environment
source venv/bin/activate

# Run the Streamlit app on a chosen port (e.g., 8503)
streamlit run A3.py --server.address 0.0.0.0 --server.port 8503