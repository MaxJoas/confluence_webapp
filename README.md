# Confluence Webapp
TODO

## Requirements:
- Python 3.10
- pip
- GCC
# Installation

## Installation on Windows with Conda
- clone repo
```
conda create -n <your-env> python=3.10 pip
conda activate <your-env>
conda install -c conda-forge m2w64-toolchain
pip install hatch
hatch build
pip install torch==1.13.0
pip install -r requirements.txt
```

## Installation wiht Python venv 
- clone repo
- pip install torch==1.13.0
- pip install -e .
- get models and sample data
- TODO prepare-all.py
## Usage
- Download the trained models from:
- Run the app with:
  
```
streamlit run confluence_webapp/src/app.py models/lc_models/unet_model_final.pth models/lc_models/d2_model_final.pth models/lc_models/sam_model_final.pth
```

## Demo
![image](https://github.com/user-attachments/assets/4e81d2db-b079-4c84-b44d-19d2beba3028)



