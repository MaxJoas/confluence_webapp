# Confluence Webapp
TODO

## Requirements:
- Python 3.10
- pip 24.02
- GCC 5.3.0
# Installation

## Installation on Windows with Conda
- clone this repo
- runn the following commands:
```
conda create -n <your-env> python=3.10 pip
conda activate <your-env>
conda install -c conda-forge m2w64-toolchain
python -m pip install torch==1.13.0
python -m pip install -r requirements.txt
hatch build

```

## Installation wiht Python venv 
- clone repo
- pip install torch==1.13.0
- pip install -e .
- get models and sample data
- TODO prepare-all.py
## Usage
- Download the trained models from [here](https://cloud.scadsai.uni-leipzig.de/index.php/f/14097626)
- Run the app with:
  
```
streamlit run confluence_webapp/src/app.py models/lc_models/unet_model_final.pth models/lc_models/d2_model_final.pth models/lc_models/sam_model_final.pth --theme.base light --theme.primaryColor blue
```

## Demo

![Saxocellwebapp](https://github.com/user-attachments/assets/d30ce3b3-1b86-40c7-b41a-aaa35471b1ca)
