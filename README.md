# Confluence Webapp
This streamlit web app acompaigns our paper "Zero-Shot, Big-Shot, Active-Shot - How to estimate cell confluence, lazily". It allows users to select one of the four fine-tuned models and upload images.
The model will then detect the cells and calculate the confluence. The confluence report can then be downloaded as CSV file.
The web app is online [here](TODO) or can be installed locally as described above

## Requirements:
- Python 3.10
- pip 24.02
- GCC 5.3.0
- Download the SAM model weights from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
  - and place into the root directory like:
  ```
  | `confluence_webapp
  | ---|src
  | ---|--- app.py
  | ---|--- ...
  | `sam_vit_h_4b8939.pth
  | `pyproject.toml

# Installation

## Installation on Windows with Conda
- clone this repo
- run the following commands:
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
- run the following commands:
```
pip install torch==1.13.0
pip install -e .
hatch build
```
## Usage
- Download the trained models from [here](https://cloud.scadsai.uni-leipzig.de/index.php/f/14097626)
- Run the app with:
  
```
streamlit run confluence_webapp/src/app.py models/lc_models/unet_model_final.pth models/lc_models/d2_model_final.pth models/lc_models/sam_model_final.pth --theme.base light --theme.primaryColor blue
```
This should output the following:
```
Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.


  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.24.114.117:8501
  External URL: http://139.18.241.93:8501

```
Open the Local URL in your browser and start using the web app.

## Demo

![Saxocellwebapp](https://github.com/user-attachments/assets/d30ce3b3-1b86-40c7-b41a-aaa35471b1ca)

## References
```
Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation, page 234–241. 279
Springer International Publishing, 2015. 280

Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. https://github.com/ 293
facebookresearch/detectron2, 2019

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. 290
Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. In 2023 IEEE/CVF International Conference on Computer Vision 291
(ICCV), pages 3992–4003, 2023.

Carsen Stringer, Tim Wang, Michalis Michaelos, and Marius Pachitariu. Cellpose: a generalist algorithm for cellular segmentation. Nature 295
Methods, 18(1):100–106, December 2020.
```




