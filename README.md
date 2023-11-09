# Mask_RCNN
MaskRCNN code for python 3.8 and TF2.5

## Installation steps:
### 1.Install anaconda software
This step is mandatory because certain python modules have dependency issue when tried to install manually, in conda this issue is not persistant as one can force install those modules. For example the Tensorflow v2.5.0 and python 3.8 throws error when installed without conda
### 2.Create virtual environment:
After installing Anaconda, verify the install by passing conda command in cmd, after installating is successfully completed,
create a virtual environment using this command -> conda create -n virtualenv |virtual_env_name| python=3.8 pip
### 3.Install modules using requirement.txt file:
This step is straight forward as indicated by the title use the provided file to install the modules
### 4.Install git software:
Git software is required to install coco api, pycocotools from github and also to pull MRCNN repo
### 5.Clone MRCNN git repo:
Clone Matter Port's github repository using command
 -> git clone https://github.com/matterport/Mask_RCNN.git
### 6.Install Microsoft visual C++ 2015 Builds Tools:
Mandatory step.
### 7.Clone COCO api git repo:
Use command -> git clone https://github.com/philferriere/cocoapi.git
### 8.Install pycocotools:
Use command -> pip install git+https://github.com/philferriere/cocoapi.get#subdirectory=PythonAPI
### 9.Download the coco weights:
COCO weigths is needed if the training data is less and coco preweigths is needed 
link - https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5



