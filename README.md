Meta just released a large model for language but their demo doesn't work out of the box locally on Mac Meta architecture.
So I made this modified version that does.

# Installation
## Install miniforge3
## Create virtual env with conda
- conda create <env> python=3.12

## Install Torch
### If MI/Radeon dGPU, setup with PyTorch-rocm6.4
- pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4 
### If laptop CPU, setup with Torch-CPU
- pip install torch torchvision torchaudio
### If laptop iGPU, setup with Torch-directml
- pip install torch-directml==0.2.5.dev240914
- pip install torchvision==0.19.1 torchaudio==2.4.1

## Install Other things
- pip install -r requirements.txt
- pip install git+https://github.com/huggingface/transformers.git sentencepiece
- pip install protobuf
- pip install soundfile
  
# How to run GUI version with rocm on dGPU
```
export HIP_VISIBLE_DEVICES=0
python gradio_gpu.py --share
```
Do not forget to export the variable or it will crash! Hopefully Apple will keep working with pytorch for full support

# Download the models
The script will use your HF_TOKEN to download the models, but if you want to download only exactly the file needed you can also download from https://huggingface.co/facebook/seamless-m4t-v2-large/tree/main all files.

