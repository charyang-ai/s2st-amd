Meta just released a large model for language but their demo doesn't work out of the box locally on Mac Meta architecture.
So I made this modified version that does.

# Installation
## Install uv
- powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
## Create virtual env with uv
- uv venv <env name> 

## Install Torch
### Setup with PyTorch-rocm6.4
- pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4 
### Setup with Torch-CPU
- pip3 install torch torchvision torchaudio
### Setup with Torch-directml
- pip3 install torch-directml==0.2.5.dev240914
- pip3 install torchvision==2.4.1 torchaudio==2.4.1

## Install Other things
- pip3 install git+https://github.com/huggingface/transformers.git sentencepiece
- pip3 install -r requirements.txt
  
# How to run
```
export HIP_VISIBLE_DEVICES=0
python gradio_gpu.py --share
```
Do not forget to export the variable or it will crash! Hopefully Apple will keep working with pytorch for full support

# Download the models
The script will use your HF_TOKEN to download the models, but if you want to download only exactly the file needed you can also download from https://huggingface.co/facebook/seamless-m4t-v2-large/tree/main the following files:
- seamlessM4T_v2_large.pt
- spm_char_lang38_tc.model
- vocoder_v2.pt

# Caveat
Unfortunately due to the poor support on the metal architecture the model isn't as precise as it is running on CUDA
