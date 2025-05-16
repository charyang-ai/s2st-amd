Meta just released a large model for language but their demo doesn't work out of the box locally on Mac Meta architecture.
So I made this modified version that does.

# How to run
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
python app.py
```
Do not forget to export the variable or it will crash! Hopefully Apple will keep working with pytorch for full support

# Download the models
The script will use your HF_TOKEN to download the models, but if you want to download only exactly the file needed you can also download from https://huggingface.co/facebook/seamless-m4t-v2-large/tree/main the following files:
- seamlessM4T_v2_large.pt
- spm_char_lang38_tc.model
- vocoder_v2.pt

# Caveat
Unfortunately due to the poor support on the metal architecture the model isn't as precise as it is running on CUDA