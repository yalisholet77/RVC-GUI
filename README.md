<div align="center">

<h1>RVC GUI is a fork of RVC "Retrieval-based-Voice-Conversion-WebUI"
<br><br>
  
  ## Voice and audio infering only


  

  
[![Licence](https://img.shields.io/github/license/liujing04/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/%E4%BD%BF%E7%94%A8%E9%9C%80%E9%81%B5%E5%AE%88%E7%9A%84%E5%8D%8F%E8%AE%AE-LICENSE.txt)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

</div>

#### GUI

![GUI](https://github.com/Tiger14n/RVC-GUI/raw/main/docs/GUI1.JPG)

## Preparing the environment


The following commands need to be executed in the environment of Python version 3.8 or higher:
```bash
# Install PyTorch-related core dependencies, skip if installed
# Reference: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

#For Windows + Nvidia Ampere Architecture(RTX30xx), you need to specify the cuda version corresponding to pytorch according to the experience of https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/issues/21
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117


```


**Notice**: `faiss 1.7.2` will raise Segmentation Fault: 11 under `MacOS`, please use `pip install faiss-cpu==1.7.0` if you use pip to install it manually.

```bash
python -m pip install -U pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Downlaod [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt/) and place it in the root folder

```
#If you are using Windows, you may also need this dictionary, skip if FFmpeg is installed
ffmpeg.exe
```
Then use this command to start RVC GUI:
```bash
python rvcgui.py
```


## Credits
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)

