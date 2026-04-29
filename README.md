requirements:

python3.10+, pip

opencv-python - CUDA build reccomended (https://github.com/cudawarped/opencv-python-cuda-wheels/releases)

torch - CUDA build reccomended

torchvision

ultralytics

tqdm

numpy==2.2.4

scipy


reccomend a CUDA capable GPU, with CUDA dev tools and cuDNN.
CPU fallback is availabe, but is SLOW!






      

example setup and run:
```git clone https://github.com/FRC-Team-955/2026-StreamScouting.git```

```pip install https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.13.0.90/opencv_contrib_python-4.13.0.90-cp37-abi3-linux_x86_64.whl```

```pip install ultralytics tqdm numpy==2.2.4 scipy```

```pip install torch --index-url https://download.pytorch.org/whl/cu128```

```cd 2026-StreamScouting```

```python3 main.py --side red --video-file Q18.mp4 --frame-drop 1```
