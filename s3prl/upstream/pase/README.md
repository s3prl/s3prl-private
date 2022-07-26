python<=3.6
torch<=1.7

1. Modify the cuda version number of **cupy** in `requirements.txt`. The default `cupy-cuda102` presumes the cuda version of **10.2**. You should change `102` according to your cuda version which can be checked by `nvcc -V` or `nvidia-smi`, notice not the cuda version of pytorch.

2. `pip install -r requirements.txt`

3. `import cupy`: test cupy can be imported.

4. `from torchqrnn import QRNN`: test QRNN can be imported.
