[![GMU](https://img.shields.io/badge/Collaborator-George_Mason_Univ_(GMU)-brightgreen.svg?style=for-the-badge)](https://jqub.ece.gmu.edu/)
[![LANL](https://img.shields.io/badge/Collaborator-Los_Alamos_National_Laboratory_(LANL)-brightgreen.svg?style=for-the-badge)](https://openfwi-lanl.github.io/)

# FWI Compression 

This is a GMU and LANL collaboration project. The implementation is based on [OpenFWI](https://github.com/lanl/OpenFWI).

## Environment setup
The following packages are required:
- pytorch v1.7.1
- torchvision v0.8.2
- scikit learn
- numpy
- matplotlib (for visualization)
- thop (for op counter)
- torchinfo (for summary of NN)
- einops (for VIT in InvLINT)

## Test the latency of model on Raspberry Pi
First, download the [tutorial dataset](https://drive.google.com/drive/folders/1u4pFvu7UnQozu-tLyyPdsQjRmAZ-kGil) and unzip it into your local directory for this project. 

Second, Run the following command to execute `My_Time_Latency.py` for a single run and measure the latency (seconds).
```python
# For InversionNet
# bs=2, the minimum size due to batch normalization
python My_Time_Latency.py --file-size 120 --batch-size 2 --output-path Invnet_models --device cpu -ds flatvel-tutorial -v tutorial_val.txt --workers 0 -m InversionNet --norm bn --save-name latency_measure

# bs=60
python My_Time_Latency.py --file-size 120 --batch-size 60 --output-path Invnet_models --device cpu -ds flatvel-tutorial -v tutorial_val.txt --workers 0 -m InversionNet --norm bn --save-name latency_measure

# bs=120
python My_Time_Latency.py --file-size 120 --batch-size 120 --output-path Invnet_models --device cpu -ds flatvel-tutorial -v tutorial_val.txt --workers 0 -m InversionNet --norm bn --save-name latency_measure
```

If you want to execute multiple runs, you can execute `My_Time_Latency_script.py` 
```python
python My_Time_Latency_script.py
```
In the starting lines of `My_Time_Latency_script.py`, you can custimize your settings.
```python
batch_size = 60
output_path = 'Invnet_models'
model_name = 'InversionNet'
num_runs = 10
```