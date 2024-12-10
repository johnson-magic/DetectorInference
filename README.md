# PrecisionAngleDetectionInference
1. 安装python

* 在Windows Command下安装miniconda
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
.\miniconda.exe
del miniconda.exe
```

2. 安装环境
* 在Anaconda Prompt下通过conda env安装环境
```
conda create -n angle-detector
conda activate angle-detector
conda install python=3.10.12
git clone https://github.com/johnson-magic/PrecisionAngleDetectionInference
cd PrecisionAngleDetectionInference
python -m pip install --upgrade pip
pip install -r requirements.txt
```


3. 启动程序
```
python infer.py
```
