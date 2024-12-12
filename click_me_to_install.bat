curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
.\miniconda.exe
del miniconda.exe
conda create -n angle-detector
conda activate angle-detector
conda install python=3.10.12
python -m pip install --upgrade pip
pip install -r requirements.txt