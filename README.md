# Face Recognition using Tensorflow 
This is not the original repo but an adaption made by ollayf to utilise the tools provided by this repo based on our project needs

If you want the original repo you can look at [David Sandberg's Facenet](https://github.com/davidsandberg/facenet)  

I have decided to adapt this repo as it is very outdated, alot of the libraries it originally uses cannot be used by me or requires alot of time (wasted) to relearn, so I have adapted it to be compatible with the newer libraries so you can use it too  

However this also means that you need to clean ALOT of the code up, so far, I have only cleaned up: 
src.compare and src.facenet

## Setup
### Installing dependencies
For ubuntu distros, you cannot install CUDA 9.0 unless u have around 14.04-16.04  
This also means that all your libraries will be VERY OLD

#### Ubuntu 18.04
CUDA=10.0  
CUDNN=7.4  
python3.5  
python3.5-tk  
python3.5-dev  

#### Windows
Probably not that complicated just install the python libraries

#### Python Libraries
This is NOT the same as the ones by Andy Sandberg, but this allows
```
pip install -r yf_reqs.txt
```

### Download Open Source Datasets
I used MS CELEB 1M dataset for this. In order to receive it you need to torrent from [MSCELEB1M Dataset](https://academictorrents.com/details/9e67eb7cc23c9417f39778a8e06cca5e26196a97/tech&hit=1&filelist=1)  
Then decode the datasets using  
```
python decode tsvs.py
```
You can even use this repo [Clean MSCELEB1M](https://github.com/EB-Dodo/C-MS-Celeb) to clean up the dataset

### Download Pre-trained Datasets
To download the pre-trained models by the original user:  
```
python download_models.py
```
You can configure the directory to download the models within the script  
