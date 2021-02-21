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

### Usage
The original authors main thought process was this. It follows the original 3 step process of face recognition:
1. Face Detection (+ cropping and alignment)  
2. Generating Face Embeddings  
3. Calculating Difference between embeddings of different images

Here he uses MTCNN to do face detection, cropping and alignment is just the process to convert the face into
into a homogeneous shape/ size for training/ inference. Then he does a prewhitening step (preprocessing) which supposedly in an attempt to homogenise. Following that, he will generate enmbeddings for the face. This embeddings generated is meant to then be compared across a database of other embeddings, typically using the Eucludian Distance metric. it then just give you a value which should not be attempted to be converted into a percentage value because MATH!  

For now theres only **4 main scripts** to use:  
- decode images:
I did NOT get this to work so don't ask me how to do it yet, i think there were problems with me downloading it  
- download_models.py
This is just a wrapper around his download and extract models script, but you can just run it and put in the MODEL_DIR to tell it where to downlaod the models from. more info about the mdoels from prev_README.md or check out the original post  
- face_detection.py  
This is supposed to be part of the main app that does the face detection (and alignment). I have written some notes that i have received as I was cleaning it up.  
- src/compare.py
This allows you to make comparison (using Euclidean Distance) between multiple  (>=2) images. This is good because it also helps you to generate a confusion matrix!! Use my compare.sh (or.bat for windows users) file to access this script. All you need to do is to change the path to the model and images. first argument is model and every subsequent argument your input will be considered as the images unless you add in additional flags  

### Remarks
