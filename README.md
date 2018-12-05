# MaskGAN.pytorch

A PyTorch attempt at reimplementing 

* MaskGAN: Better Text Generation via Filling in the _______ , William Fedus, Ian Goodfellow, Andrew M. Dai
  [[paper]](https://openreview.net/pdf?id=ByOExmWAb)


# Setting up

#### SentencePiece

I used [google/SentencePiece](https://github.com/google/sentencepiece) to bring down the vocabulary to make training easier. The trained models are available inside this repository. Install the python bindings through pip so the code can use it.

```
python3 -m pip install sentencepiece
```

#### fairseq

This code is build using the basic blocks provided by [pytorch/fairseq](https://github.com/pytorch/fairseq). Please follow instructions there to install fairseq as a library.

```
python3 -m pip install git+https://github.com/pytorch/fairseq
```

#### IMDB Reviews Dataset
```
mkdir datasets 
cd datasets
IMDB_DATASET='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
wget $IMDB_DATASET -O 
tar xvzf aclImdb_v1.tar.gz
``` 

#### Training

Launch a visdom instance for logging.

```
python3 -m pip install visdom # Install if not present.
python3 -m visdom.server &
```

Run the training script.

```
python3 -m mgan.main \
  --path datasets/aclImdb/train/ \
  --spm_path datasets/aclImdb/train/imdb.model
```
