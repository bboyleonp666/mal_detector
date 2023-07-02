# Malware detector
This is the implementation of [GEMAL](https://link.springer.com/article/10.1007/s00521-021-06808-8).

## Reverse
For reverse a binary and extract function call graphs(FCGs), please refer [frtools](https://github.com/bboyleonp666/frtools).

## Train Word2Vec model
To train a word2vec model by yourself, check `train_word2vec.py`, which you'll have to reverse the binaries according to your needs. And we have already provided our pre-trained one in `model_saved/`

## Train Detector
To train your own detector, the whole training progress is in `implement_Wu.ipynb`, follow the steps inside, training a new one would not be a big deal.

## Limitation
As the original work is only implemented on x86 family, we follow its setting. And since our reverse tool is different from the original work, it might vary a little compare to that work.