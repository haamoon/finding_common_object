# Learning to Find Common Objects

Code for the Paper [Learning to Find Common Objects Across Image Collections](https://arxiv.org/abs/1904.12936).

If you use this code, please cite our paper:

```
@misc{shaban2019learning,
    title={Learning to Find Common Objects Across Image Collections},
    author={Amirreza Shaban and Amir Rahimi and Shray Bansal and Stephen Gould and Byron Boots and Richard Hartley},
    year={2019},
    eprint={1904.12936},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Installation

* This code has been tested on Ubuntu 16.04 with Python 3.5.2 and Tensorflow 2.0.0-alpha0.
* Install [Tensorflow 2.0](https://www.tensorflow.org/install) by running `pip install tensorflow-gpu==2.0.0-alpha0`.
* Install [EasyDict](https://pypi.org/project/easydict) by running `pip install easydict`.

## How to train the network

* Unzip the mini-ImageNet dataset in `data/` folder. A few number of `.pkl` files should be located at `data/miniimagenet_v2/` folder afterwards. A downloadable link will be provided.
* We have placed a sample `config.json` file in `experiments/mini/k2n2/config.json`. This config is used to train pairwise and unary relation modules when we have two positive bags of five images each and ten negative images. You can edit or copy it for your desired task.
* Run `python train.py --experiments_dir=experiments/mini/k2n2` to start the training process. The `experiments_dir` argument should point to a folder where the `config.json` file is located.

## How to perform evaluation

* We have placed a sample `config.json` file in `experiments/mini/k8n2/config.json`. This config is used to evaluate the last checkpoint of the mentioned trained network when we have eight positive bags of five images each and ten negative images. The evaluation will be performed on the test classes of the mini-ImageNet dataset.
* Run `python eval.py --experiments_dir=experiments/mini/k8n2` to perform evaluation.
