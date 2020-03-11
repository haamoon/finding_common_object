## Learning to Find Common Objects Across Few Image Collections

Code for the paper [Learning to Find Common Objects Across Few Image Collections](https://arxiv.org/abs/1904.12936). This is a reimplementation of the original code in TF2. The results might be slightly different from the paper due to the randomness.

If you use this code, please cite our paper:

```
@inproceedings{shaban19learning,
 author = {Shaban, Amirreza* and Rahimi, Amir* and Bansal, Shray and Gould, Stephen and Boots, Byron and Hartley, Richard},
 booktitle = {Proceedings of the International Conference on Computer Vision ({ICCV})},
 title = {Learning to Find Common Objects Across Few Image Collections},
 year = {2019}
}
```

## Installation

* This code has been tested on Ubuntu 16.04 with Python 3.5.2 and Tensorflow 2.0.0.
* Install [Tensorflow 2.0](https://www.tensorflow.org/install).
* Install [EasyDict](https://pypi.org/project/easydict) by running `pip install easydict`.

## How to perform evaluation

* We have placed pre-trained models and config files `experiments/mini/bs*` directories. The config files are used to evaluate the pre-trained models. The evaluation will be performed on the test classes of the mini-ImageNet dataset.
* Run `python eval.py --experiments_dir=path/to/evaluation_directory` to perform evaluation.  The `experiments_dir` argument should point to the directory  where the `config.json` file is located.

## How to train the network

* Unzip the [mini-ImageNet](https://gtvault-my.sharepoint.com/:u:/g/personal/ashaban6_gatech_edu/EYwztplXZflChBxyeszBqa0Br66SgmavA50MR7q0JW3Tww?e=k4arwA) training dataset in `data/` folder. A few number of `.pkl` files should be located at `data/miniimagenet_v2/` folder afterwards.
* We have placed  `config.json` files for miniImageNet experiments in `experiments/mini/k*` directories. You can copy and edit them for your desired task.
* Run `python train.py --experiments_dir=path/to/training_direcotry` to start the training process. The `experiments_dir` argument should point to the directory  where the `config.json` file is located.


