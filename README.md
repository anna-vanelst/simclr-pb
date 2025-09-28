# PAC-Bayesian Risk Certificates Contrastive Learning

This repository contains the official implementation of the experiments and figures in our paper.
You can find the preprint [here](https://arxiv.org/pdf/2412.03486?).

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{van2024tight,
  title={Tight pac-bayesian risk certificates for contrastive learning},
  author={Van Elst, Anna and Ghoshdastidar, Debarghya},
  journal={To appear in SIAM Journal on Mathematics of Data Science (SIMODS)},
  year={2024}
}
```

## Modules 

- `data.py` provides utilities for data augmentation, transformation, and loading for MNIST and CIFAR-10 datasets.

- `loss.py` provides implementations of various contrastive loss functions, such as ZeroOneLoss, SimplifiedContrastiveLoss, and ContrastiveLoss.

- `model.py` provides the implementation of convolutional neural networks (CNNs): it includes standard and probabilistic versions of the networks: (1) `CNNet7l` and `ProbCNNet7l` for CIFAR-10 (2) `CNNet3l` and `ProbCNNet3l` for MNIST.

- `model_utils.py` provides utilities for neural networks trained with PAC-Bayes by Backprop.

- `train.py` provides functions to train both standard and probabilistic neural networks.

- `evaluate.py` provides the function to evaluate the average contrastive loss over a dataset using a probabilistic neural network and a given loss function.

- `pb_obj.py` provides provides the implementation of the PAC-Bayes-based objective functions.

- `risk_certificates.py` provides the implementation of various risk certificate computations for 
contrastive learning models, using bounds like Catoni, kl, classic, McDiarmid-McAllester, and kl-epsilon-modified.

- `linear_classifier.py` provides a linear classifier for feature representations learned by a SimCLR model.

- `transfer_bound.py` provides classes and methods to compute an upper-bound on the linear classifier loss based on contrastive loss. 

- `run.py` manages experiments with the `ExperimentRunner` class, including methods for training prior and posterior models.


- `run.ipynb` is a notebook used to set the experiment settings and run experiments. 
