# AE-DNN
Code for reproducing some key results of our ICPR 2020 paper *"Separation of Aleatoric and Epistemic Uncertainty in Deterministic Deep Neural Networks"*
![](plots/circles.png)

# Table of contents
1. [Introduction](#introduction)
2. [Uncertainty Histograms](#unc_hist)
    1. [Normalized](#normalized)
    1. [Logarithmic](#logarithmic)
3. [Reproduce Results](#reproduce)
    1. [Requirements](#requirements)
    2. [Quantitative Experiment](#quantitative)
    3. [Synthetic Experiment](#synthetic)


## Uncertainty Histograms<a name="unc_hist"></a>
In the following plots, we can find further uncertainty histograms. In most cases, such histograms are
representative only if we also evaluate corresponding in-distribution uncertainties.  We show normalized
uncertainty values, as well as values in logarithmic space.  This makes sense, as one may not recognize tiny
differences in a histogram that can be used to determine a threshold for separation.

### Normalized <a name="normalized"></a>

**MNIST vs. NotMNIST**

![](plots/UC_mnist.png)

**SVHN vs. CIFAR10**

![](plots/UC_svhn.png)

**CIFAR5 vs. CIFAR5**

![](plots/UC_cifar5.png)

### Logarithmic <a name="logarithmic"></a>
**MNIST vs. NotMNIST**

![](plots/UC_mnist_log.png)

**SVHN vs. CIFAR10**

![](plots/UC_svhn_log.png)

**CIFAR5 vs. CIFAR5**

![](plots/UC_cifar5_log.png)

## Reproduce Results<a name="reproduce"></a>
### Requirements <a name="requirements"></a>
All requirements can be installed with:
```
pip install -r requirements.txt
```
For cuda support, please refer to https://pytorch.org/.

Jupyter Notebooks can be opened with the bash command `jupyter notebook`.

### Experiment <a name="quantitative"></a>

**Running Experiments:**

All experimental configurations can be found in the [config file](config.yml).
For hyperparameter optimization, we applied Bayesian optimization. The results are published in the paper and
can be reproduced with the following command:
```bash
python experiment.py \
    --n_epochs 50  \
    --n_reps 5 \
    --lmb $lmb \
    --gamma $gamma \
    --dropout_rate $dropout_rate \
    --ood_factor $ood_factor \
    --lr $lr \
    --weight_decay $weight_decay \
    --method_name $method_name \
    --dataset $dataset \
    --ood_ds $ood_ds
```

**Evaluation of Experiments:**

The evaluation of experiments can be obtained in this [Jupyter Notebook](Experiments-Quantitative.ipynb).

### Synthetic <a name="synthetic"></a>
The synthetic experiments can be found in the following Jupyter Notebooks: 
- [Circle Example](Example-Circles.ipynb)
- [Gaussian Distribution Example](Experiments-Quantitative.ipynb)
