![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![GitHub version](https://img.shields.io/github/v/release/AminChrs/PostProcess)

<p align="center">
<img src="logo.png" width="400" />
</p>

This repository contains the experiments of the paper "[A Unifying Post-Processing Framework for Multi-Objective Learn-to-Defer Problems](https://arxiv.org/abs/2407.12710)" that was [published](https://neurips.cc/virtual/2024/poster/95484) in NeurIPS 2024. This paper introduces a post-processing method for solving a variety of multi-objective learning problems, including but not limited to learn-to-defer problem and fair multi-class classification problem. During this method, first an embedding function related to each objective is trained, and then an optimal classifier via linear combination of the embedding functions is obtained using validation data. This method presents an alternative paradigm compared to Lagrangian-based methods such as primal-dual methods.

## Installation

For easy installation of the package using ```pip```, you can use the following command in your terminal:
```bash
pip install postprocessing
```

## Quickstart Example
![image info](Diagram.jpg)
The flow of the codes written via d-GNP is as above figure and contains two main step of training the embedding functions related to each constraint/loss, and then to find the right combination of the embeddings to achieve optimal accuracy with a set tolerance for the constraints.

In the following, we brought a simple example of training and validating d-GNP for [ACS](https://github.com/socialfoundations/folktables) dataset:

```python
import numpy as np
import postprocessing as dgnp
from dgnp.helpers.embedding import Embedding, Classifier
from dgnp.datasetsdefer.acs_dataset import generate_ACS

# Generate Dataset
Dataset = generate_ACS()

# Define Tolerance Space
tolerance_space = np.linspace(0.01, 0.2, 1000)

# Define Coefficient Space for Linear Combination
coeff_space = np.linspace(-.5, .5, 100)
coeff_space = np.meshgrid(coeff_space, coeff_space)
coeff_space = list(zip(coeff_space[0].flatten(),
                        coeff_space[1].flatten()))

# Training
emb_loss = Embedding("loss", "rf", system="def")
emb_eo = Embedding("eo", "rf", system="def")
emb_loss.fit(Dataset)
emb_eo.fit(Dataset)

# Validation
embs = [emb_loss, emb_eo]
classifier_emb = Classifier(embs)
coeffs = classifier_emb.optimal_combination(Dataset,
                                            coeff_space,
                                            tolerance_space,)

# Test
means, stds = classifier_emb.test(coeffs, Dataset)
```

## Datasets
The following datasets can be generated and used in this package:

| Dataset | Generation Code |
|---------|-----------------| 
| The ACSIncome dataset from [Folktables](https://github.com/socialfoundations/folktables) package for income prediction |  ```generate_ACS()```
| The [COMPAS](https://www.science.org/doi/10.1126/sciadv.aao5580) dataset for prediction of recidivism | ```generate_COMPAS()``` |
## Embeddings

This package includes predefined embeddings for objectives and constraints. You can create an embedding using ```Embedding(identifier, "rf", , **kwargs)``` for Random Forest-based estimation or ```Embedding(identifier, "nn", **kwargs)``` for Neural Network-based estimation.

The available identifiers are listed below.

| Embedding | identifier | kwargs |
|-----------|------|-----|
| Deferral loss| ```"loss"``` | N/A |
| Multiclass classification loss | ```"loss_multi"```| (optional) Cost-sensitive matrix ```C=```$C$
| Expert intervention budget | ```"interv_budget"```| N/A |
| OOD | ```"ood"```| N/A |
| Long-tail classification | ```"long_tail"```| ```alpha=```[$\alpha_1, \ldots, \alpha_K$] |
| Type-K error | ```"type_K_err"```| ```system="def"```/```system="multi"```, ```K=```$$K$$ |
| Demographic parity | ```"dp"```| ```system="def"```/```system="multi"```, (optional) Effective label ```L=```$L$  | 
| Equality of opportunity | ```"eop"```| ```system="def"```/```system="multi"```, (optional) Effective label ```L=```$L$   |
| Equalized odds | ```"eodds"``` | ```system="def"```/```system="multi"```, (optional) Effective label ```L=```$L$   |

<!-- ## Requirements

To run the code in the Jupyter Notebook files, make sure you have the dependencies installed. To do this, you can run the following command in your terminal:

```sh
pip install -r requirements.txt
``` -->

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

In case that you have used codes in this repository, please consider citing our paper:

```bibtex
@inproceedings{charusaieunifying,
  title={A Unifying Post-Processing Framework for Multi-Objective Learn-to-Defer Problems},
  author={Charusaie, Mohammad-Amin and Samadi, Samira},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```