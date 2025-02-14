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

In the following, we brought a simple example of training and validating d-GNP for ACS dataset:

```python
    # Generate Dataset
    Dataset = generate_ACS()

    # Define Tolerance Space
    tolerance_space = np.linspace(0.01, 0.2, 1000)

    # Define Coefficient Space
    coeff_space = np.linspace(-.5, .5, 100)  # Eodds + ACS
    coeff_space = np.meshgrid(coeff_space, coeff_space)
    coeff_space = list(zip(coeff_space[0].flatten(),
                           coeff_space[1].flatten()))

    # Training
    emb_loss = Embedding("loss", "rf")
    emb_eo = Embedding("eo", "rf")
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
The following datasets are used for the experiments:

- The ACSIncome dataset from [Folktables](https://github.com/socialfoundations/folktables) package for income prediction
- The [COMPAS](https://www.science.org/doi/10.1126/sciadv.aao5580) dataset for prediction of recidivism
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