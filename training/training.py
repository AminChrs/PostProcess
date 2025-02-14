import numpy as np
from helpers.embedding import Embedding, Classifier
from datasetsdefer.acs_dataset import generate_ACS


def main():

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
    return tolerance_space, means, stds
