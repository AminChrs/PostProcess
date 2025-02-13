import numpy as np
from helpers.bootstrap import bootstrap_vec
from helpers.embedding import embedding, classifier
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
    emb_loss = embedding("loss", "rf")
    emb_eo = embedding("eo", "rf")
    emb_loss.fit(Dataset)
    emb_eo.fit(Dataset)

    # Validation
    embs = [emb_loss, emb_eo]
    classifier_emb = classifier(embs)
    coeffs = classifier_emb.optimal_combination(embs, Dataset,
                                                coeff_space,
                                                tolerance_space,)

    # Test
    means = []
    stds = []
    for threshold in coeffs:
        if threshold is None:
            means.append(None)
            stds.append(None)
            continue
        else:
            classifier_emb.predict(threshold, Dataset, 'test', 'estimate')
            out_test = classifier_emb.mean_emb_predict(Dataset, 'test', 'true',
                                                       mean=False)
            out_test_mean, out_test_std = bootstrap_vec(out_test)
            means.append(out_test_mean)
            stds.append(out_test_std)
    return tolerance_space, means, stds
