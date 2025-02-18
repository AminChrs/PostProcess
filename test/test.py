# Here I make all the unit tests I need in dgnp

import unittest
from datasetsdefer.acs_dataset import generate_ACS
from datasetsdefer.broward import generate_COMPAS
from datasetsdefer.hatespeech import generate_hatespeech
from helpers.embedding import Embedding, Classifier
import numpy as np


# class Test_ACS(unittest.TestCase):

    # def test_generate(self):
    #     Dataset = generate_ACS()
    #     self.assertIsNotNone(Dataset)

    # def test_embedding(self):
    #     loss = embedding("loss", "rf", system="def")
    #     eo = embedding("eo", "rf", system="def")
    #     assert loss is not None
    #     assert eo is not None

    # def test_fit_embedding(self):
    #     Dataset = generate_ACS()
    #     loss = embedding("loss", "rf", system="def")
    #     eo = embedding("eo", "rf", system="def")
    #     loss.fit(Dataset=Dataset)
    #     eo.fit(Dataset=Dataset)

    # def test_validation_embedding(self):

    #     tolerance_space = np.linspace(0.01, 0.2, 10)

    #     # Define Coefficient Space
    #     coeff_space = np.linspace(-.5, .5, 10)  # Eodds + ACS
    #     coeff_space = np.meshgrid(coeff_space, coeff_space)
    #     coeff_space = list(zip(coeff_space[0].flatten(),
    #                            coeff_space[1].flatten()))
    #     Dataset = generate_ACS()
    #     loss = Embedding("loss", "rf", system="def")
    #     eo = Embedding("eo", "rf", system="def")
    #     loss.fit(Dataset=Dataset)
    #     eo.fit(Dataset=Dataset)
    #     embs = [loss, eo]
    #     classifier_emb = Classifier(embs)
    #     coeffs = classifier_emb.optimal_combination(Dataset,
    #                                                 coeff_space,
    #                                                 tolerance_space,)
    #     self.addCleanup(lambda: print(coeffs))
    #     self.assertIsNotNone(coeffs)

    # def test_test_embedding(self):

    #     tolerance_space = np.linspace(0.01, 0.2, 10)

    #     coeff_space = np.linspace(-.5, .5, 10)  # Eodds + ACS
    #     coeff_space = np.meshgrid(coeff_space, coeff_space)
    #     coeff_space = list(zip(coeff_space[0].flatten(),
    #                            coeff_space[1].flatten()))
    #     Dataset = generate_ACS()
    #     loss = Embedding("loss", "rf", system="def")
    #     eo = Embedding("eo", "rf", system="def")
    #     loss.fit(Dataset=Dataset)
    #     eo.fit(Dataset=Dataset)
    #     embs = [loss, eo]
    #     classifier_emb = Classifier(embs)
    #     coeffs = classifier_emb.optimal_combination(Dataset,
    #                                                 coeff_space,
    #                                                 tolerance_space,)
    #     means, stds = classifier_emb.test(coeffs, Dataset)
    #     self.addCleanup(lambda: print(means, stds))
    #     self.assertIsNotNone(means)
    #     self.assertIsNotNone(stds)


# class test_Broward(unittest.TestCase):

    # def test_generate(self):
    #     Dataset = generate_COMPAS()
    #     self.assertIsNotNone(Dataset)

    # def test_embedding(self):
    #     Dataset = generate_COMPAS()
    #     loss = Embedding("loss", "nn", system="def", Dataset=Dataset)
    #     eo = Embedding("eo", "nn", system="def", Dataset=Dataset)
    #     assert loss is not None
    #     assert eo is not None

    # def test_fit_embedding(self):
    #     Dataset = generate_COMPAS()
    #     loss = Embedding("loss", "nn", system="def", Dataset=Dataset)
    #     eo = Embedding("eo", "nn", system="def", Dataset=Dataset)
    #     loss.fit(Dataset=Dataset)
    #     eo.fit(Dataset=Dataset)

    # def test_validation_embedding(self):

    #     tolerance_space = np.linspace(0.01, 0.2, 10)

    #     # Define Coefficient Space
    #     coeff_space = np.linspace(-.5, .5, 10)
    #     coeff_space = np.meshgrid(coeff_space, coeff_space)
    #     coeff_space = list(zip(coeff_space[0].flatten(),
    #                            coeff_space[1].flatten()))
    #     Dataset = generate_COMPAS()
    #     loss = Embedding("loss", "nn", system="def", Dataset=Dataset)
    #     eo = Embedding("eo", "nn", system="def", Dataset=Dataset)
    #     loss.fit(Dataset=Dataset)
    #     eo.fit(Dataset=Dataset)
    #     embs = [loss, eo]
    #     classifier_emb = Classifier(embs)
    #     coeffs = classifier_emb.optimal_combination(Dataset,
    #                                                 coeff_space,
    #                                                 tolerance_space,)
    #     self.addCleanup(lambda: print(coeffs))
    #     self.assertIsNotNone(coeffs)

    # def test_test_embedding(self):

    #     tolerance_space = np.linspace(0.01, 0.2, 10)

    #     coeff_space = np.linspace(-.5, .5, 10)
    #     coeff_space = np.meshgrid(coeff_space, coeff_space)
    #     coeff_space = list(zip(coeff_space[0].flatten(),
    #                            coeff_space[1].flatten()))
    #     Dataset = generate_COMPAS()
    #     loss = Embedding("loss", "nn", system="def", Dataset=Dataset)
    #     eo = Embedding("eodds", "nn", system="def", Dataset=Dataset)
    #     loss.fit(Dataset=Dataset)
    #     eo.fit(Dataset=Dataset)
    #     embs = [loss, eo]
    #     classifier_emb = Classifier(embs)
    #     coeffs = classifier_emb.optimal_combination(Dataset,
    #                                                 coeff_space,
    #                                                 tolerance_space,)
    #     means, stds = classifier_emb.test(coeffs, Dataset)
    #     self.addCleanup(lambda: print(means, stds))
    #     self.assertIsNotNone(means)
    #     self.assertIsNotNone(stds)


class test_hatespeech(unittest.TestCase):

    # def test_generate(self):
    #     Dataset = generate_hatespeech()
    #     self.assertIsNotNone(Dataset)

    def test_embedding(self):
        Dataset = generate_hatespeech()
        loss = Embedding("loss", "nn", system="def", Dataset=Dataset)
        eo = Embedding("eo", "nn", system="def", Dataset=Dataset)
        assert loss is not None
        assert eo is not None

def main():
    unittest.main(buffer=False)


if __name__ == '__main__':
    unittest.main()
