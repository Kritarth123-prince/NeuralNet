import unittest
import numpy as np
from src.architectures import create_cnn, create_gan
from src.evaluation import evaluate_model, evaluate_gan

class TestEvaluation(unittest.TestCase):

    def test_evaluate_model(self):
        model = create_cnn((28, 28, 1), 10)
        test_data = np.random.random((10, 28, 28, 1))
        test_labels = np.random.randint(10, size=(10,))
        results = evaluate_model(model, test_data, test_labels)
        self.assertEqual(len(results), 2)  # loss and accuracy

    def test_evaluate_gan(self):
        generator, discriminator, gan = create_gan((28, 28, 1), 100)
        images = evaluate_gan(generator, 100, num_samples=10)
        self.assertEqual(images.shape[0], 10)

if __name__ == '__main__':
    unittest.main()
