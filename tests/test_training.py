import unittest
import numpy as np
from src.architectures import create_cnn, create_gan
from src.training import train_model, train_gan

class TestTraining(unittest.TestCase):

    def test_train_model(self):
        model = create_cnn((28, 28, 1), 10)
        train_data = np.random.random((100, 28, 28, 1))
        train_labels = np.random.randint(10, size=(100,))
        history = train_model(model, train_data, train_labels, epochs=1)
        self.assertIn('accuracy', history.history)
        self.assertIn('loss', history.history)

    def test_train_gan(self):
        generator, discriminator, gan = create_gan((28, 28, 1), 100)
        data = np.random.random((100, 28, 28, 1))
        train_gan(generator, discriminator, gan, data, 100, epochs=1, batch_size=4, save_interval=1)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
