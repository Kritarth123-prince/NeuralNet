import unittest
from src.architectures import create_cnn, create_rnn, create_gan

class TestArchitectures(unittest.TestCase):

    def test_create_cnn(self):
        model = create_cnn((28, 28, 1), 10)
        self.assertEqual(model.input_shape[1:], (28, 28, 1))
        self.assertEqual(model.output_shape[1], 10)

    def test_create_rnn(self):
        model = create_rnn(100, 10)
        self.assertEqual(model.input_shape[1], 100)
        self.assertEqual(model.output_shape[1], 10)

    def test_create_gan(self):
        generator, discriminator, gan = create_gan((28, 28, 1), 100)
        self.assertEqual(generator.input_shape[1:], (100,))
        self.assertEqual(discriminator.input_shape[1:], (28, 28, 1))
        self.assertEqual(gan.input_shape[1:], (100,))

if __name__ == '__main__':
    unittest.main()
