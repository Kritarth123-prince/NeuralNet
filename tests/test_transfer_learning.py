import unittest
from src.transfer_learning import use_pretrained_model, fine_tune_model

class TestTransferLearning(unittest.TestCase):

    def test_use_pretrained_model(self):
        model = use_pretrained_model('VGG16', (224, 224, 3), 10)
        self.assertEqual(model.input_shape[1:], (224, 224, 3))
        self.assertEqual(model.output_shape[1], 10)

    def test_fine_tune_model(self):
        model = use_pretrained_model('VGG16', (224, 224, 3), 10)
        fine_tuned_model = fine_tune_model(model, 'VGG16', fine_tune_at=100)
        self.assertEqual(fine_tuned_model.input_shape[1:], (224, 224, 3))
        self.assertEqual(fine_tuned_model.output_shape[1], 10)

if __name__ == '__main__':
    unittest.main()
