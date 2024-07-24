import unittest
import numpy as np
from src.visualization import plot_training_history, visualize_images

class TestVisualization(unittest.TestCase):

    def test_plot_training_history(self):
        history = {'accuracy': [0.1, 0.2, 0.3], 'loss': [2.0, 1.5, 1.0]}
        plot_training_history(type('obj', (object,), {'history': history}))
        self.assertTrue(True)  # Check if no exceptions raised

    def test_visualize_images(self):
        images = np.random.random((5, 28, 28))
        visualize_images(images)
        self.assertTrue(True)  # Check if no exceptions raised

if __name__ == '__main__':
    unittest.main()
