import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from architectures import create_cnn
from training import train_model
from evaluation import evaluate_model
from visualization import plot_training_history

# Generate dummy data
import numpy as np

train_data = np.random.random((1000, 28, 28, 1))
train_labels = np.random.randint(10, size=(1000,))
test_data = np.random.random((200, 28, 28, 1))
test_labels = np.random.randint(10, size=(200,))

# Create and train model
model = create_cnn((28, 28, 1), num_classes=10)
history = train_model(model, train_data, train_labels, epochs=5)

# Evaluate model
results = evaluate_model(model, test_data, test_labels)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot training history
plot_training_history(history)
