def evaluate_model(model, test_data, test_labels):
    results = model.evaluate(test_data, test_labels)
    return results

def evaluate_gan(generator, noise_dim, num_samples=100):
    import numpy as np

    noise = np.random.normal(0, 1, (num_samples, noise_dim))
    generated_images = generator.predict(noise)
    return generated_images
