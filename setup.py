from setuptools import setup, find_packages

setup(
    name='NeuralNet',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'tensorflow',  # or 'torch' if using PyTorch
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'neuralnet=src.main:main',
        ],
    },
)
