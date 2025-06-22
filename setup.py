
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="cellniche",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="Graph contrastive learning for spatial transcriptomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Super-LzzZ/CellNiche",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.12.1",
        "torch-geometric>=2.6.1",
        "torch-scatter>=2.1.0",
        "torch-sparse>=0.6.15",
        "torch-cluster>=1.6.0",
        "torch-spline-conv>=1.2.1",
        "scanpy>=1.9.8",
        "anndata>=0.9.2",
        "scikit-learn>=1.3.2",
        "numpy>=1.22.4",
        "scipy>=1.10.1",
        "pandas>=2.0.3",
        "tqdm>=4.67.1",
        "networkx>=3.1",
        "node2vec>=0.5.0",
        "h5py>=3.11.0",
        "natsort>=8.4.0",
    ],
    # entry_points={
    #     "console_scripts": [
    #         "cellniche-run = cellniche.main:main",
    #     ],
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)