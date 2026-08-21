from setuptools import setup, find_packages

setup(
    name="feta-aml",
    version="1.0.0",
    description="FeTA / EP-FedProto: Federated Graph Transfer Attention for Privacy-Preserving Anti-Money Laundering",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "tqdm>=4.65",
        "pyyaml>=6.0",
    ],
)
