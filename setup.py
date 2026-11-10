from setuptools import setup, find_packages

setup(
    name="astro-pipeline-ml",
    version="0.1.0",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.4",
        "torch>=2.2",
        "optuna>=3.6",
        "pyyaml>=6.0",
    ],
)
