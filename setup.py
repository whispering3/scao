from setuptools import setup, find_packages

setup(
    name="scao",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "mypy>=1.5",
            "ruff>=0.1",
            "build",
            "twine",
        ],
        "cuda": [
            "torch>=2.0.0",
        ],
        "hf": [
            "transformers>=4.30.0",
            "datasets>=2.0.0",
        ],
        "all": [
            "transformers>=4.30.0",
            "datasets>=2.0.0",
            "mypy>=1.5",
            "ruff>=0.1",
        ],
    },
    author="SCAO Authors",
    description="Sparse Curvature-Aware Adaptive Optimizer — second-order training at near-AdamW cost",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/whispering3/scao",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
)
