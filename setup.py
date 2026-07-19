from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deep-spec",
    version="0.1.0",
    author="Zaryab Rahman",
    author_email="zaryabrahman848@gmail.com",
    description="Unentangling Vision-Language Models via Spectral Bipartite Graph Partitioning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YourUsername/Deep-Spec",
    packages=find_packages(exclude=["tests*", "experiments*"]),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)