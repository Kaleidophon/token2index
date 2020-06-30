from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="token2index",  # Replace with your own username
    version="0.9.2",
    author="Dennis Ulmer",
    description="A lightweight but powerful library for token indexing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kaleidophon/token2index",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="GPL",
    python_requires=">=3.5.3",
    keywords=[
        "indexing",
        "token",
        "nlp",
        "pytorch",
        "tensorflow",
        "numpy",
        "w2i",
        "t2i",
        "stoi",
        "itos",
        "i2t",
        "i2w",
        "deep learning",
    ],
    packages=find_packages(exclude=["docs", "dist"]),
)
