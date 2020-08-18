import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simpleTorch", # Replace with your own username
    version="0.0.3",
    author="Panos Petsagkourakis",
    author_email="p.petsagkourakis@ucl.ac.uk",
    description="This package is built on pytorch to avoid some standard steps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/panos108/Training_ANN_with_Pytorch/tree/master",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)