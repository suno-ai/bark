"""A setuptools based setup module.
See: <https://packaging.python.org/en/latest/distributing.html>
"""
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bark_pkg",
    version="0.1",
    author_email="apetree1001@email.phoenix.edu",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
