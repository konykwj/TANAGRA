# -*- coding: utf-8 -*-
"""
Installation of EREGION, which contains a number of useful classes for your analysis projects.
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


print(setuptools.find_packages())

setuptools.setup(
    name="tanagra",
    version="1.0",
    author="Bill Konyk",
    author_email="faithbynumbers@gmail.com",
    description="Text and topic analysis package",
    packages=setuptools.find_packages(exclude=['example']),

    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)