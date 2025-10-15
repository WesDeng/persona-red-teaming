#!/usr/bin/env python

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

# -*- coding: utf-8 -*-

"""Setup script for RainbowPlus."""

import os

from setuptools import find_packages
from setuptools import setup

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Define package metadata
NAME = "rainbowplus"
VERSION = "0.1.0"
DESCRIPTION = "RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search"
AUTHOR = "Knovel Engineering Lab"
AUTHOR_EMAIL = "andrew.dang@knoveleng.com"
URL = "https://github.com/knoveleng/rainbowplus"
LICENSE = "MIT"

# Define package dependencies
REQUIRED_PACKAGES = [
    "vllm>=0.6.3",
    "torch>=2.6.0",
    "openai>=1.0.0",
    "nltk>=3.8.1",
    "pydantic>=2.0.0",
    "PyYAML>=6.0",
    "datasets>=2.14.0",
    "tenacity>=8.0.0",
    "httpx>=0.24.0",
    "sentence-transformers>=2.2.0",
    "scikit-learn>=1.0.0",
    "nltk>=3.8.1",
    "sentence-transformers>=2.2.0",
    "transformers<4.54.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "lexicalrichness>=0.3.0",
    "wordcloud>=1.8.2",
]

# Define development dependencies
DEV_PACKAGES = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        "dev": DEV_PACKAGES,
    },
    python_requires=">=3.8",
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
    keywords="llm, language-model, evaluation, robustness, ai-safety",
    project_urls={
        "Bug Reports": f"{URL}/issues",
        "Source": URL,
    },
)
