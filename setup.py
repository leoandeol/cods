#!/usr/bin/python3

import setuptools

setuptools.setup(
    name="cods",
    version="0.1",
    description="Conformal Object Detection & Segmentation",
    author="Léo Andéol, Luca Mossina",
    author_email="leo@andeol.eu",
    url="https://github.com/leoandeol/cods/",
    packages=setuptools.find_namespace_packages(include=["cods.*"]),
)
