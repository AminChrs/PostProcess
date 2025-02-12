from setuptools import setup, find_packages

setup(
    name="postprocessing",  # Replace with your package name
    version="0.1.0",
    author="Amin Charusaie",
    author_email="amin.ch90@gmail.com",
    description="Multi-Objective Learning via Ensembling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AminChrs/PostProces",
    packages=["postprocessing"],  # Replace with your package name
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=open("requirements.txt").read().split("\n"),
)