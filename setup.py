import setuptools

exec(open("histoprep/_version.py").read())

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

requirements = []
with open("requirements.txt", "r") as f:
    for line in f.readlines():
        requirements.append(line.rstrip("\n"))


setuptools.setup(
    name="histoprep",
    version=__version__,
    author="jopo666",
    author_email="jopo@birdlover.com",
    description="Preprocessing module for large histological images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jopo666/HistoPrep",
    scripts=["HistoPrep"],
    packages=setuptools.find_packages(include=["histoprep", "histoprep.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="image-analysis preprocessing histology openslide pathology",
    python_requires=">=3.8",
    install_requires=requirements,
)
