import setuptools

exec(open('histoprep/_version.py').read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="histoprep",
    version=__version__,
    author="jopo666",
    author_email="jopo@birdlover.com",
    description="Preprocessing module for large histological images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jopo666/HistoPrep",
    packages=setuptools.find_packages(include=['histoprep','histoprep.*']),
    install_requires=[
        'opencv-python>=4.2.0',
        'openslide-python>=1.1.2',
        'pandas>=1.2.1',
        'Pillow>=7.0.0',
        'seaborn>=0.11.0',
        'numpy>=1.19.0',
        'tqdm>=4.0',
        'aicspylibczi>=2.8.0',
        'shapely>=1.7.0',
        'scikit-learn>0.24',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords='image-analysis preprocessing histology openslide',
    python_requires='>=3.8',
)