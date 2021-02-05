import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="histoprep",
    version="0.0.4",
    author="jopo666",
    author_email="jopo@birdlover.com",
    description="Preprocessing module for large histological images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jopo666/HistoPrep",
    packages=setuptools.find_packages(),
    install_requires=[
        'opencv-python>=4.5.0',
        'openslide-python>=1.1.2',
        'pandas>=1.2.1',
        'Pillow>=7.0.0',
        'seaborn>=0.11.0',
        'pandas==0.23.3',
        'numpy>=1.19.0',
        'tqdm>=4.0'
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