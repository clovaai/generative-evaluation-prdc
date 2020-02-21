import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prdc",
    version="0.1",
    author="NAVER Corp.",
    description="Compute precision, recall, density, and coverage metrics "
                "for two sets of vectors.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clovaai/prdc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy',
        'joblib'
    ],
)
