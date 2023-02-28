import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CphXAI",
    version="0.1.1",
    author="Philine Bommer, Zack Labe",
    author_email="philine.l.bommer@tu-berlin.de",
    description="Package for the analysis of climate change problems using XAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/philine-bommer/philine-bommer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    include_package_data=True,
    package_data={'': ['src', 'src/utils','src/read']}
)