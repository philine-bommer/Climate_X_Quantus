import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NoiseGrad_TF",
    version="0.1.1",
    author="Anna Hedstroem, Kirill Bykov",
    author_email="anna.hedstroem@tu-berlin.de",
    description="Package for NoiseGrad - enhancing network explanations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anna-hedstroem/NoiseGrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    include_package_data=True,
    package_data={'': ['src', 'srctf']}
)