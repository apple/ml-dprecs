from setuptools import setup, find_packages

setup(
    name="dprecs",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    install_requires=[
        "pyspark==2.4.2",
        "matplotlib==3.3.4",
        "pandas==1.2.1",
        "numpy==1.20",
        "scikit-learn==1.0.2",
        "tensorflow==2.11.1",
        "deepctr==0.9.3",
    ],
    extras_require={
        "dev": [
            "parameterized>=0.8.1",
        ],
        },
    python_version="3.7.9",
)
