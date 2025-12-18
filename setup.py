from setuptools import setup, find_packages

setup(
    name="dovalens",
    version="1.0.0",
    description="Automated Dataset Profiler & Report Generator",
    author="Pietro Ferreri",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy"
    ],
    entry_points={
        "console_scripts": [
            "dovalens=dovalens.cli:main",
        ],
    },
    include_package_data=True,
)
