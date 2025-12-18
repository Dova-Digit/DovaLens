from pathlib import Path
from setuptools import setup, find_packages

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="dovalens",
    version="1.0.0",  # se vuoi, potrai passare a 1.0.1 e aggiornare __init__.py
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "dovalens=dovalens.cli:main",
        ],
    },
    license="MIT",
    license_files=["LICENSE"],  # <-- non 'LICENCE'
    long_description=long_description,
    long_description_content_type="text/markdown",
)
