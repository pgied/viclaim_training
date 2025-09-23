from pathlib import Path
from setuptools import setup, find_packages
import os

here = Path(__file__).parent.resolve()

# Read install requirements from requirements.txt
requirements_file = here / "requirements.txt"
install_requires = (
    requirements_file.read_text(encoding="utf-8-sig")
    .splitlines()
    if requirements_file.exists()
    else []
)


setup(
    name="viclaim_training",
    version="0.1.0", 
    description="ViClaim Dataset Model Training",
    author="Patrick Giedemann",
    author_email="gied@zhaw.ch",
    python_requires=">=3.10",
    package_dir={"src": "src"},

    # === install the 'src' package itself plus any of its subpackages ===
    packages=["src"] + [f"src.{p}" for p in find_packages(where="src")],
    include_package_data=True, # include files from MANIFEST.in
    install_requires=install_requires, # deps from requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)