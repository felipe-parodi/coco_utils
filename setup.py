import setuptools
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "coco_utils" / "README.md").read_text()

setuptools.setup(
    name="coco_utils",
    version="0.1.0", 
    author="FP",
    description="Utilities for COCO dataset visualization, label manipulation, and file operations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where=".", include=['coco_utils', 'coco_utils.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.7', 
    install_requires=[
        "matplotlib",
        "Pillow", 
    ],

) 