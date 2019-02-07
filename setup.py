import pathlib
from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="easydatamining-easydm",
    version="1.0.0",
    description="Read the latest Easydataminig tutorials",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/easydataminig/easydm",
    author="SL",
    author_email="",
    license="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=[],
    entry_points={
        "console_scripts": [
            "easydataminig=easydm.__main__:main",
        ]
    },
)