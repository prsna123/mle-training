from setuptools import find_packages, setup

setup(
    name="ml_package",
    version="0.1.0",
    description="Just to learn about packaging python and DS ML",
    author="Prasanna",
    author_email="prasanna.babu@tigeranalytics.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests",
    ],
    python_requires=">=3.6",
)
