#building a project as a package
from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path: str) -> List[str]: 
    """
    This function will return the list of requirements..
    """
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip() != '-e .']
    return requirements





setup(
    name = "mlproject",
    version = "0.0.1",
    author = "Nagarjun",
    author_email = "nagarjunramakrishnan10@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("./requirements.txt")

)