from setuptools import find_namespace_packages, find_packages, setup

# read the contents of your requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="mojo_kirby",
    packages=find_packages() + find_namespace_packages(include=["hydra_plugins.*"]),
    install_requires=requirements,
    # For configurations to be discoverable at runtime, they should also be added to the search path.
    include_package_data=True,
)
