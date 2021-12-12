from setuptools import setup, find_packages

setup(
    name='iso',
    version='0.1.0',
    packages=find_packages(include=['iso.*']),
    install_requires=[
        'numpy',
    ]
)