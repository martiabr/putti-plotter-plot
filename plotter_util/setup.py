from setuptools import setup, find_packages

setup(
    name='plotter_util',
    version='0.1.0',
    packages=find_packages(include=['plotter_util']),
    install_requires=[
        'numpy',
    ]
)