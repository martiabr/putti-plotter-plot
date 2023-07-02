from setuptools import setup, find_packages

setup(
    name='plotter_shapes',
    version='0.1.0',
    packages=find_packages(include=['plotter_shapes']),
    install_requires=[
        'numpy',
    ]
)