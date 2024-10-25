# setup.py

from setuptools import setup, find_packages

setup(
    name='spikertools',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        # Add other dependencies as needed
    ],
    include_package_data=True,
    description='A Pythonlibrary for neuroscience data analysis of Backyard Brains SpikeRecorder Files',
    author='Greg Gage',
    author_email='gagegreg@backyardbrains.com',
    url='https://github.com/BackyardBrains/SpikerTools',
)
