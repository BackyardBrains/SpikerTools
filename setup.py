# setup.py

from setuptools import setup, find_packages

setup(
    name='spikertools',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        # Add other dependencies as needed
    ],
    extras_require={'nwb': ['pynwb>=4.2.0,<5']},
    include_package_data=True,
    description='A Python library for neuroscience data analysis of Backyard Brains SpikeRecorder Files',
    author='Greg Gage',
    author_email='gagegreg@backyardbrains.com',
    url='https://github.com/BackyardBrains/SpikerTools',
)
