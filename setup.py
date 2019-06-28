
import os
from setuptools import setup, find_packages

# Utility method to read the README.rst file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

import version
VERSION = version.__version__

CLASSIFIERS = [
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering'
]

setup(
    name='ccg_nlpy',
    version=VERSION,
    description=("?????"),
    long_description=read('README.md'),
    url='https://github.com/????',
    author='Cognitive Computation Group',
    author_email='danielkh@cis.upenn.edu',
    license='Research and Academic Use License',
    keywords="NLP, natural language processing",
    packages=find_packages(exclude=['tests.*', 'tests']),
    install_requires=['configparser', 'pulp'],
    # package_data={'ccg_nlpy': ['config/*.cfg']},
    classifiers=CLASSIFIERS,
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'mock'],
    zip_safe=False)