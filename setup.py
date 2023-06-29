from setuptools import setup
from os import path

cur_dir = path.abspath(path.dirname(__file__))

with open(path.join(cur_dir, "requirements.txt"), "r") as f:
    requirements = f.read().split()

setup(
    name='test_problems',
    version='0.1',
    packages=['test_problems'],
    url='',
    license='',
    author='Ryan Roussel',
    author_email='rroussel@slac.stanford.edu',
    description='accelerator physics test problems for use with Xopt',
    install_requires=requirements,

)
