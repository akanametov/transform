import os
import re

from setuptools import setup, find_packages

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

setup(
    name='transform',
    packages=find_packages(),
    url='https://github.com/akanametov/transform',
    author='Azamat Kanametov',
    author_email='akkanametov@gmail.com',
    install_requires=['numpy', 'torch', 'pillow'],
    version='1.0',
    license='MIT',
    description='A simple augmentation for Object Detection',
    include_package_data=True)