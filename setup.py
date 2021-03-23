"""Setup script."""

import os

from setuptools import find_packages
from setuptools import setup

version = '0.0.1'


def _get_requirements():
  """Parses requirements.txt file."""
  install_requires_tmp = []
  dependency_links_tmp = []
  with open(
      os.path.join(os.path.dirname(__file__), './requirements.txt'), 'r') as f:
    for line in f:
      package_name = line.strip()
      # Skip empty line or comments starting with "#".
      if not package_name or package_name[0] == '#':
        continue
      if package_name.startswith('-e '):
        dependency_links_tmp.append(package_name[3:].strip())
      else:
        install_requires_tmp.append(package_name)
  return install_requires_tmp, dependency_links_tmp

install_requires, dependency_links = _get_requirements()

install_requires.append('tf-nightly')
install_requires.append('tensorflow-datasets')

setup(
    name='keras-cv',
    version=version,
    description='Keras Computer Vision Library',
    url='https://github.com/keras-team/keras-cv',
    author='The Keras authors',
    author_email='keras-team@google.com',
    license='Apache License 2.0',
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    packages=find_packages(exclude=('tests',)),
    exclude_package_data={'': ['*_test.py',],},
    dependency_links=dependency_links,
    python_requires='>=3.6',
)
