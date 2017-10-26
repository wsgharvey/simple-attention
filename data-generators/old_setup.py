from setuptools import setup, find_packages
import sys

package_dir = {2: 'python2', 3: '.'}[sys.version_info.major]

setup(name='DataGenerators',
      version='1.0',
      description='Helpful Data Generators',
      author='William Harvey',
      packages=find_packages(package_dir, exclude=['contrib', 'docs', 'tests*']),
      package_dir={'': package_dir},
      package_data={'project': ['default_data.json', 'other_datas/default/*.json']}
      entry_points={
          'modified_mnist': [
              'quadrant_mnist_e = modified_mnist.quadrant_mnist.simple_quadrant.quadrant_mnist'
              ],
          },
      )
