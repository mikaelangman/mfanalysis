from setuptools import setup

setup(name='mfanalysis',
      version='0.12dev',
      description='Implements wavelet based multifractal analysis of 1d signals.',
      url='https://github.com/omardrwch/mfanalysis',
      author='Omar Darwiche Domingues',
      author_email='',
      license='MIT',
      packages=['mfanalysis'],
      install_requires=[
          'pywavelets', 'scipy', 'numpy', 'matplotlib',
      ],
      zip_safe=False)
