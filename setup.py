from setuptools import setup, find_packages

setup(
  name = 'galerkin_transformer',
  packages = find_packages(exclude=['data']),
  version = '0.1.2',
  license='MIT',
  description = 'Galerkin Transformer',
  author = 'Shuhao Cao',
  author_email = 'scao.math@gmail.com',
  url = 'https://github.com/scaomath/galerkin-transformer',
  keywords = ['transformers', 'attention', 'galerkin', 'hilbert', 'pde'],
  install_requires=[
      'seaborn',
      'torchinfo',
      'numpy',
      'torch>=1.9.0',
      'plotly',
      'scipy',
      'psutil',
      'matplotlib',
      'tqdm',
      'PyYAML',
  ],
  classifiers=[
      'Development Status :: 1 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.8',
  ],
)