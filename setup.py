from setuptools import setup, find_packages

setup(
  name = 'galerkin_transformer',
  packages=find_packages(include=['galerkin_transformer', 'galerkin_transformer.*']),
  version = '0.2.1',
  license='MIT',
  description = 'Galerkin Transformer',
  long_description='Galerkin Transformer: a linear attention without softmax',
  long_description_content_type="text/markdown",
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
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Mathematics',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.8',
  ],
)