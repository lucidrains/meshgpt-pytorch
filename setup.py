from setuptools import setup, find_packages

setup(
  name = 'meshgpt-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'MeshGPT Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/meshgpt-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention mechanisms',
    'transformers',
    'mesh generation'
  ],
  install_requires=[
    'einops>=0.7.0',
    'torch>=2.0',
    'torch_geometric',
    'vector-quantize-pytorch>=1.11.7',
    'x-transformers>=1.26.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
