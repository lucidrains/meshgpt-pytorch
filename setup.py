from setuptools import setup, find_packages

exec(open('meshgpt_pytorch/version.py').read())

setup(
  name = 'meshgpt-pytorch',
  packages = find_packages(exclude=[]),
  version = __version__,
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
    'accelerate',
    'beartype',
    'classifier-free-guidance-pytorch>=0.4.2',
    'einops>=0.7.0',
    'ema-pytorch',
    'pytorch-warmup',
    'torch>=2.0',
    'torch_geometric',
    'torchtyping',
    'vector-quantize-pytorch>=1.12.0',
    'x-transformers>=1.26.0',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
