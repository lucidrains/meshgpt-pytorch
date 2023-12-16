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
    'accelerate>=0.25.0',
    'beartype',
    'classifier-free-guidance-pytorch>=0.4.2',
    'einops>=0.7.0',
    'ema-pytorch',
    'local-attention>=1.9.0',
    'gateloop-transformer>=0.1.5',
    'pytorch-warmup',
    'pytorch-custom-utils',
    'torch>=2.0',
    'torch_geometric',
    'torchtyping',
    'tqdm',
    'vector-quantize-pytorch>=1.12.0',
    'x-transformers>=1.26.0',
    'tqdm',
    'trimesh',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
