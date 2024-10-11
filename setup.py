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
    "huggingface_hub>=0.21.4",
    'classifier-free-guidance-pytorch>=0.7.1',
    'einops>=0.8.0',
    'einx[torch]>=0.3.0',
    'ema-pytorch>=0.5.1',
    'environs',
    'gateloop-transformer>=0.2.2',
    'jaxtyping',
    'local-attention>=1.9.11',
    'numpy',
    'pytorch-custom-utils>=0.0.9',
    'rotary-embedding-torch>=0.6.4',
    'sentencepiece',
    'taylor-series-linear-attention==0.1.11',
    'torch>=2.1',
    'torch_geometric',
    'tqdm',
    'vector-quantize-pytorch>=1.18.1',
    'x-transformers>=1.30.19,<1.31',
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
