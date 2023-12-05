<img src="./meshgpt.png" width="450px"></img>

## MeshGPT - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2311.15475">MeshGPT</a>, SOTA Mesh generation using Attention, in Pytorch

Will also add text conditioning, for eventual text-to-3d asset

## Install

```bash
$ pip install meshgpt-pytorch
```

## Usage

```python
import torch
from meshgpt_pytorch import MeshAutoencoder

# autoencoder

autoencoder = MeshAutoencoder(
    dim = 512,
    encoder_depth = 6,
    decoder_depth = 6,
    num_discrete_coors = 128
)

# mock inputs

vertices = torch.randn((2, 121, 3))
faces = torch.randint(0, 121, (2, 64, 3))
face_edges = torch.randint(0, 64, (2, 2, 96))

# forward in the faces

loss = autoencoder(
    vertices = vertices,
    faces = faces,
    face_edges = face_edges
)

loss.backward()

# after much training...

face_vertex_codes = autoencoder.tokenize(
    vertices = vertices,
    faces = faces,
    face_edges = face_edges
)

# now train your transformer to generate this sequence of codes

# to decode back to continuous coordinates for each face (9 vertices)

# (batch, number of faces, vertex (3), coord (3))

face_seq_coords = autoencoder.decode_from_codes_to_faces(face_vertex_codes)

```

## Todo

- [ ] autoencoder
    - [x] encoder sageconv with torch geometric
    - [x] proper scatter mean accounting for padding for meaning the vertices and RVQ the vertices before gathering back for decoder
    - [x] complete decoder and reconstruction loss + commitment loss
    - [ ] xcit linear attention in both encoder / decoder
    - [ ] add option to use residual FSQ / LFQ, latest quantization development
    - [ ] handle variable lengthed faces last - use sink tokens when scattering

- [ ] transformer
     - [ ] make sure it trains
         - [ ] take care of sos token automatically
         - [ ] take care of eos token automatically if sequence length or mask is passed in
     - [ ] properly mask out eos logit during generation
     - [ ] generation + cache kv
     - [ ] speculative decoding option
     - [ ] hierarchical transformers (using the RQ transformer)

## Citations

```bibtex
@inproceedings{Siddiqui2023MeshGPTGT,
    title   = {MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers},
    author  = {Yawar Siddiqui and Antonio Alliegro and Alexey Artemov and Tatiana Tommasi and Daniele Sirigatti and Vladislav Rosov and Angela Dai and Matthias Nie{\ss}ner},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:265457242}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@inproceedings{Leviathan2022FastIF,
    title   = {Fast Inference from Transformers via Speculative Decoding},
    author  = {Yaniv Leviathan and Matan Kalman and Y. Matias},
    booktitle = {International Conference on Machine Learning},
    year    = {2022},
    url     = {https://api.semanticscholar.org/CorpusID:254096365}
}
```

```bibtex
@misc{yu2023language,
    title   = {Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation}, 
    author  = {Lijun Yu and José Lezama and Nitesh B. Gundavarapu and Luca Versari and Kihyuk Sohn and David Minnen and Yong Cheng and Agrim Gupta and Xiuye Gu and Alexander G. Hauptmann and Boqing Gong and Ming-Hsuan Yang and Irfan Essa and David A. Ross and Lu Jiang},
    year    = {2023},
    eprint  = {2310.05737},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{elnouby2021xcit,
    title   = {XCiT: Cross-Covariance Image Transformers},
    author  = {Alaaeldin El-Nouby and Hugo Touvron and Mathilde Caron and Piotr Bojanowski and Matthijs Douze and Armand Joulin and Ivan Laptev and Natalia Neverova and Gabriel Synnaeve and Jakob Verbeek and Hervé Jegou},
    year    = {2021},
    eprint  = {2106.09681},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
