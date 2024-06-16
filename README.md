<img src="./meshgpt.png" width="450px"></img>

## MeshGPT - Pytorch

Implementation of <a href="https://arxiv.org/abs/2311.15475">MeshGPT</a>, SOTA Mesh generation using Attention, in Pytorch

Will also add text conditioning, for eventual text-to-3d asset

Please join <a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a> if you are interested in collaborating with others to replicate this work

Update: <a href="https://github.com/MarcusLoppe">Marcus</a> has trained and uploaded <a href="https://huggingface.co/MarcusLoren/MeshGPT-preview">a working model</a> to 🤗 Huggingface!

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a>, <a href="https://a16z.com/supporting-the-open-source-ai-community/">A16Z Open Source AI Grant Program</a>, and <a href="https://huggingface.co/">🤗 Huggingface</a> for the generous sponsorships, as well as my other sponsors, for affording me the independence to open source current artificial intelligence research

- <a href="https://github.com/arogozhnikov/einops">Einops</a> for making my life easy

- <a href="https://github.com/MarcusLoppe">Marcus</a> for the initial code review (pointing out some missing derived features) as well as running the first successful end-to-end experiments

- <a href="https://github.com/MarcusLoppe">Marcus</a> for the <a href="https://github.com/lucidrains/meshgpt-pytorch/issues/18#issuecomment-1859214710">first successful training</a> of a collection of shapes conditioned on labels

- <a href="https://github.com/qixuema">Quexi Ma</a> for finding numerous bugs with automatic eos handling

- <a href="https://github.com/thuliu-yt16">Yingtian</a> for finding a bug with the gaussian blurring of the positions for spatial label smoothing

- <a href="https://github.com/MarcusLoppe">Marcus</a> yet again for running the experiments to validate that it is possible to extend the system from triangles to <a href="https://github.com/lucidrains/meshgpt-pytorch/issues/54#issuecomment-1906789076">quads</a>

- <a href="https://github.com/MarcusLoppe">Marcus</a> for identifying <a href="https://github.com/lucidrains/meshgpt-pytorch/issues/80">an issue</a> with text conditioning and for running all the experiments that led to it being resolved

## Install

```bash
$ pip install meshgpt-pytorch
```

## Usage

```python
import torch

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer
)

# autoencoder

autoencoder = MeshAutoencoder(
    num_discrete_coors = 128
)

# mock inputs

vertices = torch.randn((2, 121, 3))            # (batch, num vertices, coor (3))
faces = torch.randint(0, 121, (2, 64, 3))      # (batch, num faces, vertices (3))

# make sure faces are padded with `-1` for variable lengthed meshes

# forward in the faces

loss = autoencoder(
    vertices = vertices,
    faces = faces
)

loss.backward()

# after much training...
# you can pass in the raw face data above to train a transformer to model this sequence of face vertices

transformer = MeshTransformer(
    autoencoder,
    dim = 512,
    max_seq_len = 768
)

loss = transformer(
    vertices = vertices,
    faces = faces
)

loss.backward()

# after much training of transformer, you can now sample novel 3d assets

faces_coordinates, face_mask = transformer.generate()

# (batch, num faces, vertices (3), coordinates (3)), (batch, num faces)
# now post process for the generated 3d asset

```

For <a href="https://www.youtube.com/watch?v=NXX0dKw4SjI">text-conditioned 3d shape synthesis</a>, simply set `condition_on_text = True` on your `MeshTransformer`, and then pass in your list of descriptions as the `texts` keyword argument

ex.
```python
transformer = MeshTransformer(
    autoencoder,
    dim = 512,
    max_seq_len = 768,
    condition_on_text = True
)


loss = transformer(
    vertices = vertices,
    faces = faces,
    texts = ['a high chair', 'a small teapot'],
)

loss.backward()

# after much training of transformer, you can now sample novel 3d assets conditioned on text

faces_coordinates, face_mask = transformer.generate(
    texts = ['a long table'],
    cond_scale = 3.  # a cond_scale > 1. will enable classifier free guidance - can be placed anywhere from 3. - 10.
)

```

If you want to tokenize meshes, for use in your multimodal transformer, simply invoke `.tokenize` on your autoencoder (or same method on autoencoder trainer instance for the exponentially smoothed model)

```python

mesh_token_ids = autoencoder.tokenize(
    vertices = vertices,
    faces = faces
)

# (batch, num face vertices, residual quantized layer)
```

## Typecheck

At the project root, run

```bash
$ cp .env.sample .env
```

## Todo

- [x] autoencoder
    - [x] encoder sageconv with torch geometric
    - [x] proper scatter mean accounting for padding for meaning the vertices and RVQ the vertices before gathering back for decoder
    - [x] complete decoder and reconstruction loss + commitment loss
    - [x] handle variable lengthed faces
    - [x] add option to use residual LFQ, latest quantization development that scales code utilization
    - [x] xcit linear attention in encoder and decoder
    - [x] figure out how to auto-derive `face_edges` directly from faces and vertices
    - [x] embed any derived values (area, angles, etc) from the vertices before sage convs
    - [ ] add an extra graph conv stage in the encoder, where vertices are enriched with their connected vertex neighbors, before aggregating into faces. make optional
    - [ ] allow for encoder to noise the vertices, so autoencoder is a bit denoising. consider conditioning decoder on noise level, if varying

- [ ] transformer
    - [x] properly mask out eos logit during generation
    - [x] make sure it trains
        - [x] take care of sos token automatically
        - [x] take care of eos token automatically if sequence length or mask is passed in
    - [x] handle variable lengthed faces
        - [x] on forwards
        - [x] on generation, do all eos logic + substitute everything after eos with pad id
    - [x] generation + cache kv

- [x] trainer wrapper with hf accelerate
    - [x] autoencoder - take care of ema
    - [x] transformer

- [x] text conditioning using own CFG library
    - [x] complete preliminary text conditioning
    - [x]  make sure CFG library can support passing in arguments to the two separate calls when cond scaling (as well as aggregating their outputs)
    - [ ] polish up the magic dataset decorator and see if it can be moved to CFG library
- [x] hierarchical transformers (using the RQ transformer)
- [x] fix caching in simple gateloop layer in other repo
- [x] local attention
- [x] fix kv caching for two-staged hierarchical transformer - 7x faster now, and faster than original non-hierarchical transformer
- [x] fix caching for gateloop layers
- [x] allow for customization of model dimensions of fine vs coarse attention network
- [x] figure out if autoencoder is really necessary - it is necessary, ablations are in the paper
    - [x] when mesh discretizer is passed in, one can inject inter-face attention with the relative distance
    - [x] additional embeddings (angles, area, normal), can also be appended before coarse transformer attention

- [ ] make transformer efficient
    - [ ] reversible networks

- [ ] speculative decoding option

- [ ] spend a day on documentation

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
@article{Lee2022AutoregressiveIG,
    title   = {Autoregressive Image Generation using Residual Quantization},
    author  = {Doyup Lee and Chiheon Kim and Saehoon Kim and Minsu Cho and Wook-Shin Han},
    journal = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year    = {2022},
    pages   = {11513-11522},
    url     = {https://api.semanticscholar.org/CorpusID:247244535}
}
```

```bibtex
@inproceedings{Katsch2023GateLoopFD,
    title   = {GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling},
    author  = {Tobias Katsch},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:265018962}
}
```
