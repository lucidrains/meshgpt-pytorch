<img src="./meshgpt.png" width="450px"></img>

## MeshGPT - Pytorch

Implementation of <a href="https://arxiv.org/abs/2311.15475">MeshGPT</a>, SOTA Mesh generation using Attention, in Pytorch

Will also add text conditioning, for eventual text-to-3d asset


Please visit the orginal repo for more details: 
https://github.com/lucidrains/meshgpt-pytorch
 
### Data sources:
#### ModelNet40: https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset/data

#### ShapeNet - [Extracted model labels](https://github.com/MarcusLoppe/meshgpt-pytorch/blob/main/shapenet_labels.json) Repository: https://huggingface.co/datasets/ShapeNet/shapenetcore-gltf

#### Objaverse - [Downloader](https://github.com/MarcusLoppe/Objaverse-downloader/tree/main) Repository: https://huggingface.co/datasets/allenai/objaverse
 
## Pre-trained autoencoder on the Objaverse dataset (14k meshes, only meshes that have max 250 faces): 
This is contains only autoencoder model, I'm currently training the transformer model.<br/>
Visit the discussions [Pre-trained autoencoder & data sourcing](https://github.com/lucidrains/meshgpt-pytorch/discussions/66) for more information about the training and details about the progression.

https://drive.google.com/drive/folders/1C1l5QrCtg9UulMJE5n_on4A9O9Gn0CC5?usp=sharing

<br/>
The auto-encoder results shows that it's possible to compress many mesh models into tokens which then can be decoded and reconstruct a mesh near perfection!<br/>
The auto-encoder was trained for 9 epochs for 20hrs on a single P100 GPU.<br/><br/>


The more compute heavy part is to train a transformer that can use these tokens learn the auto-encoder 'language'.<br/>
Using the codes as a vocabablity and learn the relationship between the the codes and it's ordering requires a lot compute to train compared to the auto-encoder.<br/>
So by using a single P100 GPU it will probaly take a few weeks till I can get out a pre-trained transformer. 
<br/>
Let me know if you wish to donate any compute or I can provide you with the dataset + training notebook.
<br/><br/>

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
num_layers = 23 
autoencoder = MeshAutoencoder(     
   decoder_dims_through_depth = (128,) * 3 + (192,) * 4 + (256,) * num_layers + (384,) * 3,   
   dim_codebook = 192 , 
   codebook_size = 16384 , 
   dim_area_embed = 16,
   dim_coor_embed = 16, 
   dim_normal_embed = 16,
   dim_angle_embed = 8,

   attn_decoder_depth = 8,
   attn_encoder_depth = 4
 ).to("cuda")    
```

#### Results, it's about 14k models so with the limited training time and hardware It's a great result.
![bild](https://github.com/lucidrains/meshgpt-pytorch/assets/65302107/18949b70-a982-4d22-9346-0f40ecf21cae)

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
