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
