import pytest
import torch

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer
)

@pytest.mark.parametrize('adaptive_rmsnorm', (True, False))
def test_readme(adaptive_rmsnorm):

    autoencoder = MeshAutoencoder(
        num_discrete_coors = 128
    )

    # mock inputs

    vertices = torch.randn((2, 121, 3))            # (batch, num vertices, coor (3))
    faces = torch.randint(0, 121, (2, 2, 3))      # (batch, num faces, vertices (3))

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
        max_seq_len = 60,
        num_sos_tokens = 1,
        fine_cross_attend_text = True,
        text_cond_with_film = False,
        condition_on_text = True,
        coarse_post_gateloop_depth = 1,
        coarse_adaptive_rmsnorm = adaptive_rmsnorm
    )

    loss = transformer(
        vertices = vertices,
        faces = faces,
        texts = ['a high chair', 'a small teapot']
    )

    loss.backward()

    faces_coordinates, face_mask = transformer.generate(texts = ['a small chair'], cond_scale = 3.)

def test_cache():
    # test that the output for generation with and without kv (and optional gateloop) cache is equivalent

    autoencoder = MeshAutoencoder(
        num_discrete_coors = 128
    )

    transformer = MeshTransformer(
        autoencoder,
        dim = 512,
        max_seq_len = 12
    )

    uncached_faces_coors, _ = transformer.generate(cache_kv = False, temperature = 0)
    cached_faces_coors, _ = transformer.generate(cache_kv = True, temperature = 0)

    assert torch.allclose(uncached_faces_coors, cached_faces_coors)
