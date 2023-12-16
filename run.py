import argparse
import torch
import wandb
import os
from dataset.dataset import MeshDataset
from meshgpt_pytorch import MeshAutoencoder, MeshAutoencoderTrainer
from datetime import datetime
import re
import json
import wandb
from meshgpt_pytorch import MeshTransformer

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_directory = args.dataset_directory
    data_augment = args.data_augment
    dataset = MeshDataset(dataset_directory, data_augment)
    autoencoder = None

    run = wandb.init(
        project="meshgpt-pytorch",
        config={
            "autoencoder_path": args.autoencoder_path,
            "inference_only": args.inference_only,
            "get_max_face_count": dataset.get_max_face_count(),
            "autoencoder_learning_rate": args.autoencoder_learning_rate,
            "transformer_learning_rate": args.transformer_learning_rate,
            "architecture": "MeshGPT",
            "dataset_directory": dataset_directory,
            "data_augment": data_augment,
            "autoencoder_train": args.autoencoder_train,
            "transformer_train": args.transformer_train,
            "batch_size": args.batch_size,
            "grad_accum_every": args.grad_accum_every,
            "checkpoint_every": args.checkpoint_every,
            "device": str(device),
            "num_quantizers": args.num_quantizers,
            "autoencoder": {
                "dim": args.dim,
                "encoder_depth": args.encoder_depth,
                "decoder_depth": args.decoder_depth,
                "num_discrete_coors": args.num_discrete_coors,
            },
            "dataset_size": dataset.__len__(),
        },
    )

    if args.autoencoder_path:
        autoencoder = MeshAutoencoder(
            num_quantizers=run.config.num_quantizers,
            dim=run.config.autoencoder["dim"],
            encoder_depth=run.config.autoencoder["encoder_depth"],
            decoder_depth=run.config.autoencoder["decoder_depth"],
            num_discrete_coors=run.config.autoencoder["num_discrete_coors"],
        ).to(device)
        autoencoder.init_and_load_from(run.config.mesh_autoencoder_path)
    else:
        autoencoder = MeshAutoencoder(
            dim=run.config.autoencoder["dim"],
            encoder_depth=run.config.autoencoder["encoder_depth"],
            decoder_depth=run.config.autoencoder["decoder_depth"],
            num_discrete_coors=run.config.autoencoder["num_discrete_coors"],
        ).to(device)
        train_autoencoder(run, dataset, autoencoder)
    
    seq_len = dataset.get_max_face_count() * 6
    print(f"Sequence length: {seq_len}")
    transformer = train_transformer(autoencoder, run, dataset, device, seq_len)
    process_mesh_data(run, device, transformer)

def train_autoencoder(run, dataset, autoencoder):

    trainer = MeshAutoencoderTrainer(
        autoencoder,
        num_train_steps=2000,
        dataset=dataset,
        batch_size=run.config.batch_size,
        grad_accum_every=run.config.grad_accum_every,
        checkpoint_every_epoch=run.config.checkpoint_every,
        learning_rate=run.config.autoencoder_learning_rate,
        use_wandb_tracking=True,
    )
    trainer.train(run.config.autoencoder_train)

    current_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    trainer.save(f"checkpoints/mesh_autoencoder_{os.path.basename(run.config.dataset_directory)}_{current_time}.pt")

    return autoencoder

from datetime import datetime
from meshgpt_pytorch import MeshTransformer, MeshTransformerTrainer

def train_transformer(autoencoder, run, dataset, device, seq_len):
    transformer = MeshTransformer(
        autoencoder,
        dim=run.config.autoencoder["dim"],
        max_seq_len=seq_len,
    ).to(device)

    transformer_trainer = MeshTransformerTrainer(
        transformer,
        num_train_steps=2000,
        dataset=dataset,
        batch_size=wandb.config.batch_size,
        grad_accum_every=wandb.config.grad_accum_every,
        checkpoint_every_epoch=wandb.config.checkpoint_every,
        warmup_steps=500,
        learning_rate=wandb.config.transformer_learning_rate,
        use_wandb_tracking=True,
    )

    transformer_trainer.train(run.config.transformer_train)

    current_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    transformer_trainer.save(f"checkpoints/mesh_transformer_{os.path.basename(run.config.dataset_directory)}_{current_time}.pt")
    return transformer

def process_mesh_data(run, device, transformer):
    codes = transformer.generate(return_codes=True)

    transformer.autoencoder.eval()

    continuous_coors = transformer.autoencoder.decode_from_codes_to_faces(codes)

    continuous_coors_list = continuous_coors.cpu().tolist()

    with open("continuous_coors.json", "w") as f:
        json.dump(continuous_coors.tolist(), f)

    flat_list = [item for sublist in continuous_coors_list for item in sublist]

    vertices = [vertex for sublist in flat_list for vertex in sublist]

    faces = [[i, i + 1, i + 2] for i in range(0, len(vertices), 3)]

    dataset.convert_to_glb((vertices, faces), "output.glb")
    dataset.convert_to_obj((vertices, faces), "output.obj")

    def encode_to_pua(codes):
        flat_codes = [
            item for sublist in codes for subsublist in sublist for item in subsublist
        ]
        return "".join(chr(code + 0xF0000) for code in flat_codes)

    encoded_codes = encode_to_pua(codes.cpu().tolist())

    with open("output.obj", "r") as file:
        obj_contents = file.read()

    new_data = [
        [
            {
                "role": "system",
                "content": "This assistant can understand 3D models using the meshgpt-pytorch Unicode plane 15 codebook for 16384 triangles and the .obj 3d format.",
            },
            {"role": "user", "content": f"{encoded_codes}"},
            {"role": "assistant", "content": f"{obj_contents}"},
        ]
    ]

    data = []
    try:
        with open("chatml.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("The file 'chatml.jsonl' does not exist.")

    data = new_data + data

    with open("chatml.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory", default="dataset/unit_test")
    parser.add_argument("--data_augment", type=int, default=2)
    parser.add_argument("--autoencoder_learning_rate", type=float, default=0.4)
    parser.add_argument("--transformer_learning_rate", type=float, default=0.2)
    parser.add_argument("--autoencoder_train", type=int, default=200) # 200
    parser.add_argument("--transformer_train", type=int, default=375) # 375
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_every", type=int, default=1)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--decoder_depth", type=int, default=6)
    parser.add_argument("--num_discrete_coors", type=int, default=128)
    parser.add_argument("--inference_only", action='store_true')
    parser.add_argument("--autoencoder_path")
    parser.add_argument("--num_quantizers", type=int, default=2)
    args = parser.parse_args()

    main(args)
