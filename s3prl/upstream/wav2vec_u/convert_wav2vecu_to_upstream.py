import os
import torch
import logging

user = os.path.expanduser('~')

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    wav2vec_u_model_path = os.path.join(
        user, ".cache/torch/hub/s3prl_cache/wav2vec_u/wav2vec_u_model.pt")
    wav2vec_model_path = os.path.join(
        user, ".cache/torch/hub/s3prl_cache/wav2vec_u/wav2vec_vox_new.pt")
    wav2vec_u_model_dict = os.path.join(
        user, ".cache/torch/hub/s3prl_cache/wav2vec_u/dict.phn.txt")
    output_path = os.path.join(
        user, ".cache/torch/hub/s3prl_cache/wav2vec_u/joint_wav2vec_u_model.pth")
    layer_index = 14

    logging.info("start merging config")

    joint_dict = {}
    joint_dict["ssl"] = wav2vec_model_path
    joint_dict["u_model"] = wav2vec_u_model_path
    joint_dict["u_dict"] = wav2vec_u_model_dict
    joint_dict["interface"] = {
        "mode": "single_layer",
        "value": 14,
        "u_downsample_rate": 3,
    }

    logging.info("saving joint dictionary")
    torch.save(joint_dict, output_path)
