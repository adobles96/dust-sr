import argparse
from pathlib import Path
import pickle

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Model
from dataset import InferenceDataset
import proj_tools

DATA_DIR = Path('data')
BATCH_SIZE = 32


def main(args):
    # # Load model
    model = Model.load_from_checkpoint(
        args.model_checkpoint,
        loss=None,
        extra_metrics=None,
        metrics_on_inputs=None,
        extra_metrics_on_inputs=None,
    )
    model.eval()
    # Init Dataset
    dset = InferenceDataset(args.patches)
    loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    sr_patches = []
    for inputs in tqdm(loader, unit='patches'):
        # Run inference
        outputs = model(inputs.unsqueeze(dim=2)).squeeze().detach()
        sr_patches.append(dset.denorm(outputs).numpy())
    del dset.patches  # ease memory load

    # Reconstruct a patches dict
    sr_patches = np.concatenate(sr_patches, axis=0)
    sr_patch_dict = {}
    offset = 0
    for res, n in dset.res_to_npatches.items():
        sr_patch_dict[res] = sr_patches[offset: offset + n]
        offset += n
    with open(DATA_DIR / 'sr_patches.pkl', 'wb') as f:
        pickle.dump(sr_patch_dict, f)

    # Reproject to full sky
    npix = sr_patch_dict[10].shape[-1]
    with open(args.centers, 'rb') as f:
        centers = pickle.load(f)
    del centers[80]  # TODO delete
    new_map, overlap_map, effective_overlap_map = proj_tools.patches_to_full_sky(
        sr_patch_dict, centers, out_nside=args.out_nside, multiple=args.patchsize_multiple,
        npix=npix
    )
    np.save(DATA_DIR / 'sr_map.npy', new_map)
    np.save(DATA_DIR / 'overlap_map.npy', overlap_map)
    np.save(DATA_DIR / 'effective_overlap_map.npy', effective_overlap_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint", type=Path, help="Path to the model checkpoint to use."
    )
    parser.add_argument(
        "--patchsize_multiple", type=int, help="Ratio of angular patch size to output resolution"
        " (ie after applying the model): `patchsize = multiple * res`"
    )
    parser.add_argument(
        "--patches", type=Path, default=DATA_DIR / "inference_patches.pkl",
        help="Path to inference patches."
    )
    parser.add_argument(
        "--centers", type=Path, default=DATA_DIR / "inference_centers.pkl",
        help="Path to inference centers."
    )
    parser.add_argument(
        "--out_nside", type=int, default=4096, help="The nside for the output SR sky maps."
    )
    args = parser.parse_args()
    main(args)
