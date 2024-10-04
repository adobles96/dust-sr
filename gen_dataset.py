from argparse import ArgumentParser
import os
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm

import proj_tools
DATA_DIR = Path('data')
MAX_TRAIN_OUTPUT_RES = 60


def generate_training_patches(args):
    # load maps
    gi, gq, gu, tau, hiq, hiu, res_mask = proj_tools.get_full_sky_maps()
    if os.path.exists(DATA_DIR / 'training_centers.pkl'):
        with open(DATA_DIR / 'training_centers.pkl', 'rb') as f:
            centers = pickle.load(f)
        print('Loaded centers')
    else:
        print('Generating centers')
        centers = proj_tools.generate_centers(
            res_mask, remove_overlaps=True, multiple=args.patchsize_multiple
        )
        with open(DATA_DIR / 'training_centers.pkl', 'wb') as f:
            pickle.dump(centers, f)
    smoothed_maps = proj_tools.get_smoothed_maps(np.stack([gi, gq, gu], axis=0), res_mask)
    patches = {res: {'in': [], 'out': []} for res in centers.keys() if res <= MAX_TRAIN_OUTPUT_RES}
    for res in tqdm(centers.keys(), desc='resolutions', position=0):
        if res > MAX_TRAIN_OUTPUT_RES:
            # we don't need to train on input resolutions higher than 80' (ie output resolutions
            # higher than 20')
            continue
        lon = centers[res]['lon']
        lat = centers[res]['lat']
        pixsize_in = proj_tools.get_patch_size_deg(res, args.patchsize_multiple) / args.npix_in
        pixsize_out = proj_tools.get_patch_size_deg(res, args.patchsize_multiple) / args.npix_out
        for x, y in tqdm(
            zip(lon, lat), desc=f'patches {int(res)}\'', position=1, leave=False, total=len(lon)
        ):
            header_in = proj_tools.set_header(x, y, pixsize_in, args.npix_in)
            header_out = proj_tools.set_header(x, y, pixsize_out, args.npix_out)
            factor = 4 if args.smooth_before_resize else 1
            patches_in = np.stack([
                proj_tools.h2f(tau, header_in),
                proj_tools.h2f(hiq, header_in),
                proj_tools.h2f(hiu, header_in),
                proj_tools.h2f(smoothed_maps[factor * res][1], header_in),
                proj_tools.h2f(smoothed_maps[factor * res][2], header_in),
            ], axis=0)
            patches_out = np.stack([
                proj_tools.h2f(smoothed_maps[res][1], header_out),
                proj_tools.h2f(smoothed_maps[res][2], header_out),
            ])
            patches[res]['in'].append(patches_in)
            patches[res]['out'].append(patches_out)
        patches[res]['in'] = np.stack(patches[res]['in'], axis=0)
        patches[res]['out'] = np.stack(patches[res]['out'], axis=0)
    with open(DATA_DIR / 'training_patches.pkl', 'wb') as f:
        pickle.dump(patches, f)


def generate_inference_patches(args):
    # load maps
    gi, gq, gu, tau, hiq, hiu, res_mask = proj_tools.get_full_sky_maps()
    multiple = (
        args.patchsize_multiple / 4 if args.smooth_before_resize else args.patchsize_multiple
    )
    if os.path.exists(DATA_DIR / 'inference_centers.pkl'):
        with open(DATA_DIR / 'inference_centers.pkl', 'rb') as f:
            centers = pickle.load(f)
        print('Loaded centers')
    else:
        print('Generating centers')
        centers = proj_tools.generate_centers(
            res_mask, remove_overlaps=True, multiple=multiple
        )
        with open(DATA_DIR / 'inference_centers.pkl', 'wb') as f:
            pickle.dump(centers, f)
    smoothed_maps = proj_tools.get_smoothed_maps(np.stack([gi, gq, gu], axis=0), res_mask)
    patches = {res: [] for res in centers.keys()}
    for res in tqdm(centers.keys(), desc='resolutions', position=0):
        lon = centers[res]['lon']
        lat = centers[res]['lat']
        pixsize = proj_tools.get_patch_size_deg(res, multiple) / args.npix_in
        for x, y in tqdm(
            zip(lon, lat), desc=f'patches {int(res)}\'', position=1, leave=False, total=len(lon)
        ):
            header = proj_tools.set_header(x, y, pixsize, args.npix_in)
            patches_in = np.stack([
                proj_tools.h2f(tau, header),
                proj_tools.h2f(hiq, header),
                proj_tools.h2f(hiu, header),
                proj_tools.h2f(smoothed_maps[res][1], header),
                proj_tools.h2f(smoothed_maps[res][2], header),
            ], axis=0)
            patches[res].append(patches_in)
        patches[res] = np.stack(patches[res], axis=0)
    with open(DATA_DIR / 'inference_patches.pkl', 'wb') as f:
        pickle.dump(patches, f)


def consistency_check():
    """ Checks consitency of generated training and inference patches. """
    with open(DATA_DIR / 'training_patches.pkl', 'rb') as f:
        training_patches = pickle.load(f)
    with open(DATA_DIR / 'inference_patches.pkl', 'rb') as f:
        inference_patches = pickle.load(f)
    print()
    print('Training Patch Counts:')
    for res, patches in training_patches.items():
        print(f'{res}: {len(patches["in"]):,} in --- {len(patches["out"]):,} out')
        assert len(patches['in']) == len(patches['out'])
    print(f'Total Patches: {sum([len(p["in"]) for p in training_patches.values()]):,}')
    print()
    print('Inference Patch Counts:')
    for res, patches in inference_patches.items():
        print(f'{res}: {len(patches):,}')
    print(f'Total Patches: {sum([len(p) for p in inference_patches.values()]):,}')
    all_inputs = np.concatenate([p['in'] for p in training_patches.values()], axis=0)
    pix_in = all_inputs.shape[-1]
    print()
    print('Input pixel size:', pix_in)
    print(f'Total Pixels in Training Inputs: {len(all_inputs) * pix_in**2:,}')
    all_outputs = np.concatenate([p['out'] for p in training_patches.values()], axis=0)
    pix_out = all_outputs.shape[-1]
    print('Output pixel size:', pix_out)
    print(f'Total Pixels in Training Outputs: {len(all_outputs) * pix_out**2:,}')
    all_inf = np.concatenate([p for p in inference_patches.values()], axis=0)
    pix_inf = all_inf.shape[-1]
    print('Inference pixel size:', pix_inf)
    print()
    print(f'Total Pixels in Inference Patches: {len(all_inf) * pix_inf**2:,}')
    for i, map_name in enumerate(['tau', 'hiq', 'hiu', 'q_lr', 'u_lr']):
        print(map_name)
        print()
        print('Training Inputs:')
        print('min:', all_inputs[:, i].min())
        print('max:', all_inputs[:, i].max())
        print('mean:', all_inputs[:, i].mean())
        print('std:', all_inputs[:, i].std())
        print()
        print('Inference:')
        print('min:', all_inf[:, i].min())
        print('max:', all_inf[:, i].max())
        print('mean:', all_inf[:, i].mean())
        print('std:', all_inf[:, i].std())
        print()
    print('––––––––––––––––––––––––––')
    print()
    for i, map_name in enumerate(['q_hr', 'u_hr']):
        print(map_name)
        print('Training Outputs:')
        print('min:', all_outputs[:, i].min())
        print('max:', all_outputs[:, i].max())
        print('mean:', all_outputs[:, i].mean())
        print('std:', all_outputs[:, i].std())
        print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--patchsize_multiple", type=int, default=20,
                        help="Ratio of patch size to output resolution, ie 20 means patches have "
                        "angular size of 20 * res_out per side.")
    parser.add_argument("--smooth_before_resize", action="store_true")
    parser.add_argument("--npix_in", type=int, default=80)
    parser.add_argument("--npix_out", type=int, default=320)
    args = parser.parse_args()
    generate_training_patches(args)
    generate_inference_patches(args)
    consistency_check()
