## Setup
Aside from the requirements in `requirements.txt` you'll need to install `reproject` from source,
specifically from `https://github.com/adobles96/reproject`. Clone the repo locally and then run
`pip install -e path/to/local/reproject`.

## Files
- `proj_tools.py`: Utility file that contains most of the heavy lifting around data manipulation:
    - Projections to and from healpix to flat
    - Smoothing logic to make patches uniresolution and to create the training inputs
    - Tiling logic (similar to ForSE+)
    - etc.
- `gen_dataset.py`: A script used to generate two sets of patches:
    1. `training_patches.pkl`: A dict containing both input and output image patches at different
        resolutions for trianing.
    2. `inference_patches.pkl`: A dict containing input image patches for inference.
- `dataset.py`: A module containing `torch.utils.data.Dataset` objects to be used during training
    and inference.
- `train.py`: The training script.
- `inference.py`: A script that runs a trained model on inference patches, to produce a
    super-resolution version of the full sky. It generates three files:
    1. `sr_patches.pkl`: A dict mapping resolution to an array of `(N, C, H, W)` patches.
    2. `sr_map.npy`: A numpy array of `(C, npix_full_sky)` holding a full sky healpix map for
        `Q` and `U`.
    3. `overlap_map.npy`: A numpy array of `(C, npix_full_sky)` holding a map of the full sky
        counting the number of overlaping patches per pixel.
    4. `effective_overlap_map.npy`: A numpy array of `(C, npix_full_sky)` holding a map of the full
        sky counting the number of overlaping patches per pixel, weighted by their apodizations.
- `plotting.py`: Plotting utilities used to generate plots for Minkowski Functionals and Power
    Spectra at the patch level.

## Workflow
1. Run `gen_dataset.py` with desired arguments. Eg:
`python gen_dataset.py --smooth_before_resize --npix_out 80` 
2. Decide on a run config (`config.yml`)
3. Run `train.py`. Eg:
`python train.py`
4. Run `inference.py` with the desired checkpoint produced during training. This will generate
    super-resolution patches and the full sky map. Eg:
`python inference.py --model_checkpoint 'path/to/checkpoint.ckpt' --patchsize_multiple 20`
5. Use outputs to run analyses. See `sample_plots.ipynb` for examples.

## Things to try?
- `MSELoss` instead of `L1Loss`
- Different `patchsize_multiple`
- Increase patch overlap to generate more training patches (decrease overlap for inference patches)
- Shift centers to generate more training data
- Rotate patches in full sky to generate more training data.
- Architectural changes
    - Transfer Learning
    - Pure convolution architecture
    - Pure vision transformer?
