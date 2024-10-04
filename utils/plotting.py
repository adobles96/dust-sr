from typing import Union

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import pymaster as nmt
import torch


# Global settings
# matplotlib.rc('font', family='serif', serif='cm10')
# matplotlib.rc('text', usetex=True)


def plot_polarization(image: torch.Tensor):
    # cm = "inferno" if i % 9 == 0 else "twilight"
    vmax = torch.max(torch.abs(image)).item()
    # vmin = torch.min(torch.abs(image)).item() if i % 9 == 0 else -vmax
    vmin = -vmax
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    image = plt.colormaps['inferno'](norm(image.squeeze().cpu().numpy()))
    return plt.imshow(image)


def plot_tau(image: torch.Tensor):
    vmax = torch.max(torch.abs(image)).item()
    vmin = torch.min(torch.abs(image)).item()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    image = plt.colormaps['inferno'](norm(image.squeeze().cpu().numpy()))
    return plt.imshow(image)


# TODO use Jay's package
@jit(nopython=True)
def estimate_marchingsquare_fix(data, threshold):
    """ Correction? Copied from ForSE+ Repo."""
    width = data.shape[0]
    height = data.shape[1]
    f = u = chi = 0
    for i in range(width - 1):
        for j in range(height - 1):
            pattern = (
                (data[i, j] > threshold) 
                + (data[i+1, j] > threshold) * 2
                + (data[i+1, j+1] > threshold) * 4
                + (data[i, j+1] > threshold) * 8
            )
            if pattern == 0:
                continue
            elif pattern == 1:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])
                f = f + 0.5 * a1 * a4
                u = u + np.sqrt(a1 * a1 + a4 * a4)
                chi = chi + 0.25
                continue
            elif pattern == 2:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (data[i + 1, j] - data[i + 1, j + 1])
                f = f + 0.5 * (1 - a1) * (a2)
                u = u + np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                chi = chi + 0.25
                continue
            elif pattern == 3:
                a2 = (data[i + 1, j] - threshold) / (data[i + 1, j] - data[i + 1, j + 1])
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])
                f = f + a2 + 0.5 * (a4 - a2)
                u = u + np.sqrt(1 + (a4 - a2) * (a4 - a2))
                continue
            elif pattern == 4:
                a2 = (data[i + 1, j] - threshold) / (data[i + 1, j] - data[i + 1, j + 1])
                a3 = (data[i, j + 1] - threshold) / (data[i, j + 1] - data[i + 1, j + 1])
                f = f + 0.5 * (1 - a2) * (1 - a3)
                u = u + np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                chi = chi + 0.25
                continue
            elif pattern == 5:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (data[i + 1, j] - data[i + 1, j + 1])
                a3 = (data[i, j + 1] - threshold) / (data[i, j + 1] - data[i + 1, j + 1])
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])
                f = f + 1 - 0.5 * (1 - a1) * a2 - 0.5 * a3 * (1 - a4)
                u = u + np.sqrt((1 - a1) * (1 - a1) + a2 * a2) + np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                chi = chi + 0.5
                continue
            elif pattern == 6:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j])
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1])
                f = f + (1 - a3) + 0.5 * (a3 - a1)
                u = u + np.sqrt(1 + (a3 - a1) * (a3 - a1))
                continue
            elif pattern == 7:
                a3 = (data[i, j + 1] - threshold) / (data[i, j + 1] - data[i + 1, j + 1])
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])
                f = f + 1 - 0.5 * a3 * (1 - a4)
                u = u + np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                chi = chi - 0.25
                continue
            elif pattern == 8:
                a3 = (data[i, j + 1] - threshold) / (data[i, j + 1] - data[i + 1, j + 1])
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])
                f = f + 0.5 * a3 * (1 - a4)
                u = u + np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                chi = chi + 0.25
                continue
            elif pattern == 9:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a3 = (data[i, j + 1] - threshold) / (data[i, j + 1] - data[i + 1, j + 1])
                f = f + a1 + 0.5 * (a3 - a1)
                u = u + np.sqrt(1 + (a3 - a1) * (a3 - a1))
                continue
            elif pattern == 10:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (data[i + 1, j] - data[i + 1, j + 1])
                a3 = (data[i, j + 1] - threshold) / (data[i, j + 1] - data[i + 1, j + 1])
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])
                f = f + 1 - 0.5 * a1 * a4 + 0.5 * (1 - a2) * (1 - a3)
                u = u + np.sqrt(a1 * a1 + a4 * a4) + np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                chi = chi + 0.5
                continue
            elif pattern == 11:
                a2 = (data[i + 1, j] - threshold) / (data[i + 1, j] - data[i + 1, j + 1])
                a3 = (data[i, j + 1] - threshold) / (data[i, j + 1] - data[i + 1, j + 1])
                f = f + 1 - 0.5 * (1 - a2) * (1 - a3)
                u = u + np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                chi = chi - 0.25
                continue
            elif pattern == 12:
                a2 = (data[i + 1, j] - threshold) / (data[i + 1, j] - data[i + 1, j + 1])
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])
                f = f + (1 - a2) + 0.5 * (a2 - a4)
                u = u + np.sqrt(1 + (a2 - a4) * (a2 - a4))
                continue
            elif pattern == 13:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (data[i + 1, j] - data[i + 1, j + 1])
                f = f + 1 - .5 * (1 - a1) * a2
                u = u + np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                chi = chi - 0.25
                continue
            elif pattern == 14:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])
                f = f + 1 - 0.5 * a1 * a4
                u = u + np.sqrt(a1 * a1 + a4 * a4)
                chi = chi - 0.25
                continue
            elif pattern == 15:
                f += 1
                continue
    return f, u, chi


def get_functionals_fix(im, nevals=25):
    """ Copy-pasted from https://github.com/yaojian95/ForSEplus/blob/60f4aa13a3cdcab6254374a1d9eb4cf92b2b9c0f/src/ForSEplus/utility.py#L348 """
    vmin = im.min(axis=(2, 3))
    vmax = im.max(axis=(2, 3))
    rhos = np.linspace(vmin, vmax, nevals).transpose(1, 2, 0)
    f = np.zeros_like(rhos)
    u = np.zeros_like(rhos)
    chi = np.zeros_like(rhos)
    for k, rho in np.ndenumerate(rhos):
        f[k], u[k], chi[k] = estimate_marchingsquare_fix(im[k[0], k[1]], rho)  # really slow?
    return rhos, f, u, chi


def rescale_min_max(img, a=-1, b=1):
    return (
        (b - a) * (img - img.min(axis=(-2, -1), keepdims=True))
        / (img.max(axis=(-2, -1), keepdims=True) - img.min(axis=(-2, -1), keepdims=True)) + a
    )


def get_gauss_QU(resol: float, n_samples: int = 1):
    """ Generates a random realization of GRF."""
    l, cl_tt, cl_ee, cl_bb, cl_te = np.loadtxt('data/cls.txt', unpack=True)
    beam = np.exp(-(0.25 * np.pi/180 * l)**2)
    cl_tt *= beam
    cl_ee *= beam
    cl_bb *= beam
    cl_te *= beam
    Lx = np.radians(20 * resol / 60)
    Ly = np.radians(20 * resol / 60)
    qs, us = [], []
    for _ in range(n_samples):
        _, mpq, mpu = nmt.synfast_flat(
            80, 80, Lx, Ly, np.array([cl_tt, cl_te, 0 * cl_tt, cl_ee, 0 * cl_ee, cl_bb]), [0, 2]
        )
        qs.append(mpq)
        us.append(mpu)
    return np.stack([np.stack(qs, axis=0), np.stack(us, axis=0)], axis=1)


def plot_MF(imgs: Union[torch.Tensor, np.ndarray], plot_gauss: bool = False, resol: int = 15):
    """ Plots Minkowski Functionals for the given images.

    Args:
        imgs (torch.Tensor | np.ndarray): A batch of images of shape (batch, C, H, W), where C is
            the number of channels (eg Q, U, tau etc.), H is the height and W is the width.
        plot_gauss (bool): Whether to plot the gaussian case as well.
        resol (int): The resolution in arcminutes.
    """
    # TODO Add overlapping stuff
    # See here for generating gaussian case:
    # https://namaster.readthedocs.io/en/latest/api/pymaster.utils.html#pymaster.utils.synfast_flat
    imgs = _tensor_to_np(imgs)
    # normed_Q = imgs[:, 0] / 1  # should divide by Q std and multiply by ss ratio Q??
    mT = rescale_min_max(imgs)
    rhos, f, u, chi = get_functionals_fix(mT)
    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
    S = ['Q', 'U']  # CHECK ORDERING!!!
    if plot_gauss:
        # (n, 80, 80)
        gauss = get_gauss_QU(resol, imgs.shape[0])
        mT = rescale_min_max(gauss)
        g_rhos, g_f, g_u, g_chi = get_functionals_fix(mT)
        g_y = [g_f, g_u, g_chi]
    for i in range(2):
        for j, y in enumerate([f, u, chi]):
            # label = r'$\tilde{m}^{%s, 20^{\circ}}_{12^\prime}$' % S[j] if j == 0 else None
            # label_t = r'$\tilde{m}^{I, 20^{\circ}}_{12^\prime}$' if j == 0 else None
            label = f'Prediction {S[i]}' if j == 0 else None
            label_t = f'Ground Truth {S[i]}' if j == 0 else None
            axes[i, j].ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
            mean = np.mean(y[:, i, ...], axis=0)
            std = np.std(y[:, i, ...], axis=0)
            # all rhos are the same
            axes[i, j].fill_between(
                rhos[0, 0], mean - std, mean + std, lw=1, alpha=0.5, color='#F87217', label=label
            )
            axes[i, j].plot(rhos[0, 0], mean, lw=3, ls='--', color='#D04A00')
            mean_gt = np.mean(y[:, 2 + i, ...], axis=0)
            std_gt = np.std(y[:, 2 + i, ...], axis=0)
            axes[i, j].fill_between(
                rhos[0, 0], mean_gt - std_gt, mean_gt + std_gt, lw=2, label=label_t,
                edgecolor='black', facecolor='None'
            )
            axes[i, j].plot(rhos[0, 0], mean_gt, lw=2, ls='--', color='black')
            axes[i, j].set_ylabel(r'$\mathcal{V}_{%s}(\rho$) %s'%(j, S[i]), fontsize=25)
            if plot_gauss:
                mean_g = np.mean(g_y[j][:, i, ...], axis=0)
                std_g = np.std(g_y[j][:, i, ...], axis=0)
                axes[i, j].fill_between(
                    g_rhos[0, 0], mean_g - std_g, mean_g + std_g, lw=1, alpha=0.5, color='#569A62',
                    label='Gaussian'
                )
                axes[i, j].plot(g_rhos[0, 0], mean_g, lw=3, ls='--', color='#246830')
            if i == 1:
                axes[i, j].set_xlabel(r'$\rho$', fontsize=25)
            if j == 0:
                axes[i, j].legend()
    plt.tight_layout()
    return fig


def get_patch_size_deg(res: float) -> float:
    return 20 * res / 60


def make_apo_mask(H, W, Lx, Ly, res):
    mask = np.ones((H, W)).flatten()
    xarr = np.ones(H)[:, None] * np.arange(W)[None, :] * Lx / W
    yarr = np.ones(W)[None, :] * np.arange(H)[:, None] * Ly / H
    mask[np.where(xarr.flatten() < Lx / 100.)] = 0
    mask[np.where(xarr.flatten() > 99 * Lx / 100.)] = 0
    mask[np.where(yarr.flatten() < Ly / 100.)] = 0
    mask[np.where(yarr.flatten() > 99 * Ly / 100.)] = 0
    mask = mask.reshape([H, W])
    aposize = get_patch_size_deg(res) * 0.1
    mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=aposize, apotype="C1")
    return mask


def _tensor_to_np(imgs: Union[torch.Tensor, np.ndarray]):
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()
    elif not isinstance(imgs, np.ndarray):
        raise ValueError('imgs must be a torch.Tensor or np.ndarray.')
    return imgs


def normalize(Q, U):
    # TODO implement
    return Q, U


def plot_power_spectra_patch(imgs: Union[torch.Tensor, np.ndarray], resol: float = 15):
    """ Plots the power spectra for the given images.

    Args:
        imgs (torch.Tensor | np.ndarray): A batch of images of shape (batch, C, H, W), where C is
            the number of channels (eg Q, U, tau etc.), H is the height and W is the width.
        resol (float): The image resolution in arcminutes.
    """
    imgs = _tensor_to_np(imgs) * 1e6  # use micro Kelvin
    H = W = 80
    Lx = np.radians(20 * resol / 60)
    Ly = np.radians(20 * resol / 60)
    # l0_bins = np.arange(W / 8) * 8 * np.pi / Lx
    # lf_bins = (np.arange(W / 8) + 1) * 8 * np.pi / Lx
    # l_min = Lx
    # l_max = 1 / ((W / Lx) * 2)
    l0_bins = np.arange(300, 2000, 80)   # original bounds 20, 1000
    lf_bins = np.arange(300, 2000, 80) + 79
    # l0_bins = np.arange(40, 720, 50)   # original bounds 20, 1000
    # lf_bins = np.arange(40, 720, 50) + 49
    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    ells_uncoupled = b.get_effective_ells()
    print(ells_uncoupled[0], ells_uncoupled[-1])
    # mask stuff
    mask = make_apo_mask(H, W, Lx, Ly, resol)
    Q_normed, U_normed = imgs[0, 0], imgs[0, 1]
    Q_gt_normed, U_gt_normed = imgs[0, 2], imgs[0, 3]

    # https://namaster.readthedocs.io/en/latest/api/pymaster.field.html#pymaster.field.NmtFieldFlat
    f_NN = nmt.NmtFieldFlat(Lx, Ly, mask, [Q_normed, U_normed], purify_b=True)
    f_GT = nmt.NmtFieldFlat(Lx, Ly, mask, [Q_gt_normed, U_gt_normed], purify_b=True)
    w22 = nmt.NmtWorkspaceFlat()
    wgt = nmt.NmtWorkspaceFlat()
    w22.compute_coupling_matrix(f_NN, f_NN, b)
    wgt.compute_coupling_matrix(f_GT, f_GT, b)
    # https://namaster.readthedocs.io/en/latest/api/pymaster.workspaces.html#pymaster.workspaces.compute_coupled_cell_flat
    cl_NN_coupled = nmt.compute_coupled_cell_flat(f_NN, f_NN, b)
    cl_GT_coupled = nmt.compute_coupled_cell_flat(f_GT, f_GT, b)
    # https://namaster.readthedocs.io/en/latest/api/pymaster.workspaces.html#pymaster.workspaces.NmtWorkspace.decouple_cell
    # Problem is cl_NN_coupled is singular
    cl_NN_uncoupled = w22.decouple_cell(cl_NN_coupled)
    cl_GT_uncoupled = wgt.decouple_cell(cl_GT_coupled)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    names = ['EE', 'BB']
    ps_nn = [cl_NN_uncoupled[0], cl_NN_uncoupled[3]]
    ps_gt = [cl_GT_uncoupled[0], cl_GT_uncoupled[3]]
    for i in range(2):
        # axes[i].loglog(ells_uncoupled, cls_all[0][i*3],  '--', lw=2, color='Black', alpha=0.5, label = 'GNILC 80 amin')
        # axes[i].loglog(ells_uncoupled, cls_all[1][i*3], '-', label='GNILC+Gauss 12 amin', lw=4, color='#569A62', alpha=0.7)
        # TODO verify masking is legit
        # mask = ps_gt[i] > 1e-18
        mask = np.ones_like(ells_uncoupled).astype(bool)  # equivalent to no mask
        axes[i].loglog(ells_uncoupled[mask], ps_nn[i][mask], '-', label='Model Output', lw=1,
                       color='#F87217', alpha=0.7)
        axes[i].loglog(ells_uncoupled[mask], ps_gt[i][mask], '-', label='Ground Truth', lw=1,
                       color='#569A62', alpha=0.7)
        # axes[i].set_ylim(1e-6, 2e-1)
        # axes[i].set_xlim(500, 2000)
        # axes[i].set_xticks([40, 100, 400, 1000])
        axes[i].set_title(f'{names[i]}', fontsize=18)
        axes[i].set_xlabel(r'Multipole $\ell$', fontsize=18)
        axes[i].set_ylabel(r'$C_\ell$ [$\mu K^2$]', fontsize=18)
    # plt.xticks([40, 100, 400, 1000])
    axes[0].legend(fontsize=15)

    return fig
