from typing import Union

import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import pymaster as nmt
import torch

PATCHSIZE_MULTIPLE = 20  # ie patch side length is 20 * resol


# TODO use Jay's package
@jit(nopython=True)
def estimate_marchingsquare_fix(data, threshold):
    """ Copied from ForSE+ Repo. """
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
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a3 = (data[i, j + 1] - threshold) / (data[i, j + 1] - data[i + 1, j + 1])
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
        f[k], u[k], chi[k] = estimate_marchingsquare_fix(im[k[0], k[1]], rho)
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


def plot_MF(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray, None] = None,
    plot_gauss: bool = False,
    resol: int = 15
):
    """ Plots Minkowski Functionals for the given images.

    Args:
        preds (torch.Tensor | np.ndarray): A batch of images of shape (batch, 2, H, W), where the
            channel dimension is the predicted Q and U in that order.
        targets (torch.Tensor | np.ndarray): A batch of images of shape (batch, 2, H, W), where the
            channel dimension is the ground truth Q and U in that order (if any).
        plot_gauss (bool): Whether to plot the gaussian case as well.
        resol (int): The resolution in arcminutes to use for the Gaussian MFs (used only if
            plot_gauss is True).
    """
    preds = _tensor_to_np(preds)
    mT = rescale_min_max(preds)
    rhos, f, u, chi = get_functionals_fix(mT)
    y = [f, u, chi]
    if targets is not None:
        targets = _tensor_to_np(targets)
        mT_gt = rescale_min_max(targets)
        rhos_gt, f_gt, u_gt, chi_gt = get_functionals_fix(mT_gt)
        gt_y = [f_gt, u_gt, chi_gt]
    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
    S = ['Q', 'U']
    if plot_gauss:
        # (n, 80, 80)
        gauss = get_gauss_QU(resol, preds.shape[0])
        mT = rescale_min_max(gauss)
        g_rhos, g_f, g_u, g_chi = get_functionals_fix(mT)
        g_y = [g_f, g_u, g_chi]
    for i in range(2):
        for j in range(3):
            # label = r'$\tilde{m}^{%s, 20^{\circ}}_{12^\prime}$' % S[j] if j == 0 else None
            # label_t = r'$\tilde{m}^{I, 20^{\circ}}_{12^\prime}$' if j == 0 else None
            label = f'Prediction {S[i]}' if j == 0 else None
            label_t = f'Ground Truth {S[i]}' if j == 0 else None
            axes[i, j].ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
            mean = np.mean(y[j][:, i, ...], axis=0)
            std = np.std(y[j][:, i, ...], axis=0)
            # all rhos are the same
            axes[i, j].fill_between(
                rhos[0, 0], mean - std, mean + std, lw=1, alpha=0.5, color='#F87217', label=label
            )
            axes[i, j].plot(rhos[0, 0], mean, lw=3, ls='--', color='#D04A00')
            if targets is not None:
                mean_gt = np.mean(gt_y[j][:, 2 + i, ...], axis=0)
                std_gt = np.std(gt_y[j][:, 2 + i, ...], axis=0)
                axes[i, j].fill_between(
                    rhos[0, 0], mean_gt - std_gt, mean_gt + std_gt, lw=2, label=label_t,
                    edgecolor='black', facecolor='None'
                )
                axes[i, j].plot(rhos[0, 0], mean_gt, lw=2, ls='--', color='black')
            if plot_gauss:
                mean_g = np.mean(g_y[j][:, i, ...], axis=0)
                std_g = np.std(g_y[j][:, i, ...], axis=0)
                axes[i, j].fill_between(
                    g_rhos[0, 0], mean_g - std_g, mean_g + std_g, lw=1, alpha=0.5, color='#569A62',
                    label='Gaussian'
                )
                axes[i, j].plot(g_rhos[0, 0], mean_g, lw=3, ls='--', color='#246830')
            axes[i, j].set_ylabel(r'$\mathcal{V}_{%s}(\rho$) %s' % (j, S[i]), fontsize=25)
            if i == 1:
                axes[i, j].set_xlabel(r'$\rho$', fontsize=25)
            if j == 0:
                axes[i, j].legend()
    plt.tight_layout()
    return fig


def get_patch_size_deg(res: float) -> float:
    return PATCHSIZE_MULTIPLE * res / 60


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
    return imgs


def plot_power_spectra_patch(
    resol: float,
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray, None] = None,
):
    """ Plots the power spectra for the given images.

    Args:
        resol (float): The image resolution in arcminutes.
        preds (torch.Tensor | np.ndarray): An image of shape (2, H, W), where the channel dimension
            is the predicted Q and U in that order.
        targets (torch.Tensor | np.ndarray): An image of shape (2, H, W), where the channel
            dimension is the ground truth Q and U in that order (if any).
    """
    preds = _tensor_to_np(preds) * 1e6  # use micro Kelvin
    H = W = preds.shape[-1]
    Lx = np.radians(PATCHSIZE_MULTIPLE * resol / 60)
    Ly = np.radians(PATCHSIZE_MULTIPLE * resol / 60)
    # l0_bins = np.arange(W / 8) * 8 * np.pi / Lx
    # lf_bins = (np.arange(W / 8) + 1) * 8 * np.pi / Lx
    # l_min = Lx
    # l_max = 1 / ((W / Lx) * 2)
    l0_bins = np.arange(300, 2000, 80)
    lf_bins = np.arange(300, 2000, 80) + 79
    # l0_bins = np.arange(40, 720, 50)
    # lf_bins = np.arange(40, 720, 50) + 49
    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    ells_uncoupled = b.get_effective_ells()
    # mask stuff
    mask = make_apo_mask(H, W, Lx, Ly, resol)
    # consider renaming
    Q_normed, U_normed = preds[0], preds[1]
    print(mask.shape, preds[0].shape)

    # https://namaster.readthedocs.io/en/latest/api/pymaster.field.html#pymaster.field.NmtFieldFlat
    f_NN = nmt.NmtFieldFlat(Lx, Ly, mask, [Q_normed, U_normed], purify_b=True)
    w22 = nmt.NmtWorkspaceFlat()
    w22.compute_coupling_matrix(f_NN, f_NN, b)
    # https://namaster.readthedocs.io/en/latest/api/pymaster.workspaces.html#pymaster.workspaces.compute_coupled_cell_flat
    cl_NN_coupled = nmt.compute_coupled_cell_flat(f_NN, f_NN, b)
    # https://namaster.readthedocs.io/en/latest/api/pymaster.workspaces.html#pymaster.workspaces.NmtWorkspace.decouple_cell
    # Problem is cl_NN_coupled is singular
    cl_NN_uncoupled = w22.decouple_cell(cl_NN_coupled)
    ps_nn = [cl_NN_uncoupled[0], cl_NN_uncoupled[3]]

    if targets is not None:
        Q_gt_normed, U_gt_normed = targets[0], targets[1]
        f_GT = nmt.NmtFieldFlat(Lx, Ly, mask, [Q_gt_normed, U_gt_normed], purify_b=True)
        wgt = nmt.NmtWorkspaceFlat()
        wgt.compute_coupling_matrix(f_GT, f_GT, b)
        cl_GT_coupled = nmt.compute_coupled_cell_flat(f_GT, f_GT, b)
        cl_GT_uncoupled = wgt.decouple_cell(cl_GT_coupled)
        ps_gt = [cl_GT_uncoupled[0], cl_GT_uncoupled[3]]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    names = ['EE', 'BB']
    for i in range(2):
        mask = np.ones_like(ells_uncoupled).astype(bool)  # equivalent to no mask
        axes[i].loglog(ells_uncoupled[mask], ps_nn[i][mask], '-', label='Model Output', lw=1,
                       color='#F87217', alpha=0.7)
        if targets is not None:
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


def plot_patch(inps, targets):
    inps = inps.squeeze()
    targets = targets.squeeze()
    fig, axes = plt.subplots(1, 7, figsize=(20, 4))
    titles = ['TAU', 'Q_HI', 'U_HI', 'Q_LR', 'U_LR', 'Q_HR', 'U_HR']

    # Loop through each image and plot it on the respective axis
    for i, ax in enumerate(axes):
        if i < 5:
            img_plot = ax.imshow(inps[i])
        else:
            img_plot = ax.imshow(targets[i - 5])
        ax.set_title(titles[i])
        plt.colorbar(img_plot, ax=ax)  # Attach colorbar to each image

    plt.tight_layout()
    plt.show()


def plot_inf_patch(inps):
    inps = inps.squeeze()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ['TAU', 'Q_HI', 'U_HI', 'Q_LR', 'U_LR']

    # Loop through each image and plot it on the respective axis
    for i, ax in enumerate(axes):
        img_plot = ax.imshow(inps[i])
        ax.set_title(titles[i])
        plt.colorbar(img_plot, ax=ax)  # Attach colorbar to each image

    plt.tight_layout()
    plt.show()


def plot_centers(centers):
    plt.figure(figsize=(21, 7))

    for res in centers.keys():
        lat = centers[res]['lat']
        lon = centers[res]['lon']
        plt.scatter(lon, lat, s=1, label=fr"{int(res)}$'$ ({len(lat)} projections)", lw=5)
    plt.xlabel('$l$ [$^\circ$]')
    plt.ylabel('$b$ [$^\circ$]')
    plt.xticks(np.arange(0, 360, 15), np.arange(0, 360, 15))
    # for tick in plt.gca().xaxis.get_major_ticks():
    #     tick.label.set_rotation(90)
    plt.yticks(np.arange(-90, 90, 10), np.arange(-90, 90, 10))
    plt.legend()
    plt.show()
