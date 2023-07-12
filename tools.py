import numpy as np
import pandas as pd
import modalno_kladivo_analiza as mka
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv
import math
import matplotlib.gridspec as gridspec
from IPython.display import clear_output


def get_Y_J(m, c, k, fr, unit='Hz', frf_type='receptance', return_eigvals=False, rcond=1e-15, print_=True):
    """
    Calculate admitance matrix of the joint.
    """
    mass_matrix = np.diag(2 * list(m))
    damping_matrix = np.block([[np.diag(c), -np.diag(c)],
                               [-np.diag(c), np.diag(c)]])
    stiffness_matrix = np.block([[np.diag(k), -np.diag(k)],
                                 [-np.diag(k), np.diag(k)]])

    return get_Y_joint(mass_matrix, damping_matrix, stiffness_matrix, fr, unit=unit, return_eigvals=return_eigvals,
                       rcond=rcond, print_=print_, frf_type=frf_type)


def get_Y_joint(mass_matrix, damping_matrix, stiffness_matrix, fr, frf_type='receptance', unit='Hz',
                return_eigvals=False, rcond=1e-15, print_=True):
    """
    Function calculates joint FRF.

    Parameters
    ----------
    mass_matrix: joint mass matrix
    damping_matrix: joint damping matrix
    stiffness_matrix: joint stiffness matrix
    fr: np.array with frequencies
    frf_type: 'receprance', 'mobility' or 'accelerance'
    unit: 'Hz' or 'rad/s'
    return_eigvals: (bool)
    rcond: rcond for Moore-Penrose inverse
    print_: type of matrix in verse is printed (bool)

    Returns: Joint Y matrix
    -------

    """
    if unit == 'rad':
        omega = fr
    elif unit == 'Hz':
        omega = 2 * np.pi * fr
    else:
        print('invalid unit')
        return

    try:
        Y_J = np.linalg.inv(np.einsum('ij,k->kij', mass_matrix, -1 * (omega ** 2)) +
                            np.einsum('ij,k->kij', damping_matrix, 1j * omega) + stiffness_matrix)
        if print_:
            print('Regular matrix inverse')
    except np.linalg.LinAlgError:
        if print_:
            print('Moore-Penrose inverse')
        Y_J = np.linalg.pinv(np.einsum('ij,k->kij', mass_matrix, -1 * (omega ** 2)) +
                             np.einsum('ij,k->kij', damping_matrix, 1j * omega) + stiffness_matrix, rcond=rcond)

    if frf_type == 'mobility':
        Y_J *= 1.j * omega
    elif frf_type == 'accelerance':
        Y_J = np.einsum('ijk,i->ijk', Y_J, -1*omega**2)

    if return_eigvals:
        eigvals, _ = np.linalg.eig(np.linalg.inv(mass_matrix) @ stiffness_matrix)
        return eigvals, Y_J
    else:
        return Y_J


def inv_by_SVD(G):
    """
    Aproksimacija inverza matrike z uporabo SVD metode (Linear Algebra: A modern introduction p. 625)
    :param G:
    :param n_kept:
    :return:
    """
    u, sigma, vh = np.linalg.svd(G, full_matrices=True)
    sigma_inv = np.zeros_like(G, dtype=float)
    g_inv = np.zeros_like(G)
    for i in range(sigma.shape[0]):
        sigma_inv[i, :, :] = np.diag(1/sigma[i, :])
        g_inv[i] = np.matmul(np.conj(vh[i]).T, (sigma_inv[i] @ np.conj(u[i]).T))
    return g_inv


def get_combinations(x, y, z):
    """
    Function returns list of all combinations for inserted x, y and z values.
    x - list of x values (list)
    y - list of y values (list)
    z - list of z values (list)
    """
    combinations = np.zeros((len(x)*len(y)*len(z), 3))
    count = 0
    for i in x:
        for j in y:
            for k in z:
                combinations[count, :] = np.array((i, j, k))
                count += 1
    return combinations


def B_matrix(n_A, n_B, int_A, int_B, n_J=12):
    """
    Function generates a Boolean matrix for substructurig purposes.
    params:
    n_A: number of DoFs of substructure A (int)
    n_B: number of DoFs of substructure B (int)
    n_J: number of DoFs of the joint J (int); half the DoFs belong to the substructure A and half to the substructure B
    int_A: interface DoFs indices for substructure A (int)
    int_B: interface DoFs indices for substructure B (int)
    """
    B = np.zeros((n_J, n_A + n_J + n_B))
    B[:int(n_J/2), int_A] = -1 * np.eye(int(n_J/2))
    B[:int(n_J/2), n_A + np.arange(int(n_J/2))] = np.eye(int(n_J/2))
    B[int(n_J/2):, n_A + int(n_J/2) + np.arange(int(n_J/2))] = -1 * np.eye(int(n_J/2))
    B[int(n_J/2):, n_A + n_J + int_B] = np.eye(int(n_J/2))
    return B


def get_dfs(MK, xlsx_file, VP=False, interface=None):
    """
    Import padas DataFrames from the xlsx files and update locations.
    """
    df_acc = pd.read_excel(xlsx_file, sheet_name='Sensors')
    df_chn = pd.read_excel(xlsx_file, sheet_name='Channels')
    df_imp = pd.read_excel(xlsx_file, sheet_name='Impacts')
    df_chn_up = MK.update_locations_df(df_chn)
    df_imp_up = MK.update_locations_df(df_imp)
    if VP:
        df_chn_VP = pd.read_excel(xlsx_file, sheet_name='VP_Channels')
        df_imp_VP = pd.read_excel(xlsx_file, sheet_name='VP_RefChannels')
        df_chn_VP_up = MK.update_locations_df(df_chn_VP)
        df_imp_VP_up = MK.update_locations_df(df_imp_VP)
        # edit groupings
        params = {'interface': interface}
        df_chn_up.Grouping = df_chn_up.apply(edit_grouping, axis=1, **params)
        df_imp_up.Grouping = df_imp_up.apply(edit_grouping, axis=1, **params)
        df_chn_VP_up.Grouping = 100
        df_imp_VP_up.Grouping = 100
        # TODO: Prilagoditev za več virtualnih točk

        return df_acc, df_chn_up, df_imp_up, df_chn_VP_up, df_imp_VP_up
    return df_acc, df_chn_up, df_imp_up


def edit_grouping(x, interface):
    """z
    Prepare Groupings for VPT.
    """
    if np.equal(x[['Position_1', 'Position_2', 'Position_3']].astype(float).values, interface).all(axis=1).any():
        return 100
    else:
        return 1


def get_index(point, df):
    """Fuction returns the location (index) of given point in given dataframe"""
    return np.where((point[0] == df['Position_1']) & (point[1] == df['Position_2']) & (point[2] == df['Position_3']))[0]


def compare_FRF(acc_point, imp_point, Y1, Y2, df_chn1, df_chn2, df_imp1, df_imp2):
    """
    function returns FRF from given impact point (imp_point) to the given response point (acc_point)
    """
    chn1_ind = get_index(acc_point, df_chn1)
    chn2_ind = get_index(acc_point, df_chn2)
    imp1_ind = get_index(imp_point, df_imp1)
    imp2_ind = get_index(imp_point, df_imp2)
    return Y1[:, chn1_ind, :][:, :, imp1_ind], Y2[:, chn2_ind, :][:, :, imp2_ind]


def T(matrix, axes):
    """
    Multidimensional matrix transpose (short for np.transpose)
    matrix: np.array to be transposed (np.array)
    axes: tuple with order of matrix dimensions (tuple)

    return transposed matrix (np.array)
    """
    return matrix.transpose(axes)


def sigmoid(x, shift_f=0, length_f=1):
    """
    Function returns sigmoid function for given x values shifted and stretched for values given by parameters shift_f
    and length.
    shift_f: transition frequncy shift
    length_f: lenth of transition interval - 0-1 -> smaller value means longer transition
    """
    return 1/(1+np.e**(length_f*(-(x-shift_f))))


def rev_sigmoid(x, shift_f=0, length_f=1):
    """
    Sigmoid function going from 1 to 0.
    """
    return 1/(1 + np.e ** (length_f * (x - shift_f)))


def step(x, x_step):
    """Function returns step function for given x values with given location of step"""
    return np.where(x < x_step, 0, 1)


def match_coordinates(points, mesh, atol):
    """
    Returns 'points' which match with points in 'mesh'.
    :param points: np.array with point coordinates (shape: n×3)
    :param mesh: rst.mesh.nodes
    :param atol: absolute tolerance when comparing point and node coordinates
    return: np.array of mesh elements coresponding to given points
    """
    # TODO: Matching according to distance; alternative to matching according to coordinates
    mask = np.zeros(mesh.shape[0])
    order = []
    for i in points:
        try:
            order.append(np.where(np.isclose(mesh[:, 0], i[0], atol=atol) &
                                  np.isclose(mesh[:, 1], i[1], atol=atol) &
                                  np.isclose(mesh[:, 2], i[2], atol=atol))[0][0])
        except IndexError:
            pass
        mask += np.isclose(mesh[:, 0], i[0], atol=atol) &\
                np.isclose(mesh[:, 1], i[1], atol=atol) &\
                np.isclose(mesh[:, 2], i[2], atol=atol)
    if len(order) == points.shape[0]:
        print(f'OK. {len(order)}/{points.shape[0]} points matched to mesh')
    else:
        print(f'Points missing. {len(order)}/{points.shape[0]} points matched to mesh')
    if len(order) == 0:
        print('No points matched to mesh')
        return None
    return mesh[np.array(order)]


# Transformacija med dvema ortogonalnima koordinatnima sistemoma
# primer slovarja za transformacijo: transformation_dict = {'x':'x', 'y':'-z', 'z':'y'}


def check_signs(x1, x2):
    if (x1[0] == '-') and (x2[0] == '-'):
        return ''
    elif (x1[0] == '-') or (x2[0] == '-'):
        return '-'
    else:
        return ''


def transform_system(arr, transformation_dict):
    arr_transformed = []
    for i in arr:
        tr_val = transformation_dict[i[-1]]
        arr_transformed.append(check_signs(i, tr_val) + tr_val[-1])
    return arr_transformed


# Primerjava FRF
def plot_ampl_phase(Y, i, j, labels, x_lim, title=False, ls=None, figsize=(15, 5)):
    # %matplotlib inline
    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    # fig.suptitle('Primerjava: $\mathbf{Y_{num}}$ vs. $\mathbf{Y_{exp}}$')
    if title:
        fig.suptitle(f"Acc: {i}, imp: {j}")
    for ind, (Y_, lab_) in enumerate(zip(Y, labels)):
        if ls is not None:
            ls_ = ls[ind]
        else:
            ls_ = '-'

        ax[0].semilogy(abs(Y_[:x_lim, i, j]), label=lab_, ls=ls_)
    ax[0].grid()
    ax[0].set_xlabel('f [hz]')
    ax[0].set_ylabel('A [m/(s$^2$N)]')
    ax[0].legend(loc=(1.01, 0))
    for ind, (Y_, lab_) in enumerate(zip(Y, labels)):
        if ls is not None:
            ls_ = ls[ind]
        else:
            ls_ = '-'

        ax[1].plot(np.angle(Y_[:x_lim, i, j]), ls=ls_)
    ax[1].grid()
    ax[1].set_xlabel('f [hz]')
    ax[1].set_ylabel(r'$\varphi$ [rad]')
    ax[1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax[1].set_yticklabels([r'-$\pi$', r'-$\pi$/2', '0', r'$\pi$/2', r'$\pi$'])

    # plt.xlim(1700,2200)
    # plt.ylim(.5*1e1, 1.1*1e3)


def plot_origin(plotter, scale, loc=None):
    if loc is None:
        loc = [0, 0, 0]
    x_arrow = pv.Arrow((loc[0], loc[1], loc[2]), (1, 0, 0), scale=scale)
    y_arrow = pv.Arrow((loc[0], loc[1], loc[2]), (0, 1, 0), scale=scale)
    z_arrow = pv.Arrow((loc[0], loc[1], loc[2]), (0, 0, 1), scale=scale)
    plotter.add_mesh(x_arrow, color='#87100c')  # dark red
    plotter.add_mesh(y_arrow, color='#167509')  # dark green
    plotter.add_mesh(z_arrow, color='#06064f')  # dark blue


def plt_font_setup(family='serif', font='Computer Modern Roman', fontsize=10):
    """
    Function sets font properties for matplotlib plot
    Parameters
    ----------
    family
    font
    fontsize

    Returns
    -------

    """
    plt.rcParams['font.family'] = family
    plt.rcParams['font.sans-serif'] = font
    plt.rcParams['font.size'] = fontsize


def coh_plot(coh_matrix, vmin=0, vmax=1, figsize=(5, 6), display_averages=False):
    width = 4
    height = math.ceil(coh_matrix.shape[0]/coh_matrix.shape[1]*width)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f'Average coherence: {np.sum(coh_matrix)/(np.multiply(*np.shape(coh_matrix))):.4f}')
    gs = fig.add_gridspec(height+2, 7)
    # coh matrix
    coh_matrix_ax = fig.add_subplot(gs[:height, :width])
    im_coh_matrix = coh_matrix_ax.imshow(coh_matrix, vmin=vmin, vmax=vmax)
    # avg rows
    coh_chn_avg_ax = fig.add_subplot(gs[:height, width:width+2], sharey=coh_matrix_ax)
    coh_chn_avg_ax.set_xticks([])
    row_averages = np.sum(coh_matrix, axis=1)[:, np.newaxis]/coh_matrix.shape[1]
    coh_chn_avg_ax.imshow(row_averages, vmin=vmin, vmax=vmax)

    if display_averages:
        count = 0
        for (j, i), label in np.ndenumerate(row_averages):
            textcolor = "black"
            if count % 2 == 0:
                plt.text(i+2.5, j, f"{label:.2f}", ha='center', va='center', color=textcolor, fontsize=12)
            if count % 2 != 0:
                plt.text(i+7, j, f"{label:.2f}", ha='center', va='center', color=textcolor, fontsize=12)
            count += 1

    # avg columns
    coh_imp_avg_ax = fig.add_subplot(gs[height:height+1, 0:width], sharex=coh_matrix_ax)
    coh_imp_avg_ax.set_yticks([])
    col_averages = np.sum(coh_matrix, axis=0)[:, np.newaxis]/coh_matrix.shape[0]
    coh_imp_avg_ax.imshow(col_averages.T, vmin=vmin, vmax=vmax)

    if display_averages:
        count = 0
        for (i, j), label in np.ndenumerate(col_averages):
            textcolor = "black"
            if count % 2 == 0:
                plt.text(i, j+5, f"{label:.2f}", ha='center', va='center', color=textcolor, fontsize=12, rotation=90)
            if count % 2 != 0:
                plt.text(i, j+9, f"{label:.2f}", ha='center', va='center', color=textcolor, fontsize=12, rotation=90)
            count += 1
    # colorbar
    colorbar_ax = fig.add_subplot(gs[:height, width+2:width+3])
    fig.colorbar(im_coh_matrix, cax=colorbar_ax, orientation='vertical', fraction=.3)



def FRF_reconstruction(ω, ω_r, damping_r, θ, damping_model, dp_loc=0):
    """
    Modal superposition based FRF reconstruction. Enables reconstruction of single admitance matrix row or column
    :param ω: numeric array of frquencies
    :param ω_r: numeric array of natural frequencies (rad/s)
    :param damping_r: numeric array of damping factors (η in hysteretic damping model, ζ in viscous damping model)
    :param damping_model: hyst (hysteretic) or visc (viscous)
    :param dp_loc: driving point DoF index
    """
    Y_rec = np.zeros((ω.shape[0], 1, θ.shape[0]), dtype=complex)

    # hysteretic damping model
    if damping_model == 'hyst':
        for i in range(len(ω_r)):
            Y_rec[:, 0, :] += np.einsum('i,j->ij', 1 / (ω_r[i] ** 2 - ω ** 2 + 1.j * damping_r[i] * ω_r[i] ** 2),
                                        (θ[dp_loc, i] * θ[:, i]))

    # viscous damping model
    elif damping_model == 'visc':
        for i in range(len(ω_r)):
            clen1 = np.einsum('i,j->ij',
                              1 / (ω_r[i] * damping_r[i] + 1.j * (ω - ω_r[i] * np.sqrt(1 - damping_r[i] ** 2))),
                              (θ[:, i] * θ[dp_loc, i]))
            clen2 = np.einsum('i,j->ij',
                              1 / (ω_r[i] * damping_r[i] + 1.j * (ω + ω_r[i] * np.sqrt(1 - damping_r[i] ** 2))),
                              np.conj(θ[:, i] * θ[dp_loc, i]))
            Y_rec[:, 0, :] += clen1 + clen2

    # invalid damping model
    else:
        raise Exception(
            "Invalid damping model. Enter 'hyst' for hysteretic damping model or 'visc' for viscous damping model.")
    return Y_rec


def FRF_ij_reconstruction(ω, ω_r, damping_r, θ_i,  θ_j, damping_model):
    """
    Modal superposition based FRF reconstruction. Enables reconstruction of single admitance matrix row or column
    :param ω: numeric array of frquencies
    :param ω_r: numeric array of natural frequencies (rad/s)
    :param damping_r: numeric array of damping factors (η in hysteretic damping model, ζ in viscous damping model)
    :param damping_model: hyst (hysteretic) or visc (viscous)
    :param dp_loc: driving point DoF index
    """
    Y_rec = np.zeros((ω.shape[0]), dtype=complex)

    # hysteretic damping model
    if damping_model == 'hyst':
        for i in range(len(ω_r)):
            Y_rec += np.einsum('i,j->ij', 1 / (ω_r[i] ** 2 - ω ** 2 + 1.j * damping_r[i] * ω_r[i] ** 2),
                               (θ_i[i] * θ_j[i]))

    # viscous damping model
    elif damping_model == 'visc':
        for i in range(len(ω_r)):
            clen1 = (θ_i[i] * θ_j[i]) / (ω_r[i] * damping_r[i] + 1.j * (ω - ω_r[i] * np.sqrt(1 - damping_r[i] ** 2)))
            clen2 = np.conj(θ_i[i] * θ_j[i]) / (ω_r[i] * damping_r[i] + 1.j * (ω + ω_r[i] * np.sqrt(1 - damping_r[i] ** 2)))
            Y_rec += clen1 + clen2

    # invalid damping model
    else:
        raise Exception(
            "Invalid damping model. Enter 'hyst' for hysteretic damping model or 'visc' for viscous damping model.")
    return Y_rec


# PCA
class PCA:
    def __init__(self, H, p=None):
        self.H = H
        self.p = p
        self.col_avg, self.col_std, self.H_adj, self.eigvals, self.eigvecs, self.Cmatch = self.get_reduction_matrix()

    def get_reduction_matrix(self):
        col_avg = self.H.mean(axis=0)  # average column value
        print('Average over columns - done')
        col_std = np.sum((self.H - col_avg)**2, axis=0)/self.H.shape[0]  # standard deviation for over columns
        print('Standard deviation over columns - done')
        H_adj = (self.H - col_avg)/np.sqrt(col_std*self.H.shape[0])  # H matrix adjustment
        print('Adjusted H matrix - done')
        C = np.conj(H_adj.T) @ H_adj  # Correlation matrix
        print('Correlation matrix - done')
        eigvals, eigvecs = np.linalg.eig(C)  # eignevalue problem solution
        print('Eigenproblem - done')
        clear_output()
        return col_avg, col_std, H_adj, eigvals, eigvecs, C

    def get_projection_matrix(self, H_=None, p=None):
        if p is not None:
            self.p = p
        if H_ is None:
            H_adj_ = self.H_adj
        else:
            H_adj_ = (H_ - self.col_avg) / np.sqrt(self.col_std * self.H.shape[0])
        return np.einsum('ki,ij->kj', H_adj_, self.eigvecs[:, :self.p])

    def reconstruct(self, H_=None):
        if H_ is None:
            H_ = self.H
        H_adj_R = np.einsum('ij,kj->ik', self.get_projection_matrix(H_), self.eigvecs[:, :self.p])
        return H_adj_R * (np.sqrt(self.col_std*self.H.shape[0])) + self.col_avg

    def plot_results(self, H_, plot_PC_scores, plot_reconstruction, no_scores, i):
        """TODO: bolj posplošiti - trenutno samo za izhodiščni H"""
        if plot_reconstruction and plot_PC_scores:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax_1, ax_2 = ax[0], ax[1]
        elif plot_reconstruction:
            fig, ax_2 = plt.subplots(1, 1)
        elif plot_PC_scores:
            fig, ax_1 = plt.subplots(1, 1)

        if plot_PC_scores:
            ax_1.semilogy(abs(self.get_PCA_scores(H_)[:, :no_scores]))
            ax_1.set_title('PCA scores')

        if plot_reconstruction:
            ax_2.semilogy(abs(self.reconstruct())[:, i], label='reconstruction', c='k', lw=3)
            ax_2.semilogy(abs(self.H)[:, i], '--', label='original', c='y');
            ax_2.legend()
            ax_2.grid();


def PCA_old(H, p, reconstruct=False, compare=False, i=None, show_scores=False, no_scores=None):
    """
    Principal component analysis implementation. H must be of shape: n×m, where n (rows) is a
    number of frequency points and m (columns) is a number of channels.
    H: FRF matrix
    p: numer of principal components kept after reduction
    reconstruct: reconstruction of FRF after dim. reduction
    compare: plot graph with comparison of original vs reconstructed FRF
    i: index of plotted FRF in comparison graph
    show_scores: plot pca results (scores)
    no_scores: number of scores plotted (must be less than p)
    """
    m, n = H.shape[0], H.shape[1]

    row_avg = H.mean(axis=1)  # average row value
    H_adj = (H - row_avg[:, np.newaxis])  # H matrix adjustment
    eigvals, eigvecs = np.linalg.eig(np.conjugate(H_adj.T) @ H_adj)  # eignevalue problem solution

    PCA_scores = H_adj @ eigvecs[:, :p]  # PCA

    if compare and show_scores:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax_1, ax_2 = ax[0], ax[1]
    elif compare:
        fig, ax_2 = plt.subplots(1, 1)
    elif show_scores:
        fig, ax_1 = plt.subplots(1, 1)
    if show_scores:
        ax_1.semilogy(abs(PCA_scores[:, :no_scores]))
        ax_1.set_title('PCA scores')

    if reconstruct:  # reconstruction
        PCA_rec = PCA_scores @ np.conjugate(eigvecs[:, :p].T) + row_avg[:, np.newaxis]

        if compare:
            ax_2.semilogy(abs(PCA_rec)[:, i], label='reconstruction', c='k', lw=3)
            ax_2.semilogy(abs(H)[:, i], '--', label='original', c='y');
            ax_2.legend()
            ax_2.grid();
        return PCA_scores, PCA_rec
    return PCA_scores