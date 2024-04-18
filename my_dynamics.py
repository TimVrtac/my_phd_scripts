import numpy as np
import pyFBS
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tools import H, T
import timeit
from scipy import signal
import math
from numba import njit


def LMFBS(Y, Bc, Be, TSVD=False, reduction=0):
    """
    Implementation of th LM-FBS equation (Lagrange Multiplier Freqency Based Substructuring).
    Args:
        Y: admitance matrix of the uncoupled system
        Bc: boolean matrix connecting the corresponding interface channel DoFs
        Be: boolean matrix connecting the corresponding interface impact DoFs
        TSVD: if True the truncated singular value decomposition is performed on Y_int matrix
        reduction: number of singular values not to be taken into account in TSVD

    Returns: coupled admitance matrix

    """
    Y_int = Bc @ Y @ Be.T
    if TSVD:
        Y_int = pyFBS.TSVD(Y_int, reduction=reduction)
    return Y - Y @ Be.T @ np.linalg.pinv(Y_int) @ Bc @ Y


def decouple(Y_AB, Y_A, SVT=False, r=None, reduction_basis='A', TSVD=False, reduction=0, print_=True):
    """
    Function performs decoupling of substructure A from structure AB using LM-FBS method.
    :param Y_AB: admitance matrix of structure Y_AB. Must be constructed in such manner that the interface DoFs are
    placed after internal DoFs.
    |Y_ii Y_ib|
    |Y_bi Y_bb|
    b - boundary (interface DoFs)
    :param SVT: if True Singular Vector Transformation (SVT) is performed
    :param r: size of SVT reduction bases
    :param reduction_basis: 'A' or 'AB' (str)
    :param Y_A: admitance matrix of substructure Y_A (Consists only of the interface DoFs).ž
    :param TSVD: if True the truncated singular value decomposition is performed on Y_int matrix
    :param reduction: number of singular values not to be taken into account in TSVD
    :param print_: if True the progress of the decoupling is printed
    :return: decoupled admitance Y_B
    """
    if SVT:
        Y_AB, Y_A = SVT_4_decoupling(Y_AB, Y_A, r=r, reduction_basis=reduction_basis)
        if print_:
            print('SVT process finished!')

    # construction of the uncoupled admitance matrix
    Y_AB_B = np.zeros((Y_AB.shape[0], Y_AB.shape[1]+Y_A.shape[1], Y_AB.shape[2]+Y_A.shape[2]), dtype=complex)
    Y_AB_B[:, :Y_AB.shape[1], :Y_AB.shape[2]] = Y_AB
    Y_AB_B[:, Y_AB.shape[1]:, Y_AB.shape[2]:] = -1*Y_A

    # construction of the signed Boolean matrices
    B_c = np.zeros((Y_A.shape[2], Y_AB_B.shape[1]))
    B_e = np.zeros((Y_A.shape[2], Y_AB_B.shape[2]))
    B_c[:, Y_AB_B.shape[1]-2*Y_A.shape[2]:Y_AB_B.shape[1]-Y_A.shape[2]] = np.eye(Y_A.shape[2])
    B_c[:, Y_AB_B.shape[1]-Y_A.shape[2]:] = -1*np.eye(Y_A.shape[2])
    B_e[:, Y_AB_B.shape[2]-2*Y_A.shape[2]:Y_AB_B.shape[2]-Y_A.shape[2]] = np.eye(Y_A.shape[2])
    B_e[:, Y_AB_B.shape[2]-Y_A.shape[2]:] = -1*np.eye(Y_A.shape[2])

    # LM-FBS decoupling
    Y_B = LMFBS(Y_AB_B, B_c, B_e, TSVD=TSVD, reduction=reduction)
    if print_:
        print('Decoupling process finished!')

    return Y_B, B_c, B_e


def couple(Y_AB, Y_A, SVT=False, r=None, reduction_basis='A', TSVD=False, reduction=0, print_=True):
    """
    Function performs decoupling of substructure A from structure AB using LM-FBS method.
    :param Y_AB: admitance matrix of structure Y_AB. Must be constructed in such manner that the interface DoFs are
    placed after internal DoFs.
    |Y_ii Y_ib|
    |Y_bi Y_bb|
    b - boundary (interface DoFs)
    :param SVT: if True Singular Vector Transformation (SVT) is performed
    :param r: size of SVT reduction bases
    :param reduction_basis: 'A' or 'AB' (str)
    :param Y_A: admitance matrix of substructure Y_A (Consists only of the interface DoFs).ž
    :param TSVD: if True the truncated singular value decomposition is performed on Y_int matrix
    :param reduction: number of singular values not to be taken into account in TSVD
    :param print_: if True the progress of the coupling is printed
    :return: decoupled admitance Y_B
    """
    if SVT:
        Y_AB, Y_A = SVT_4_decoupling(Y_AB, Y_A, r=r, reduction_basis=reduction_basis)
        if print_:
            print('SVT process finished!')

    # construction of the uncoupled admitance matrix
    Y_AB_B = np.zeros((Y_AB.shape[0], Y_AB.shape[1]+Y_A.shape[1], Y_AB.shape[2]+Y_A.shape[2]), dtype=complex)
    Y_AB_B[:, :Y_AB.shape[1], :Y_AB.shape[2]] = Y_AB
    Y_AB_B[:, Y_AB.shape[1]:, Y_AB.shape[2]:] = Y_A

    # construction of the signed Boolean matrices
    B_c = np.zeros((Y_A.shape[2], Y_AB_B.shape[1]))
    B_e = np.zeros((Y_A.shape[2], Y_AB_B.shape[2]))
    B_c[:, Y_AB_B.shape[1]-2*Y_A.shape[2]:Y_AB_B.shape[1]-Y_A.shape[2]] = np.eye(Y_A.shape[2])
    B_c[:, Y_AB_B.shape[1]-Y_A.shape[2]:] = -1*np.eye(Y_A.shape[2])
    B_e[:, Y_AB_B.shape[2]-2*Y_A.shape[2]:Y_AB_B.shape[2]-Y_A.shape[2]] = np.eye(Y_A.shape[2])
    B_e[:, Y_AB_B.shape[2]-Y_A.shape[2]:] = -1*np.eye(Y_A.shape[2])

    # LM-FBS decoupling
    Y_B = LMFBS(Y_AB_B, B_c, B_e, TSVD=TSVD, reduction=reduction)
    if print_:
        print('Decoupling process finished!')

    return Y_B, B_c, B_e


def SVT_4_decoupling(frf_AB, frf_A, r, reduction_basis='A', ext_dof=None):
    """
    Function performs a singular vector transformation (SVT) for the admitance matrices frf_AB and frf_A using the left
    and right singular vectors of the frf_A matrix, obtained by SVD. 'r' elements of singular vectors are kept.
    :param frf_AB: admitance marix of structure AB
    :param frf_A: admitance matrix of substructure A (only interface DoFs)
    :param r: number of kept values of the left and right singular vectors
    :param reduction_basis: 'A' or 'AB' (str)
    :param ext_dof: number of external degrees of freedom
    :return: transformed admitance matrices Y_AB_ and Y_A_
    """
    if ext_dof is None:
        ext_dof = frf_AB.shape[1]-frf_A.shape[1]

    # SVD - condition: frf_A contains only interface DoFs
    if reduction_basis == 'A':
        U_, s_, Vh_ = np.linalg.svd(frf_A)
    elif reduction_basis == 'AB':
        U_, s_, Vh_ = np.linalg.svd(frf_AB[:, ext_dof:, ext_dof:])

    # transformation matrices
    T_u_A = U_[:, :, :r].transpose(0, 2, 1).conjugate()
    T_f_A_H = Vh_[:, :r, :].transpose(0, 2, 1).conjugate()

    # transformation matrices expansion to the non-interface DoFs
    T_u_A_full = np.zeros((T_u_A.shape[0], ext_dof+T_u_A.shape[1], ext_dof+T_u_A.shape[2]), dtype=complex)
    T_u_A_full[:, :ext_dof, :ext_dof] = np.eye(ext_dof)
    T_u_A_full[:, ext_dof:, ext_dof:] = T_u_A
    T_f_A_H_full = np.zeros((T_f_A_H.shape[0], ext_dof+T_f_A_H.shape[1], ext_dof+T_f_A_H.shape[2]), dtype=complex)
    T_f_A_H_full[:, :ext_dof, :ext_dof] = np.eye(ext_dof)
    T_f_A_H_full[:, ext_dof:, ext_dof:] = T_f_A_H

    # transformed FRFs
    Y_AB_ = T_u_A_full@frf_AB@T_f_A_H_full
    Y_A_ = T_u_A@frf_A@T_f_A_H

    return Y_AB_, Y_A_


def SVT_back_projection(Y, r):
    """
    Back projection of Y matrix using Singular Vector Transformation (SVT)
    Args:
        Y: input matrix (numpy array)
        r: SVT reduction bases size (int)

    Returns: back projected Y matrix

    """
    # SVD
    U, s, Vh = np.linalg.svd(Y)

    # reduction bases
    U_r = U[:, :, :r]
    V_r = H(Vh[:, :r, :])

    # Transformation matrices
    T_u_A = H(U_r)
    T_f_A_H = V_r

    # Filter matrices
    F_u = U_r @ T_u_A  # for compatibility
    F_f = V_r @ H(T_f_A_H)  # for equilibrium

    return F_u @ Y @ F_f


# --------------------------------------------------------------------------------------------------------------------
# SEMM - with trust function


def SEMM_tf(Y_exp, Y_num, df_chn_exp, df_imp_exp, df_chn_num, df_imp_num, W=None, semm_type=None):
    """
    Implementation of SEMM method with trust function.
    Args:
        Y_exp: experimental admittance matrix
        Y_num: numerical admittance matrix
        df_chn_exp: dataframe with experimental channel locations
        df_imp_exp: dataframe with experimental impact locations
        df_chn_num: dataframe with numerical channel locations
        df_imp_num: dataframe with numerical impact locations
        W: np array representing a trust function
        semm_type: semm type (str) - 'fully_extend' (other types not implemented yet)

    Returns:

    """
    # skupne prostostne stopnje
    common_chn = pyFBS.find_locations_in_data_frames(df_chn_exp, df_chn_num)
    common_imp = pyFBS.find_locations_in_data_frames(df_imp_exp, df_imp_num)
    cmn_chn_exp, cmn_chn_num, cmn_imp_exp, cmn_imp_num = common_chn[:, 0], common_chn[:, 1],\
        common_imp[:, 0], common_imp[:, 1]

    # glavna, eliminirana in prekrivna matrika
    t_start = timeit.default_timer()
    Y_par = Y_num.copy()
    if semm_type == 'fully_extend':
        Y_rem = Y_num.copy()
    else:
        Y_rem = (Y_num[:, cmn_chn_num, :][:, :, cmn_imp_num]).copy()
    Y_ov = Y_exp[:, cmn_chn_exp, :][:, :, cmn_imp_exp].copy()

    # regular SEMM
    if W is None:
        W = np.ones(Y_exp.shape[0])

    # adm. matrika nesklopljenega sistema
    Y_unc = np.zeros((Y_par.shape[0],
                      Y_par.shape[1] + Y_rem.shape[1] + Y_ov.shape[1],
                      Y_par.shape[2] + Y_rem.shape[2] + Y_ov.shape[2],), dtype=complex)
    Y_unc[:, :Y_par.shape[1], :Y_par.shape[2]] = Y_par
    Y_unc[:, Y_par.shape[1]:Y_par.shape[1] + Y_rem.shape[1], Y_par.shape[2]:Y_par.shape[2] + Y_rem.shape[2]] = -1 * Y_rem
    Y_unc[:, -Y_ov.shape[1]:, -Y_ov.shape[2]:] = Y_ov
    t_end_unc = timeit.default_timer()
    print(f'Y_uncoupled built --> {t_end_unc - t_start:.3f} s')

    # Prostostne stopnje za sklop - odklop
    t_start_bool = timeit.default_timer()
    if semm_type == 'fully_extend':
        dof_chn_dec, dof_chn_coup = np.arange(Y_rem.shape[1]), cmn_chn_num
        dof_imp_dec, dof_imp_coup = np.arange(Y_rem.shape[2]), cmn_imp_num
    else:
        dof_chn_dec, dof_chn_coup = cmn_chn_exp, np.arange(Y_rem.shape[1])
        dof_imp_dec, dof_imp_coup = cmn_imp_exp, np.arange(Y_rem.shape[2])

        # Boolovi matriki
    B_c = np.zeros((Y_rem.shape[1] + len(cmn_chn_exp), Y_par.shape[1] + Y_rem.shape[1] + Y_ov.shape[1]))
    B_e = np.zeros((Y_rem.shape[2] + len(cmn_imp_exp), Y_par.shape[2] + Y_rem.shape[2] + Y_ov.shape[2]))
    # za kompatibilnistni pogoj
    # - odklop odstranjenega modela
    B_c[:len(dof_chn_dec), dof_chn_dec] = -1 * np.eye(len(dof_chn_dec))
    B_c[:len(dof_chn_dec), Y_par.shape[1]:Y_par.shape[1] + Y_rem.shape[1]] = np.eye(len(dof_chn_dec))
    # - priklop prekrivnega modela
    B_c[len(dof_chn_dec):, Y_par.shape[1] + dof_chn_coup] = -1 * np.eye(len(cmn_chn_exp))
    B_c[len(dof_chn_dec):, Y_par.shape[1] + Y_rem.shape[1] + cmn_chn_exp] = np.eye(len(cmn_chn_exp))
    # za ravnotežni pogoj
    # - odklop odstranjenega modela
    B_e[:len(dof_imp_dec), dof_imp_dec] = -1 * np.eye(len(dof_imp_dec))
    B_e[:len(dof_imp_dec), Y_par.shape[2]:Y_par.shape[2] + Y_rem.shape[2]] = np.eye(len(dof_imp_dec))
    # - priklop prekrivnega modela
    B_e[len(dof_imp_dec):, Y_par.shape[2] + dof_imp_coup] = -1 * np.eye(len(cmn_imp_exp))
    B_e[len(dof_imp_dec):, Y_par.shape[2] + Y_rem.shape[2] + cmn_imp_exp] = np.eye(len(cmn_imp_exp))
    # kompatibilnostne matrike z utežno funkcijo
    B_c = np.einsum('i,jk->ijk', np.ones(Y_par.shape[0]), B_c)
    B_e = np.einsum('i,jk->ijk', np.ones(Y_par.shape[0]), B_e)
    B_c[:, :, :Y_par.shape[1]] = np.einsum('ijk,i->ijk', B_c[:, :, :Y_par.shape[1]], W)
    B_e[:, :, :Y_par.shape[2]] = np.einsum('ijk,i->ijk', B_e[:, :, :Y_par.shape[2]], W)
    # B_c[:, :, Y_par.shape[1]+Y_rem.shape[1]:] = np.einsum('ijk,i->ijk', B_c[:, :, -Y_ov.shape[1]:], W1)
    # B_e[:, :, Y_par.shape[2]+Y_rem.shape[2]:] = np.einsum('ijk,i->ijk', B_e[:, :, -Y_ov.shape[2]:], W1)
    t_end_bool = timeit.default_timer()
    print(f'B_c, B_e built --> {t_end_bool - t_start_bool:.3f} s')

    # LM-FBS
    t_start_lmfbs = timeit.default_timer()
    axes = (0, 2, 1)
    Y_coupl = Y_unc - Y_unc @ T(B_e, axes) @ np.linalg.pinv(B_c @ Y_unc @ T(B_e, axes)) @ B_c @ Y_unc
    t_end_lmfbs = timeit.default_timer()
    print(f'Y_semm built --> {t_end_lmfbs - t_start_lmfbs:.3f} s')
    print('-----------------------------------------------')
    print(f't_total: {t_end_lmfbs-t_start:.3f} s')

    # rezanje odvečnih prostostnih stopenj
    return B_c, B_e, Y_coupl[:, :Y_num.shape[1], :][:, :, :Y_num.shape[2]]


# ------------------------------------------------------------------------------------------------------------------- #
# Joint modelling
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
    Function calculates joint FRF. - More Flexible than get_Y_J.

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

    :returns: Joint Y matrix
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


# ------------------------------------------------------------------------------------------------------------------- #
# Primerjava FRF
def plot_FRFs(Y, i, j, labels, x_lim, title=False, ls=None, figsize=(15, 5),  FRF_type='receptance', figure=None,
                    axes=None):
    """
    Function plots amplitude and phase of FRFs listed in Y.
    Args:
        FRF_type: 'receptance', 'mobility' or 'accelerance'; 'dynamic_stiffness', 'nechanical_impedance', 'apparent mass'
        Y: list of admittance matrices
        i: row index
        j: column index
        labels: labels for legend for each admittance matrix
        x_lim: tuple of x limits
        title: graph title
        ls: list of line styles
        figsize: figure size

    Returns: None

    """
    # %matplotlib inline
    plt.rcParams['font.size'] = 14
    if (figure is None) and (axes is None):
        fig, ax = plt.subplots(2, 1, figsize=figsize)
    else:
        fig = figure
        ax = axes
    # fig.suptitle('Primerjava: $\mathbf{Y_{num}}$ vs. $\mathbf{Y_{exp}}$')
    if title:
        if title is True:
            fig.suptitle(f"Acc: {i}, imp: {j}")
        elif type(title) == str:
            fig.suptitle(title)
        else:
            print('Invalid title type')
            return

    for ind, (Y_, lab_) in enumerate(zip(Y, labels)):
        if ls is not None:
            ls_ = ls[ind]
        else:
            ls_ = '-'

        ax[0].semilogy(abs(Y_[:x_lim, i, j]), label=lab_, ls=ls_)
    ax[0].grid()
    ax[0].set_xlabel('f [hz]')
    if FRF_type == 'receptance':
        ax[0].set_ylabel('A [m/N]')
    elif FRF_type == 'mobility':
        ax[0].set_ylabel('A [m/(sN)]')
    elif FRF_type == 'accelerance':
        ax[0].set_ylabel('A [m/(s$^2$N)]')
    elif FRF_type == 'dynamic_stiffness':
        ax[0].set_ylabel('Z [N/m]')
    elif FRF_type == 'mechanical_impedance':
        ax[0].set_ylabel('Z [N/(m/s)]')
    elif FRF_type == 'apparent_mass':
        ax[0].set_ylabel('Z [N/(m/s$^2$)]')
    else:
        print('Invalid FRF type')
        return
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


def coh_plot(coh_matrix, vmin=0, vmax=1, figsize=(5, 6), display_averages=False):
    """
    Function plots matrix of coherence values.
    Args:
        coh_matrix: coherence matrix
        vmin: bottom limit for colorbar
        vmax: upper limit for colorbar
        figsize: size of the figure
        display_averages: text display of average coherence values

    Returns:

    """
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


# ----------------------------- FRF reconstruction ----------------------------------------------------------------- #
def FRF_reconstruction(ω, ω_r, damping_r, θ, damping_model, dp_loc=0):
    """
    Modal superposition based FRF reconstruction. Enables reconstruction of single admitance matrix row or column
    :param ω: numeric array of frquencies
    :param ω_r: numeric array of natural frequencies (rad/s)
    :param damping_r: numeric array of damping factors (η in hysteretic damping model, ζ in viscous damping model)
    :param damping_model: hyst (hysteretic) or visc (viscous)
    :param dp_loc: driving point DoF index
    :param θ: numeric array of mode shapes
    :return: reconstructed FRF
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


# -------------------------------------------------------------------------------------------------------------------#
def Fourier_transform(meas_data, dt, force_window_=False, min_max_ratio=100, exp_window=False, exp_w_end=.01,
                      show=False, acc_chn=3, time_delay=0):
    """
    Fourier transform of measurements with modal hammer.
    Meas_data dictionary form: {'key:file name': np.array(freqencies, channels, subsequent measurements)}

    - channel 0: force measurement
    - the following channels: acceleration measurements

    :param meas_data: dictionary with measurements
    :param dt: time interval between samples
    :param force_window_: if True the force window is applied
    :param min_max_ratio: ratio of max. force value and individual measurement over which the force window is applied
    :param show: plot excitation forces
    :param acc_chn: number of acceleration DoFs per sensor
    :param time_delay: time delay between force and acceleration measurements
    :param exp_window: if True the exponential window is applied
    :param exp_w_end: end of the exponential window
    :return:
    acc - dictionary of accelerations {'key-file name':np.array(frequencies, channels, subsequent measurements)}
    force - dictionary of forces {'key-file name':np.array(frequencies, channels, subsequent measurements)}
    fr - numpy array of discrete frequency points
    """
    file_names = list(meas_data.keys())
    acc = {}
    force = {}
    for i in tqdm(file_names, leave=False):
        x = meas_data[i][:, 1:, :]
        f = meas_data[i][:, 0, :]
        if force_window_:
            for j in range(f.shape[1]):
                f[:, j][np.where(abs(max(f[:, j]) / f[:, j]) > min_max_ratio)] = 0
        if show:
            plt.plot(f[:150])
        if exp_window:
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    if i == file_names[0] and j == 0 and k == 0:
                        plt.plot(x[:, j, k])
                    w_x = np.exp(np.log(exp_w_end) * np.arange(x.shape[0]) / (x.shape[0] - 1))  # eksponentno okno
                    x[:, j, k] = w_x * x[:, j, k]
                    if i == file_names[0] and j == 0 and k == 0:
                        plt.plot(x[:, j, k], alpha=.5)
        n = x.shape[0]
        if x.shape[0] % 2 == 0:
            shape = int(n / 2 + 1)
        else:
            shape = int((n + 1) / 2)
        X = np.zeros((shape, acc_chn, x.shape[2]), dtype=complex)
        F = np.zeros((shape, f.shape[1]), dtype=complex)
        for j in range(x.shape[2]):
            for L in range(acc_chn):
                fr = np.fft.rfftfreq(x.shape[0], d=dt)
                X[:, L, j] = np.fft.rfft(x[:, L, j])*np.exp(-1.j * 2 * np.pi * fr * time_delay)
            F[:, j] = np.fft.rfft(f[:, j])
        acc[i] = X
        force[i] = F

    return acc, force, fr


# -------------------------------------------------------------------------------------------------------------------#
# Experimental modal analysis
@njit
def get_FRF(X, F, filter_list=None, estimator='H1', kind='admittance'):
    """
    Function calculates frequency response functions (FRF) from measurement data.
    :param X: np.array of accelerations (frequencies, repeated measurements)
    :param F: np.array of accelerations (frequencies, repeated measurements)
    :param filter_list: list of indices of measurements to be excluded from the FRF calculation
    :param estimator: FRF estimator (H1, H2)
    :param kind: FRF type (admittance/impedance)
    :return: averaged FRF
    """
    N = X.shape[1]
    # Izračun cenilk prenosne funkcije
    if estimator == 'H1':
        S_fx_avg = np.zeros_like(X[:, 0])
        S_ff_avg = np.zeros_like(F[:, 0])
    elif estimator == 'H2':
        S_xx_avg = np.zeros_like(X[:, 0])
        S_xf_avg = np.zeros_like(F[:, 0])
    else:
        S_fx_avg, S_ff_avg, S_xx_avg, S_xf_avg = None, None, None, None
        raise Exception('Invalid estimator. Enter H1 or H2.')
    for i in range(N):
        if estimator == 'H1':
            if filter_list is not None:
                if i not in filter_list:
                    S_fx_avg += np.conj(F[:, i]) * X[:, i]
                    S_ff_avg += np.conj(F[:, i]) * F[:, i]
            else:
                S_fx_avg += np.conj(F[:, i]) * X[:, i]
                S_ff_avg += np.conj(F[:, i]) * F[:, i]
        elif estimator == 'H2':
            if filter_list is not None:
                if i not in filter_list:
                    S_xx_avg += np.conj(X[:, i]) * X[:, i]
                    S_xf_avg += np.conj(X[:, i]) * F[:, i]
            else:
                S_xx_avg += np.conj(X[:, i]) * X[:, i]
                S_xf_avg += np.conj(X[:, i]) * F[:, i]
        else:
            print('Invalid estimator')
            return
    if estimator == 'H1':
        if kind == 'admittance':
            return S_fx_avg / S_ff_avg
        elif kind == 'impedance':
            return S_ff_avg / S_fx_avg
        else:
            print('Invalid FRF type')
            return
    elif estimator == 'H2':
        if kind == 'admittance':
            return S_xx_avg / S_xf_avg
        elif kind == 'impedance':
            return S_xf_avg / S_xx_avg
        else:
            print('Invalid FRF type')
            return


def LAC(s_1, s_2, mean=False):
   
    """Compute Local Amplitude Criterion (LAC) of (complex or real-valued) signals s_1 and s_2.
    -----------
    Parameters:
    -----------
    s_1, s_2 : numpy.ndarray with identical shapes or at least shapes such that (s_1 + s_2) returns a valid result.
    """

    lac_ = ((2 * np.abs(s_1.conj() * s_2)) / ((s_1.conj() * s_1) + (s_2.conj() * s_2))).real
   
    if mean:
        return np.mean(lac_)
   
    else:
        return lac_


""" def MAC(X_1, X_2, mean=False):
    Modal assurance criterion (MAC) of two sets of mode shapes.
    mac_ = (np.abs(X_1)) """


def find_nat_fr(frf, plot=False):
    """
    Function identifies natural frequencies for the entered frequency response function (FRF).
    Args:
        frf: Frequency response function (numpy.array)
        plot: if True FRF is drawn together with identified natural frequencies

    Returns: list of identified natural frequencies

    """
    # find natural frequencies
    eig_fr = []
    for i in range(frf.shape[0]-1):
        if ((np.angle(frf[i]) > np.pi/2) & (np.angle(frf[i+1]) < np.pi/2) & (np.angle(frf[i+1]) > -1)) | \
                ((np.angle(frf[i]) > -np.pi/2) & (np.angle(frf[i+1]) < -np.pi/2) & (np.angle(frf[i]) < 1)):
            eig_fr.append(i+1)
    # draw graph
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        # x_lim = (0, 2000)
        ax[0].semilogy(np.abs(frf[:]))
        ax[0].semilogy(eig_fr, np.abs(frf[eig_fr]), 'o')
        ax[1].plot(np.angle(frf[:]))
        ax[1].plot(eig_fr, np.angle(frf[eig_fr]), 'o')
        # ax[0].set_xlim(x_lim[0], x_lim[1])
        # ax[1].set_xlim(x_lim[0], x_lim[1])
        ax[0].grid()
        ax[1].grid()
    return eig_fr


def MIMO_csd(x, f, fs=1, segment=2048, overlap=1024, n_fft=None, window='hann'):
    """
    Avtor: Domen
    :param x: časovna vrsta izmerjenega odziva
    :param f: časovna vrsta izmerjenega vzbujanj
    :param fs: frekvenca vzorčenja
    :param segment: število vzorcev v posameznem bloku meritve
    :param overlap: število vzorcev ki se prekrivajo med dvema zaporednima blokoma
    :param n_fft: število frekvenčnih linij za FFT
    :param window: tip oknjenja
    :return freq: vektor frekvenc
    :return H1: cenilka frekvenčnih prenosnih funkcij
    """

    if n_fft is None:
        n_fft = segment

    G_xf = np.zeros((n_fft // 2 + 1, x.shape[1], f.shape[1]), dtype=complex)
    G_ff = np.zeros((n_fft // 2 + 1, f.shape[1], f.shape[1]), dtype=complex)

    for i in tqdm(range(f.shape[1])):
        for j in range(x.shape[1]):
            freq, G_xf[:, j, i] = signal.csd(x[:, j], f[:, i], fs, window=window, axis=0, nperseg=segment,
                                             noverlap=overlap, nfft=n_fft)
        for k in range(f.shape[1]):
            G_ff[:, i, k] = signal.csd(f[:, i], f[:, k], fs, window=window, axis=0, nperseg=segment, noverlap=overlap,
                                       nfft=n_fft)[1]

    G_ff_inv = np.linalg.pinv(G_ff)
    H1 = G_xf @ G_ff_inv

    return freq, H1


def transform_response(from_to, array, omega):
    """
    Transformations beween acceleration, mobility and receptance.
    """
    from_, to_ = from_to.split('->')
    if (from_ == 'acc' and to_ == 'mob') or (from_ == 'mob' and to_ == 'rec'):
        return acc_to_mob_to_rec(array, omega)
    elif from_ == 'acc' and to_ == 'rec':
        arr_ = acc_to_mob_to_rec(array, omega)
        return acc_to_mob_to_rec(arr_, omega)
    elif (from_ == 'rec' and to_ == 'mob') or (from_ == 'mob' and to_ == 'acc'):
        return rec_to_mob_to_acc(array, omega)
    elif from_ == 'rec' and to_ == 'acc':
        arr_ = rec_to_mob_to_acc(array, omega)
        return rec_to_mob_to_acc(arr_, omega)
    else:
        print('Invalid transform: check \'from_to\' value!')
        return


def acc_to_mob_to_rec(array, omega):
    """
    Integration in frequency domain.
    """
    array_str = get_einsum_str(array)
    return np.einsum(array_str+','+array_str[0]+'->'+array_str, array, 1/(1.j*omega))


def rec_to_mob_to_acc(array, omega):
    """
    Derivation in frequency domain.
    """
    array_str = get_einsum_str(array)
    return np.einsum(array_str+','+array_str[0]+'->'+array_str, array, 1.j*omega)


def get_einsum_str(array):
    """
    Prepares string for numpy.einsum based on the size of the input array.
    """
    signs = 'ijklmnopqr'
    return signs[:len(array.shape)]


def Ewins_Gleeson(FRF, freq, fr_min, fr_max, eigfr, N, reconstruct=False, return_points=False):
    """
    Implementation of Ewins-Gleeson method for modal parameter estimation.
    Args:
        FRF: frequency response function
        freq: frequency vector
        fr_min: minimum frequency for estimation
        fr_max: maximum frequency for estimation
        eigfr: natural frequencies
        N: number of points for estimation
        reconstruct: reconstruct FRF from estimated parameters
        return_points:

    Returns:

    """
    Sigma = freq[np.linspace(0, len(freq[fr_min:fr_max]), N, dtype=int)]
    Alpha = FRF.real[np.linspace(0, len(freq[fr_min:fr_max]), N, dtype=int)]
    R = np.ones((len(Sigma), len(eigfr)))
    for i, j in enumerate(Sigma):
        R[i, :] = 1/(eigfr**2 - j**2)
    C = np.linalg.pinv(R)@Alpha
    eta = np.zeros(len(eigfr))
    for i, j in enumerate(eigfr):
        eta[i] = abs(C[i])/((abs(FRF)[np.argwhere(freq == j)[0]])*j**2)
    if reconstruct:
        FRF_rec = np.zeros_like(FRF[fr_min:fr_max])
        for i, j, k in zip(eigfr, C, eta):
            FRF_rec += j/(i**2-freq[fr_min:fr_max]**2 + 1.j*k*i**2)
        if return_points:
            return C, eta, FRF_rec, Sigma
        return C, eta, FRF_rec
    if return_points:
        return C, eta, Sigma
    return C, eta


def get_eigfr(FRF, freq, fr_min_max_list):
    """
    Function returns eigenfrequencies from FRF.
    Args:
        FRF: frequency response function
        freq: frequency vector
        fr_min_max_list: limits for eigenfrequency estimation

    Returns:

    """
    eigfr = np.zeros(len(fr_min_max_list))
    for j, i in enumerate(fr_min_max_list):
        eigfr[j] = freq[(i[0] < freq) & (i[1] > freq)][np.argmax(abs(FRF[(i[0] < freq) & (i[1] > freq)]))]
    return eigfr


def force_window(F_arr, t_w, sample_rate):
    """
    Function applies force window to the force signal array.
    Parameters
    ----------
    F_arr: Force signal array
    t_w: time at which window is applied
    sample_rate: sampling rate of force signal.

    :returns: windowed force signal
    -------

    """
    start_ind = int(np.ceil(t_w*sample_rate))
    F_arr_new = F_arr.copy()
    F_arr_new[start_ind:] = 0
    return F_arr_new
