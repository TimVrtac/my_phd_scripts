import numpy as np
import pyFBS


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

    Y_int = B_c @ Y_AB_B @ B_e.T
    if TSVD:
        Y_int = pyFBS.TSVD(Y_int, reduction=reduction)

    # LM-FBS decoupling
    Y_B = Y_AB_B - Y_AB_B @ B_e.T @ np.linalg.pinv(Y_int) @ B_c @ Y_AB_B
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

    Y_int = B_c @ Y_AB_B @ B_e.T
    if TSVD:
        Y_int = pyFBS.TSVD(Y_int, reduction=reduction)

    # LM-FBS decoupling
    Y_B = Y_AB_B - Y_AB_B @ B_e.T @ np.linalg.pinv(Y_int) @ B_c @ Y_AB_B
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
    cmn_chn_exp, cmn_chn_num, cmn_imp_exp, cmn_imp_num = common_chn[:, 0], common_chn[:, 1], common_imp[:, 0], common_imp[:, 1]

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
    Y_unc[:, Y_par.shape[1]:Y_par.shape[1] + Y_rem.shape[1],
    Y_par.shape[2]:Y_par.shape[2] + Y_rem.shape[2]] = -1 * Y_rem
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
    #B_c[:, :, Y_par.shape[1]+Y_rem.shape[1]:] = np.einsum('ijk,i->ijk', B_c[:, :, -Y_ov.shape[1]:], W1)
    #B_e[:, :, Y_par.shape[2]+Y_rem.shape[2]:] = np.einsum('ijk,i->ijk', B_e[:, :, -Y_ov.shape[2]:], W1)
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
def plot_ampl_phase(Y, i, j, labels, x_lim, title=False, ls=None, figsize=(15, 5)):
    """
    Function plots amplitude and phase of FRFs listed in Y.
    Args:
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