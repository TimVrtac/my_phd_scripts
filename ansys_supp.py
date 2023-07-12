from ansys.mapdl import reader
import numpy as np
from tqdm.notebook import tqdm


def unpack_rst(rst_path):
    rst = reader.read_binary(rst_path)
    return rst, get_values_from_rst(rst)


# iz pyFBSa
def get_values_from_rst(rst, dofs = None):
    """
    Source: pyFBS
    Return eigenvalues and eigenvectors for a given rst file.
    :param rst: rst file
    :rtype: (array(float), array(float), array(float))
    """
    eigen_freq = rst.time_values * 2 * np.pi  # from Hz to rad/s
    eigen_val = eigen_freq ** 2
    eigen_vec = []
    eigen_vec_strain = []

    for i in range(len(rst.time_values)):
        nnum, disp = rst.nodal_displacement(i)
        eigen_vec.append(disp.flatten())
        try:
            nnum, strain = rst.nodal_elastic_strain(i)
            eigen_vec_strain.append(strain.flatten())
        except:
            pass
    eigen_vec = np.asarray(eigen_vec).T
    try:
        eigen_vec_strain = np.asarray(eigen_vec_strain).T
    except:
        pass
    return eigen_freq, eigen_val, eigen_vec, eigen_vec_strain


def get_eigvec(points, directions, nodes, eigvec, atol=1e-8, factor=1000, print_=True, no_dofs=3, return_indices=False):
    """
    Function gets the mode shapes in DoFs of specified points and directions.
    :param points: points in which mode shapes are to be extracted (np.array)
    :param directions: directions in which mode shapes are to be extracted (list of strings)
    :param nodes: nodes of the model - from rst file (np.array)
    :param eigvec: eigenvectors - from rst file (np.array)
    :param atol: tolerance of point locations for finding nodes (float)
    :param factor: factor by which point coordinates are multiplied (float)
    :param print_: if True, prints the search details (bool)
    :param no_dofs: number of dofs per node (int) - depends on the FE type
    """
    dir_dict = {'x': 0, 'y': 1, 'z': 2}
    dofs = []
    indices = []
    for i, j in zip(points, directions):
        points_ind = np.where(np.isclose(nodes*factor, i, atol=atol).all(axis=1))[0][0]
        indices.append(points_ind)
        if no_dofs == 3:
            dofs.extend([points_ind*3 + dir_dict[j[-1]]])
        elif no_dofs == 1:
            dofs.extend([points_ind])  # in this case directions are not used
        if print_:
            print(i, j, points_ind, points_ind*3+dir_dict[j[-1]], dir_dict[j[-1]], j)
    # get eigvec
    eigvec_ = eigvec[dofs, :]
    if eigvec_.shape[0] == points.shape[0]:
        print(f'OK. {eigvec_.shape[0]}/{points.shape[0]} eigenvectors found.')
    else:
        print(f'Eigenvectors missing. {eigvec_.shape[0]}/{points.shape[0]} eigenvectors found.')
    if return_indices:
        return eigvec_, indices
    else:
        return eigvec_


def get_FRF(acc_eig_vec, imp_eig_vec, eig_freq, freq, modes=None, damping=0.003):
    """
    Function returns accelerance FRF calculated using modal superposition for inserted eigenvectors and natural
    frequencies.
    :param acc_eig_vec: channel dofs' eigenvectors (np.array)
    :param imp_eig_vec: excitation dofs' eigenvectors (np.array)
    :param eig_freq: natural frequencies (np.array)
    :param freq: frequencies at which FRF is calculated (np.array)
    :param modes: number of modes included in modal superposition (int)
    :param damping: general damping factor (float) or list of damping factors for chosen number of modes (list of
    floats)
    :return: accelerance type FRF (np.array)
    """
    omega = 2*np.pi*freq
    if modes is None:
        modes = acc_eig_vec.shape[1]
    FRF = np.zeros([acc_eig_vec.shape[0], imp_eig_vec.shape[0], omega.shape[0]], dtype=complex)
    for i in tqdm(range(modes), leave=False):
        vec1 = acc_eig_vec[:, i][:, np.newaxis]
        vec2 = imp_eig_vec[:, i][np.newaxis]
        val = eig_freq[i]
        if type(damping) == float:
            FRF += np.einsum('ij,k->ijk', (vec1@vec2), 1/(val**2-omega**2 + 1.j*damping*val**2))
        else:
            FRF += np.einsum('ij,k->ijk', (vec1 @ vec2), 1 / (val ** 2 - omega ** 2 + 1.j * damping[i] * val ** 2))
    return FRF
