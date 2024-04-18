from ansys.mapdl import reader
import numpy as np
from tqdm.notebook import tqdm
import pyvista as pv
from tools import match_coordinates
from IPython.display import clear_output, display
from ipywidgets import Output
import sys
from numba import njit


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


def get_sign(x):
    if x[0] == '-':
        return -1
    else:
        return 1


def get_eigvec(points, directions, nodes, eigvec, atol=1e-8, factor=1, verbose=0, no_dofs=3, return_indices=False):
    """
    Function gets the mode shapes in DoFs of specified points and directions.
    :param points: points in which mode shapes are to be extracted (np.array)
    :param directions: directions in which mode shapes are to be extracted (list of strings)
    :param nodes: nodes of the model - from rst file (np.array)
    :param eigvec: eigenvectors - from rst file (np.array)
    :param atol: tolerance of point locations for finding nodes (float)
    :param factor: factor by which point coordinates are multiplied (float)
    :param verbose: 0 - no print, 1- only number of matched eigvecs returned, 2 - printed detailes of each matched point (int)
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
        if verbose==2:
            print(i, j, points_ind, points_ind*3+dir_dict[j[-1]], dir_dict[j[-1]], j)
    # get eigvec
    eigvec_ = eigvec[dofs, :]
    signs = np.array(list(map(get_sign, directions)))
    eigvec_ = np.einsum('i,ij->ij', signs, eigvec_)
    if (eigvec_.shape[0] == points.shape[0]) and (verbose>0):
        print(f'OK. {eigvec_.shape[0]}/{points.shape[0]} eigenvectors found.')
    elif (eigvec_.shape[0] != points.shape[0]) and (verbose>0):
        print(f'Eigenvectors missing. {eigvec_.shape[0]}/{points.shape[0]} eigenvectors found.')
    if return_indices:
        return eigvec_, indices
    else:
        return eigvec_


# @njit
def get_FRF(acc_eig_vec, imp_eig_vec, eig_freq, freq, damping_type='hyst', modes=None, damping=0.003):
    """
    Function returns accelerance FRF calculated using modal superposition for inserted eigenvectors and natural
    frequencies.
    :param acc_eig_vec: channel dofs' eigenvectors (np.array)
    :param imp_eig_vec: excitation dofs' eigenvectors (np.array)
    :param eig_freq: natural frequencies in rad/s (np.array)
    :param freq: frequencies at which FRF is calculated (np.array)
    :param damping_type: type of damping model ('viscous' (also 'visc') or 'hysteretic' (also 'hyst')) (str)
    :param modes: number of modes included in modal superposition (int)
    :param damping: general damping factor (float) or list of damping factors for chosen number of modes (list of
    floats)
    :return: accelerance type FRF (np.array)
    """
    omega = 2*np.pi*freq
    if modes is None:
        modes = acc_eig_vec.shape[1]
    FRF = np.zeros([acc_eig_vec.shape[0], imp_eig_vec.shape[0], omega.shape[0]], dtype=complex)
    # for i in tqdm(range(modes), leave=False):
    for i in range(modes):
        vec1 = acc_eig_vec[:, i][:, np.newaxis]
        vec2 = imp_eig_vec[:, i][np.newaxis]
        val = eig_freq[i]
        if type(damping) == float:
            damping_i = damping
        else:
            damping_i = damping[i]
        if (damping_type == 'hyst') or (damping_type == 'hysteretic'):
            FRF += hyst_damping_single_mode_contrib(vec1, vec2, val, omega, damping_i)  # np.einsum('ij,k->ijk', (vec1@vec2), 1/(val**2-omega**2 + 1.j*damping*val**2))
        elif (damping_type == 'visc') or (damping_type == 'viscous'):
            #print('ERROR in viscous damping!!!!')
            FRF += visc_damping_single_mode_contrib(vec1, vec2, val, omega, damping_i)
        else:
            raise ValueError('Wrong damping type.')

    return FRF


def hyst_damping_single_mode_contrib(vec1, vec2, eig_freq, omega, damping):
    """
    Function returns single mode contribution to FRF with hysteresis damping.
    Args:
        vec1 (np.array): response dof eigenvector
        vec2 (np.array): excitation dof eigenvector
        eig_freq (float): eigenfrequency
        omega (np.array): array of frequencies in rad/s
        damping (float): damping factor

    Returns:
        np.array: individual mode contribution to admittance matrix
    """
    return np.einsum('ij,k->ijk', (vec1@vec2), 1/(eig_freq**2-omega**2 + 1.j*damping*eig_freq**2))


def visc_damping_single_mode_contrib(vec1, vec2, eig_freq, omega, damping):
    """
    Function returns single mode contribution to FRF with viscous damping.
    Args:
        vec1 (np.array): response dof eigenvector
        vec2 (np.array): excitation dof eigenvector
        eig_freq (float): eigenfrequency
        omega (np.array): array of frequencies in rad/s
        damping (float): damping factor
    
    Returns:
        np.array: individual mode contribution to admittance matrix
        """
    #clen1 = np.einsum('ij,k->ijk', vec1 @ vec2, 1 / (eig_freq * damping + 1.j * (omega - eig_freq * np.sqrt(1 - damping ** 2))))
    #clen2 = np.einsum('ij,k->ijk', np.conj(vec1) @ np.conj(vec2), 1 / (eig_freq * damping + 1.j * (omega + eig_freq * np.sqrt(1 - damping ** 2))))
    return np.einsum('ij,k->ijk', (vec1@vec2), 1/(eig_freq**2-omega**2 + 2*1.j*damping*eig_freq*omega)) #clen1 + clen2


# ---------------------------------------------------------------------------------
# Visualization using PyVista
class PlotObj:
    """Class for plotting from .rst file (Ansys results) and choosing points on the plot."""
    def __init__(self, rst, atol, notebook=False, origin_scale=None, track_click_position=False, plot_nodes=False):
        self.rst = rst
        self.atol = atol
        self.plotter = pv.Plotter(notebook=notebook)
        self.plotter.add_key_event('x', self.close_plot)
        if notebook is False:
            self.plotter.add_text('press "x" to close the plot')
        self.xyz = 'x'
        self.chosen_points = {f'Position_{i + 1}': [] for i in range(3)}
        self.chosen_points_upd = []
        self.dof_to_ind = {'x': 0, 'y': 1, 'z': 2}
        self.ind_to_dof = {1: 'y', 2: 'z', 3: 'x'}

        # plot object
        self.plotter.add_mesh(rst.grid, show_edges=True, style='wireframe')
        if plot_nodes:
            self.plotter.add_points(rst.grid.points, render_points_as_spheres=True, point_size=4., color='black',
                                    opacity=0.5)

        self.plotter.show()

        if origin_scale is not None:
            self.plot_origin(scale=origin_scale)

        if track_click_position:
            self.plotter.track_click_position(callback=self.get_coordinates, side='right', double=False, viewport=False)
            # Click position feedback
            self.out = Output()
            display(self.out)

    def show_plot(self):
        self.plotter.show()

    def close_plot(self):
        self.plotter.close()
        sys.exit(0)

    def add_points(self, points, color='red'):
        """
        Function draws a group of points.
        :param points: numpy array of points
        :param color: color of points
        :return: None
        """
        self.plotter.add_points(points, render_points_as_spheres=True, point_size=10, color=color)
        self.plotter.update()

    def get_coordinates(self, position):
        """
        Callback function for track_click_position function.
        :param position: coordinates of the clicked point
        :return: None
        """
        i = self.dof_to_ind[self.xyz]
        self.chosen_points[f'Position_{i + 1}'].append((np.array(position[i])))
        with self.out:
            print(self.xyz, i, position[i])
        if self.xyz == 'z':
            clicked_point = np.array([[self.chosen_points[f'Position_1'][-1],
                                       self.chosen_points[f'Position_2'][-1],
                                       self.chosen_points[f'Position_3'][-1]]], dtype=float)
            clicked_point_upd = match_coordinates(clicked_point, self.rst.mesh.nodes, atol=self.atol)
            self.plotter.add_points(clicked_point, render_points_as_spheres=True, point_size=10., color='red')
            if clicked_point_upd is not None:
                self.chosen_points_upd.append(clicked_point_upd)
                self.plotter.add_points(clicked_point_upd, render_points_as_spheres=True, point_size=10.,
                                        color='orange')
                with self.out:
                    self.out.clear_output(wait=False)
                    print('Clicked point location: ', clicked_point, '\nUpdated point location: ', clicked_point_upd)
            self.plotter.update()
            # print(clicked_point, clicked_point_upd)
            self.out.clear_output(wait=True)
        self.xyz = self.ind_to_dof[i + 1]
        return

    def get_chosen_points(self):
        """
        Function returns chosen points coordinates.
        Returns: np.array of chosen points coordinates
        """
        return np.array(self.chosen_points_upd).squeeze()

    def plot_origin(self, scale, loc=None):
        """
        Function plots origin of the coordinate system.
        Args:
            scale: scale of the arrows
            loc: location of the origin

        Returns: None

        """
        if loc is None:
            loc = [0, 0, 0]
        x_arrow = pv.Arrow((loc[0], loc[1], loc[2]), (1, 0, 0), scale=scale)
        y_arrow = pv.Arrow((loc[0], loc[1], loc[2]), (0, 1, 0), scale=scale)
        z_arrow = pv.Arrow((loc[0], loc[1], loc[2]), (0, 0, 1), scale=scale)
        self.plotter.add_mesh(x_arrow, color='#87100c')  # dark red
        self.plotter.add_mesh(y_arrow, color='#167509')  # dark green
        self.plotter.add_mesh(z_arrow, color='#06064f')  # dark blue
# ---------------------------------------------------------------------------------
