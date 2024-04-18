import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv
import math
import matplotlib.gridspec as gridspec
from IPython.display import clear_output
import webbrowser
from tempfile import NamedTemporaryFile
import pyperclip


def df_window(df):
    """
    Function for displaying pandas DataFrame in a new window.
    Args:
        df: dataframe to be displayed

    Returns: None
    """
    # Open pd.DataFrame in a new window
    with NamedTemporaryFile(mode='r+', delete=False, suffix='.html') as f:
        df.to_html(f)
    webbrowser.open(f.name)


def inv_by_SVD(G):
    """
    Approximation of matrix inverse using SVD method (Linear Algebra: A modern introduction p. 625)
    :param G: matrix to be inverted
    :param n_kept: number of singular values to be kept
    :return: inverted matrix
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


def match_coordinates(points, mesh, atol, verbose=0):
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
            if verbose > 1:
                print('Missing point: ', i)
            pass
        mask += np.isclose(mesh[:, 0], i[0], atol=atol) &\
                np.isclose(mesh[:, 1], i[1], atol=atol) &\
                np.isclose(mesh[:, 2], i[2], atol=atol)
    if len(order) == points.shape[0] and verbose >= 1:
        print(f'OK. {len(order)}/{points.shape[0]} points matched to mesh')
    elif len(order) != points.shape[0]:
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


# ---------------------------------------------------------------------------------
# PyVista add-ons
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


def imshow_add_vals(vals, color='white', fontsize=10, ax=None, w_lim=None, label_format='float', decimals=3):
    """
    Function adds values to the imshow plot.
    Args:
        im: imshow plot
        vals: values to be added (2D np array)
        color: color of the text
        fontsize: fontsize of the text
        ax: axis of the plot

    Returns: None

    """
    if ax is None:
        ax = plt.gca()
    if w_lim is None:
        w_lim = 0
    for (j, i), label in np.ndenumerate(vals):
        if label < w_lim:
            color = 'white'
        else:
            color = 'black'
        if label_format == 'float':
            label = f'{label:.{decimals}f}'
        elif label_format == 'int':
            label = f'{label:.0f}'
        elif label_format == 'sci':
            label = f'{label:.{decimals}e}'
        ax.text(i, j, f'{label}', ha='center', va='center', color=color, fontsize=fontsize)
    return


def H(matrix):
    """
    Hermitian transpose of matrix
    Args:
        matrix: matrix to be transposed

    Returns: Hermitian transpose of matrix

    """
    return matrix.transpose(0, 2, 1).conjugate()


def get_one_line():
    """
    Function returns one line text from multi line text copied to clipboard - result is also copied to clipboard
    Returns: one line text (str)

    """
    multi_line_text = str(pyperclip.paste())
    lines = multi_line_text.split('\r\n')
    one_line_text = ''
    for line_ in lines:
        if one_line_text == '':
            one_line_text += line_
        elif one_line_text[-1] == '-':
            one_line_text += line_
        else:
            one_line_text += ' ' + line_
    pyperclip.copy(one_line_text)
    return one_line_text
