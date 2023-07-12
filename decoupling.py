import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyFBS
import os
from tools import T
import timeit


def create_xlsx(nodes, name, VP=False, VP_positions=None, VP_chn_DoFs=['ux', 'uy', 'uz', 'rx', 'ry', 'rz'], VP_refchn_DoFs=None, direction=('x', 'y', 'z')):
    """
    Function creates the .xlsx file with sensor and impact data which is to be used in the pyFBS funtions.
    :param nodes: numpy array of node coordinates (rows - individual nodes, columns - x,y,z coordinates)
    :param name: name of the .xlsx file
    :param VP: if True the virtual point data is also created
    :param VP_position: x,y and z coordinates of the virtual point
    :param VP_DoFs: Virtual Point's degrees of freedom
        3 translational (ux, uy, uz), 3 rotational (rx, ry, rz), 3 extensional (ex,ey,ez), 3 torsional (tx, ty, tz),
        6 skewing (sxy, sxz, syz, syx, szx, szy) (list)
    :param VP_DoFs: Virtual Point's degrees of freedom
        3 translational (fx, fy, fz), 3 rotational (mx, my, mz), 3 extensional (ex,ey,ez), 3 torsional (tx, ty, tz),
        6 skewing (sxy, sxz, syz, syx, szx, szy) (list)
    :param direction: list of axes of interest ('x', 'y', 'z')
    :return: none
    """
    # node coordinates
    if VP_positions is None:
        VP_positions = []

    if type(VP_positions) != list:
        VP_positions = [VP_positions]
    positions = pd.DataFrame(nodes)

    positions.rename(columns={0: 'Position_1', 1: 'Position_2', 2: 'Position_3'}, inplace=True)

    # sensors sheet
    sensors = {'Name': [], 'Description': [], 'Grouping': [], 'Quantity': [],
               'Orientation_1': [], 'Orientation_2': [], 'Orientation_3': []}
    for i in range(len(nodes)):
        sensors['Name'].append('s_' + str(i))
        sensors['Description'].append('s_' + str(i))
        sensors['Quantity'].append('Acceleration')
        if VP and (tuple(nodes[i]) in VP_positions):
            sensors['Grouping'].append(VP_positions.index(tuple(nodes[i]))+1)
        else:
            sensors['Grouping'].append(100)
        sensors['Orientation_1'].append(0)
        sensors['Orientation_2'].append(0)
        sensors['Orientation_3'].append(0)

    columns_sens = ['Name', 'Description', 'Grouping', 'Quantity',
                    'Position_1', 'Position_2', 'Position_3',
                    'Orientation_1', 'Orientation_2', 'Orientation_3']
    Sensors = pd.concat([pd.DataFrame(sensors), positions], axis=1)[columns_sens]

    # channels sheet
    channels = {'Name': [], 'Description': [], 'Grouping': [], 'Quantity': [],
                'Position_1': [], 'Position_2': [], 'Position_3': [],
                'Direction_1': [], 'Direction_2': [], 'Direction_3': []}
    axes = ['x', 'y', 'z']
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for i in range(len(nodes)):
        for j, k in enumerate(axes):
            if k in direction:
                channels['Name'].append('s_' + str(i) + k)
                channels['Description'].append('s_' + str(i))
                channels['Quantity'].append('Acceleration')
                if VP and (tuple(nodes[i]) in VP_positions):
                    channels['Grouping'].append(VP_positions.index(tuple(nodes[i]))+1)
                else:
                    channels['Grouping'].append(100)
                channels['Direction_1'].append(directions[j][0])
                channels['Direction_2'].append(directions[j][1])
                channels['Direction_3'].append(directions[j][2])
                channels['Position_1'].append(positions.iloc[i]['Position_1'])
                channels['Position_2'].append(positions.iloc[i]['Position_2'])
                channels['Position_3'].append(positions.iloc[i]['Position_3'])
    Channels = pd.DataFrame(channels)

    # impacts sheet
    impacts = {'Name': [], 'Description': [], 'Grouping': [], 'Quantity': []}
    axes = ['x', 'y', 'z']
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for i in range(len(nodes)):
        for j, k in enumerate(axes):
            if k in direction:
                impacts['Name'].append('F_' + str(i) + k)
                impacts['Description'].append('F_' + str(i))
                impacts['Quantity'].append('Force')
                if VP and (tuple(nodes[i]) in VP_positions):
                    impacts['Grouping'].append(VP_positions.index(tuple(nodes[i]))+1)
                else:
                    impacts['Grouping'].append(100)
    Impacts = pd.concat([pd.DataFrame(impacts), Channels[['Position_1', 'Position_2', 'Position_3',
                                                          'Direction_1', 'Direction_2', 'Direction_3']]], axis=1)
    
    if VP:
        if VP_refchn_DoFs is None:
            VP_refchn_DoFs = VP_chn_DoFs
        VP_refchn_DoFs = np.array(VP_refchn_DoFs)
        ref_dofs = np.array(['fx', 'fy', 'fz', 'mx', 'my', 'mz'])

        for i, j in zip(['ux', 'uy', 'uz', 'rx', 'ry', 'rz'], ref_dofs):
            dof_indices = list(np.where(VP_refchn_DoFs == i))
            if len(dof_indices) > 0:
                VP_refchn_DoFs[dof_indices] = j

        vp_channels = {'Grouping': [], 'Quantity': [], 'Name': [], 'Description': [],
                        'Position_1': [], 'Position_2': [], 'Position_3': [],
                        'Direction_1': [], 'Direction_2': [], 'Direction_3': []}
        vp_ref_channels = {'Grouping': [], 'Quantity': [], 'Name': [], 'Description': [],
                            'Position_1': [], 'Position_2': [], 'Position_3': [],
                            'Direction_1': [], 'Direction_2': [], 'Direction_3': []}

        for VP_position in VP_positions:
            for k, i in enumerate(VP_chn_DoFs):
                for j in [vp_channels, vp_ref_channels]:
                    j['Grouping'].append(VP_positions.index(VP_position)+1)
                    if j == vp_channels:
                        j['Quantity'].append('Acceleration')
                        j['Name'].append(f'VP_{i}')
                        j['Description'].append(i)
                    else:
                        j['Quantity'].append('Force')
                        j['Name'].append(f'VP_{VP_refchn_DoFs[k]}')
                        j['Description'].append(VP_refchn_DoFs[k])

                    j['Position_1'].append(VP_position[0])
                    j['Position_2'].append(VP_position[1])
                    j['Position_3'].append(VP_position[2])
                    j['Direction_1'].append(directions[k% 3][0])
                    j['Direction_2'].append(directions[k % 3][1])
                    j['Direction_3'].append(directions[k % 3][2])
        VP_channels = pd.DataFrame(vp_channels)
        VP_ref_channels = pd.DataFrame(vp_ref_channels)


    # export
    with pd.ExcelWriter(name + '.xlsx') as writer:
        Sensors.to_excel(writer, sheet_name='Sensors')
        Channels.to_excel(writer, sheet_name='Channels')
        Impacts.to_excel(writer, sheet_name='Impacts')
        if VP:
            VP_channels.to_excel(writer, sheet_name='VP_Channels')
            VP_ref_channels.to_excel(writer, sheet_name='VP_RefChannels')


# naredi tako, da bo .xlsx po defaultu temp file
def cut_positioning(x_offset, y_offset, z_offset, cut_nodes, MK_model, ext_nodes=[], xlsx_path='',
                    xlsx_name='none', remove_xlsx=True, VP=False, cut_VP_position=None, VP_chn_DoFs=['ux', 'uy', 'uz', 'rx', 'ry', 'rz'],
                    VP_refchn_DoFs=None, direction=('x', 'y', 'z')):
    """
    Function calculates node coordinates on the basis of cut position on the model which is given by offsets from
    coordinate system origin in individual directions. Coordinates are used to construct data frame which can be used in
    pyFBS' FRF_synth function.
    Args:
        x_offset: offset in x-axis direction (float)
        y_offset: offset in y-axis direction (float)
        z_offset: offset in z-axis direction (float)
        cut_nodes: nodes of the cut (numpy array)
        MK_model: pyFBS.MK_model
        ext_nodes: external nodes (numpy array) - not to be used in decoupling
        xlsx_path: path to the directiory where .xlsx file is to be saved
        xlsx_name: name of the .xlsx file
        remove_xlsx: if True created .xlsx file is removed after import

    Returns: sensors DataFrame, channels DataFrame, impacts DataFrame

    """
    # xlsx_path bi bil lahko na default v temp direktorij, xlsx_name pa 'none'

    # koordinates of the cut in the model's coordinate system
    new_nodes = np.array(cut_nodes.copy())
    new_nodes[:, 0] += x_offset
    new_nodes[:, 1] += y_offset
    new_nodes[:, 2] += z_offset
    x_max, y_max, z_max = max(MK_model.nodes[:, 0]), max(MK_model.nodes[:, 1]), max(MK_model.nodes[:, 2])

    # control: cut must lay inside of boundaries of model's geometry
    if (max(new_nodes[:, 0]) > x_max) | (max(new_nodes[:, 1]) > y_max) | (max(new_nodes[:, 2]) > z_max):
        print('Error: Invalid offset. Cut lays out of model geometry.')
        return

    # all nodes
    nodes = list(ext_nodes) + list(new_nodes)

    if type(cut_VP_position) != list:
        VP_position = [cut_VP_position]
    # Virtual point position
    if VP:
        VP_position = []
        for i in cut_VP_position:
            VP_position.append((i[0]+x_offset, i[1]+y_offset, i[2]+z_offset))
    else:
        VP_position = None
    # create .xlsx file
    create_xlsx(nodes, xlsx_path + xlsx_name, VP=VP, VP_positions=VP_position, direction=direction, VP_chn_DoFs=VP_chn_DoFs, VP_refchn_DoFs=VP_refchn_DoFs)

    # import .xlsx
    df_acc = pd.read_excel(xlsx_path + xlsx_name + '.xlsx', sheet_name='Sensors')
    df_chn = pd.read_excel(xlsx_path + xlsx_name + '.xlsx', sheet_name='Channels')
    df_imp = pd.read_excel(xlsx_path + xlsx_name + '.xlsx', sheet_name='Impacts')



    # update locations
    df_acc = MK_model.update_locations_df(df_acc)
    df_chn = MK_model.update_locations_df(df_chn)
    df_imp = MK_model.update_locations_df(df_imp)

    if VP:
        df_chn_cut_VP = pd.read_excel(xlsx_path + xlsx_name + '.xlsx', sheet_name='VP_Channels')
        df_refchn_cut_VP = pd.read_excel(xlsx_path + xlsx_name + '.xlsx', sheet_name='VP_RefChannels')
        df_chn_cut_VP = MK_model.update_locations_df(df_chn_cut_VP)
        df_refchn_cut_VP = MK_model.update_locations_df(df_refchn_cut_VP)

        for i in [df_chn, df_imp]:
            i.drop_duplicates(subset=['Position_1','Position_2','Position_3','Direction_1','Direction_2','Direction_3'], inplace=True)

        return df_acc, df_chn, df_imp, df_chn_cut_VP, df_refchn_cut_VP


    for i in [df_chn, df_imp]:
        i.drop_duplicates(subset=['Position_1','Position_2','Position_3','Direction_1','Direction_2','Direction_3'], inplace=True)
        i['count'] = np.arange(i.shape[0])
        i.set_index('count', inplace=True)
        
    if os.path.exists(xlsx_path + xlsx_name + '.xlsx') and remove_xlsx:
        os.remove(xlsx_path + xlsx_name + '.xlsx')

    return df_acc, df_chn, df_imp


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


def H(matrix):
    return matrix.transpose(0, 2, 1).conjugate()


def common_nodes(MK_model_AB, MK_model_A, x_offset=0, y_offset=0, z_offset=0, decimals=3, print_=True):
    x = np.intersect1d(np.around(MK_model_A.nodes[:, 0], decimals), np.around(MK_model_AB.nodes[:, 0] - x_offset, decimals))
    y = np.intersect1d(np.around(MK_model_A.nodes[:, 1], decimals), np.around(MK_model_AB.nodes[:, 1] - y_offset, decimals))
    z = np.intersect1d(np.around(MK_model_A.nodes[:, 2], decimals), np.around(MK_model_AB.nodes[:, 2] - z_offset, decimals))

    if print_:
        print(f'common coordinates:\nx-axis: {x}\ny-axis: {y}\nz-axis: {z}')


# --------------------------------------------------------------------------------------------------------------------
# SEMM - with trust function


def SEMM_tf(Y_exp, Y_num, df_chn_exp, df_imp_exp, df_chn_num, df_imp_num, W=None, semm_type=None):
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


# --------------------------------------------------------------------------------------------------------------------
# razrez za celoten izračun - kasneje v svojo datoteko

#import numpy as np
#import matplotlib.pyplot as plt
import pyFBS
import pathlib
import os
import sys
from tqdm.notebook import tqdm
#import pandas as pd
import winsound

#meas_dir = str(pathlib.Path().parent.absolute()) + os.sep + 'Meritve'
#measurements = os.listdir(meas_dir)
#module_path = 'C:\\Users\\timvr\\Documents\\FAKS\\Doktorski_studij\\Python moduli'
#sys.path.insert(-1, module_path)

import modalno_kladivo_analiza as mka


class StructureAnalysis:
    def __init__(self,  AB_model, A_model, measurements, sample_rate, *fr_interval):
        """

        Args:
            AB_model: pyFBS.MK_model of coupled structure
            A_model: pyFBS.MK_model of decoupled substructure
            measurements: measurement data - dictionary of the following structure:
                                {keys/ file names which indicate the excitation (i) and response DoFs (acc)
                                --> example: i10_acc1.npy}
            *fr_interval:
        """
        self.AB_model = AB_model
        self.A_model = A_model
        self.fr_min = fr_interval[0]
        self.fr_max = fr_interval[1]
        self.meas_data = measurements
        self.dt = 1 / sample_rate

        # analysis of measurement data -> formation of admitance matrix
        file_names = list(self.meas_data.keys())

        acc_dict, f_dict, fr = mka.Fourier_transform(file_names, self.meas_data, force_window=True, min_max_ratio=100,
                                                  exp_window=False, exp_w_end=1e-2, show=False)

        # število mest meritev pospeškomerov, število mest udarcev
        split_list = list(map(lambda x: x.split('.')[0].split('_'), file_names))
        acc_no = max(list(map(lambda x: (int(x[1][3:]) - 1) // 3 + 1, split_list)))
        imp_no = max(list(map(lambda x: (int(x[0][1:]) - 1) // 3 + 1, split_list)))

        Y_exp_ = np.zeros((acc_dict[measurements[0]].shape[0], acc_no, imp_no), dtype=complex)
        for j in tqdm(measurements):
            name_split = j.split('.')[0].split('_')
            acc = int(name_split[1][3:])
            imp = int(name_split[0][1:])
            for chn in range(acc_dict[measurements[0]].shape[1]):
                if chn == 2:
                    if int((acc - 1) / 3) == int((imp - 1) / 3):
                        Y_exp_[:, int((acc - 1) / 3), int((imp - 1) / 3)] = mka.get_FRF(-1 * acc_dict[j][:, chn, :],
                                                                                        f_dict[j])
                    else:
                        Y_exp_[:, int((acc - 1) / 3), int((imp - 1) / 3)] = mka.get_FRF(acc_dict[j][:, chn, :],
                                                                                        f_dict[j])
                    if int((imp - 1) / 3) == 9 and int((acc - 1) / 3) != 9:
                        Y_exp_[:, 9, int((acc - 1) / 3)] = mka.get_FRF(acc_dict[j][:, chn, :], f_dict[j])

        #Y_exp = Y_exp_[fr_min:fr_max, :, :]
# ---------------------------------------------------------------------------------------------------------------------
# old
def get_interface_nodes(MK_model, x_limits, y_limit, err=0.5):
    """
    Function returns the nodes on the interface of substructures - for specific case
    :param MK_model: pyFBS.MK_model
    :param x_limits: 2 values
    :param y_limit: 1 value
    :param err: error
    :return: list of interface nodes
    """
    condition_x1 = (x_limits[0] - err < MK_model.nodes[:, 0]) & (MK_model.nodes[:, 0] < x_limits[0] + err) & (
                MK_model.nodes[:, 1] > y_limit + err)
    condition_x2 = (x_limits[1] - err < MK_model.nodes[:, 0]) & (MK_model.nodes[:, 0] < x_limits[1] + err) & (
                MK_model.nodes[:, 1] > y_limit + err)
    condition_y1 = (y_limit - err < MK_model.nodes[:, 1]) & (MK_model.nodes[:, 1] < y_limit + err) & (
                x_limits[0] - err < MK_model.nodes[:, 0]) & (MK_model.nodes[:, 0] < x_limits[1] + err)

    interface_nodes = list(set(np.where(condition_x1 | condition_x2 | condition_y1)[0]))

    return interface_nodes
