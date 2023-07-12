import pandas as pd
import numpy as np


def create_xlsx(nodes, name, VP=False, VP_positions=None, VP_chn_DoFs=None,
                VP_refchn_DoFs=None, direction=('x', 'y', 'z')):
    """
    Function creates the .xlsx file with sensor and impact data which is to be used in the pyFBS funtions.
    :param nodes: numpy array of node coordinates (rows - individual nodes, columns - x,y,z coordinates)
    :param name: name of the .xlsx file
    :param VP: if True the virtual point data is also created
    :param VP_positions: x,y and z coordinates of the virtual point
    :param VP_chn_DoFs: Virtual Point's degrees of freedom - channels
        3 translational (ux, uy, uz), 3 rotational (rx, ry, rz), 3 extensional (ex,ey,ez), 3 torsional (tx, ty, tz),
        6 skewing (sxy, sxz, syz, syx, szx, szy) (list)
    :param VP_refchn_DoFs: Virtual Point's degrees of freedom - reference channels (input channels)
        3 translational (fx, fy, fz), 3 rotational (mx, my, mz), 3 extensional (ex,ey,ez), 3 torsional (tx, ty, tz),
        6 skewing (sxy, sxz, syz, syx, szx, szy) (list)
    :param direction: list of axes of interest ('x', 'y', 'z')
    :return: none
    """
    if VP_refchn_DoFs is None:
        VP_refchn_DoFs = ['ux', 'uy', 'uz', 'rx', 'ry', 'rz']

    if VP_chn_DoFs is None:
        VP_chn_DoFs = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']

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
            sensors['Grouping'].append(VP_positions.index(tuple(nodes[i])) + 1)
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
                    channels['Grouping'].append(VP_positions.index(tuple(nodes[i])) + 1)
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
                    impacts['Grouping'].append(VP_positions.index(tuple(nodes[i])) + 1)
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
                    j['Grouping'].append(VP_positions.index(VP_position) + 1)
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
                    j['Direction_1'].append(directions[k % 3][0])
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
