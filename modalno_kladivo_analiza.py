import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy import signal


def Fourier_transform(meas_data, dt, force_window=False, min_max_ratio=100, exp_window=False, exp_w_end=.01,
                      show=False, acc_chn=3, time_delay=0):
    """
    Fourierova transformacija meritev z modalnim kladivom.
    Meas_data dictionary form: {'key:file name':np.array(frekvence, kanali, ponovljene meritve)}

    - kanal 0: meritev sile
    - nadaljni kanali: meritev pospeškov

    :param file_names: keys under which individual data are stored inside the meas_data dictionary
    :param meas_data: dictionary with measurements
    :param dt: time interval between samples
    :param force_window: if True the force window is applied
    :param min_max_ratio: ratio of max. force value and indiviual measurement over which the force window is applied
    :param show: plot excitation forces
    :return:
    acc - dictionary of accelerations {'key-file name':np.array(frevence, kanali, ponovitve)}
    force - dictionary of forces {'key-file name':np.array(frevence, kanali, ponovitve)}
    fr - numpy array of discrete frequency points
    """
    # meritve pri udarcu z modalnim kladivom
    file_names = list(meas_data.keys())
    acc = {}
    force = {}
    for i in tqdm(file_names, leave=False):
        x = meas_data[i][:, 1:, :]
        f = meas_data[i][:, 0, :]
        if force_window:
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
            for l in range(acc_chn):
                fr = np.fft.rfftfreq(x.shape[0], d=dt)
                X[:, l, j] = np.fft.rfft(x[:, l, j])*np.exp(-1.j * 2 * np.pi * fr * time_delay)
            F[:, j] = np.fft.rfft(f[:, j])
        acc[i] = X
        force[i] = F


    return acc, force, fr


def get_FRF(X, F, filter_list=None, estimator='H1'):
    """
    Function calculates frequency response functions (FRF) from measurement data.
    :param X: np.array of accelerations (frequencies, repeated measurements)
    :param F: np.array of accelerations (frequencies, repeated measurements)
    :param filter_list: list of indices of measurements to be excluded from the FRF calculation
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
    for i in range(N):
        if estimator == 'H1':
            if filter_list != None:
                if i not in filter_list:
                    S_fx_avg += np.conj(F[:, i]) * X[:, i]
                    S_ff_avg += np.conj(F[:, i]) * F[:, i]
            else:
                S_fx_avg += np.conj(F[:, i]) * X[:, i]
                S_ff_avg += np.conj(F[:, i]) * F[:, i]
        elif estimator == 'H2':
            if filter_list != None:
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
        return S_fx_avg / S_ff_avg
    elif estimator == 'H2':
        return S_xx_avg / S_xf_avg


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
        #x_lim = (0, 2000)
        ax[0].semilogy(np.abs(frf[:]))
        ax[0].semilogy(eig_fr, np.abs(frf[eig_fr]), 'o')
        ax[1].plot(np.angle(frf[:]))
        ax[1].plot(eig_fr, np.angle(frf[eig_fr]), 'o')
        #ax[0].set_xlim(x_lim[0], x_lim[1])
        #ax[1].set_xlim(x_lim[0], x_lim[1])
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


# transformacije med odzivnimi veličinami
def transform_response(from_to, array, omega):
    """
    Transformacije med odzivi.
    """
    from_, to_ = from_to.split('->')
    if (from_=='acc' and to_=='mob') or (from_=='mob' and to_=='rec'):
        return acc_to_mob_to_rec(array, omega)
    elif from_=='acc' and to_=='rec':
        arr_ = acc_to_mob_to_rec(array, omega)
        return acc_to_mob_to_rec(arr_, omega)
    elif (from_=='rec' and to_ =='mob') or (from_=='mob' and to_ =='acc'):
        return rec_to_mob_to_acc(array, omega)
    elif from_=='rec' and to_=='acc':
        arr_ = rec_to_mob_to_acc(array, omega)
        return rec_to_mob_to_acc(arr_, omega)
    else:
        print('Invalid transform: check \'from_to\' value!')
        return


def acc_to_mob_to_rec(array, omega):
    """
    Integracija v frekvenčni domeni.
    """
    array_str = get_einsum_str(array)
    return np.einsum(array_str+','+array_str[0]+'->'+array_str, array, 1/(1.j*omega))


def rec_to_mob_to_acc(array, omega):
    """
    Odvod v frekvenčni domeni.
    """
    array_str = get_einsum_str(array)
    return np.einsum(array_str+','+array_str[0]+'->'+array_str, array, 1.j*omega)


def get_einsum_str(array):
    """
    Priprava string-a za numpy.einsum glede na velikost vhodne matrike.
    """
    signs = 'ijklmnopqr'
    return signs[:len(array.shape)]


def Ewins_Gleeson(FRF, freq, fr_min, fr_max, eigfr, N, reconstruct=False, return_points=False):
    Sigma = freq[np.linspace(0,len(freq[fr_min:fr_max]), N,dtype=int)]
    Alpha = FRF.real[np.linspace(0,len(freq[fr_min:fr_max]), N, dtype=int)]
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
    eigfr = np.zeros(len(fr_min_max_list))
    for j, i in enumerate(fr_min_max_list):
        eigfr[j] =freq[(i[0]<freq)&(i[1]>freq)][np.argmax(abs(FRF[(i[0]<freq)&(i[1]>freq)]))]
    return eigfr


def force_window(F_arr, t_w, sample_rate):
    """
    Function applies force window to the force signal array.
    Parameters
    ----------
    F_arr: Force signal array
    t_w: time at which window is applied
    sample_rate: sampling rate of force signal.

    Returns: windowed force signal
    -------

    """
    start_ind = int(np.ceil(t_w*sample_rate))
    F_arr_new = F_arr.copy()
    F_arr_new[start_ind:] = 0
    return F_arr_new

#def order_measurements(meas_dict):