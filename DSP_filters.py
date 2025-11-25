## -*- coding: utf-8 -*-
"""
DEfinitions for  BBNoise_Quad_N
Author CR.
"""
import numpy as np
import math
import scipy.signal as ss 
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from scipy import interpolate, signal
import seaborn as sns
import os
os.environ["OMP_NUM_THREADS"] = "1"
random.seed(10)
Ndft    = 2**14
plt.rcParams['figure.figsize'] = [4, 4]
# %% PSD and FILTERS
"""FFT and PSD"""
def calc_PSDs(signal,Fs, Ndft, WINDOW,p_ref   = 20e-6):#, Noverlap):
    """
    Returns the array of psd for one event on one raw data.

    Inputs
    ------
    signal : array imported from [pascals .mat]
    Fs: int sample rate

    Returns
    -------
    DATA_psd_events : : array [event, samples(freq), channel]
    freq_welch: array frequency vector 
    Notes
    ------
    """
    freq_welch, signal_PSD = ss.welch(signal, Fs, window=WINDOW, noverlap=Ndft // 4, #4 #2
                                      nperseg=Ndft, scaling='spectrum') # noverlap=Noverlap,
    
    signal_PSD = 10*np.log10((signal_PSD)/(p_ref**2)) #in SPL
       
    return signal_PSD, freq_welch

def highpass_filter(data, cutoff, Fs, order=4): # for avoiding low frequencies wich is contaminating my signals
    """
    Returns the data filterde by high-pass.

    Inputs
    ------
    data : array imported from [pascals .mat]
    Fs: int sample rate
    cutoff : frequency in [Hz].

    Returns
    -------
    filtered : filtered array on time [pascals]
    Notes
    ------
    """
    nyq = 0.5 * Fs
    b,a = ss.butter(order, cutoff/ nyq , btype='high', analog=False)
    #b, a = butter(order, [low, high], btype='bandpass')
    filtered = ss.lfilter(b, a, data)
    return filtered

def notch_filter(data, notchf, Q, Fs): # for filtering the first BPFs and harmonics
    """
    Returns the data filterde by noth filter.

    Inputs
    ------
    data : array imported from [pascals .mat]
    notchf : frequency in [Hz]
    Fs: int sample rate
    Q : factor notch

    Returns
    -------
    filtered : filtered array on time [pascals]
    Notes
    ------
    """

    b, a = ss.iirnotch(notchf, Q, Fs)
    filtered = ss.filtfilt(b , a , data)
    
    return filtered

""" high-pass filter """
def filter_data_hp (signal, Fs, filter_order=4, fc=50, btype='highpass'):
    my_filter = ss.butter(filter_order,fc, btype='highpass',output='sos',fs=Fs)
    """
    from FABIO
    Filter time-domain data at given filter order, cutoff frequency(ies),
    filter type, and overwrite result over original data. Uses Butterworth
    filter topology, and applies fwd-bkwd filtering.
    """
    
    pt_hp = ss.sosfiltfilt(my_filter, signal)
    return pt_hp



""" low-pass filter """
def filter_data_lp (signal, Fs, filter_order=4, fc=10000, btype='lowpass'):
    my_filter = ss.butter(filter_order,fc, btype='lowpass', output='sos',fs=Fs)
    """
    from FABIO
    Filter time-domain data at given filter order, cutoff frequency(ies),
    filter type, and overwrite result over original data. Uses Butterworth
    filter topology, and applies fwd-bkwd filtering.
    """
    
    pt_lp = ss.sosfiltfilt(my_filter, signal)
    return pt_lp

""" Bnad-pass filter """
def filter_data_bp (signal, Fs, fc_hihg_pass=10, fc_low_pass = 10000):
    
    HP_signal = filter_data_hp (signal, Fs, filter_order=3,
                                          fc=fc_hihg_pass, btype='highpass') # HIghpass for cleaning the signal. cutoff 50Hz
    pt_bp = filter_data_lp (HP_signal, Fs, filter_order=3,
                                          fc=fc_low_pass, btype='lowpass') # HIghpass for cleaning the signal. cutoff 10000Hz
    return pt_bp 

""" peak filter """
def filter_data_peak (signal, Fs, f0=100, Qfact = 30):
    # Design the peak filter
    b, a = ss.iirpeak(f0, Qfact, Fs)
    
    # Apply the filter to your signal
    
    pt_bpeak = ss.lfilter(b, a, signal)
    return pt_bpeak



def calc_Envelope (signal, Fs):
    analytic_signal = ss.hilbert(signal)
    envelope =  np.abs(analytic_signal) #The amplitude envelope is given by magnitude of the analytic signal.
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    f_envelope = (np.diff(instantaneous_phase) / (2.0*np.pi) * Fs)
    return envelope, f_envelope

# def bandpass_filter(data, lowcut, highcut, Fs, order):
#     """
#     Returns the data filterde by band pass.

#     Inputs
#     ------
#     data : array imported from [pascals .mat]
#     lowcut: frecuency in [Hz]
#     Highcut: frequency in [Hz]
#     Fs: int sample rate
#     order : order filter

#     Returns
#     -------
#     filtered : filtered array on time [pascals]
#     Notes
#     ------
#     """
#     nyq = 0.5 * Fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = signal.butter(order, [low, high], btype='bandpass')
#     filtered = lfilter(b, a, data)
#     return filtered

def spectrogram_plot (x_t, Fs, fl, fu):
    """
    Inputs
    ------
    x_t : data in time domain [pascals]
    lowcut: frecuency in [Hz]
    Highcut: frequency in [Hz]
    Fs: int sample rate
    order : order filter

    Returns
    -------
    filtered : filtered array on time [pascals]
    Notes
    ------

    """
    nperseg = (2**15)//6 #10990
    nfft = 2**15#2**18
    noverlap = int(nperseg//1.2)
    
    fig, (ax0) = plt.subplots(figsize=(7,3))
    f, t, Sxx = ss.spectrogram(x_t, nperseg=(2**15)//6, fs=Fs, nfft=nfft, noverlap=noverlap)
    plt.pcolormesh(t, f, 10*np.log10(Sxx/(20e-6)**2),vmin=10,vmax=60,cmap='viridis',shading='auto') # vmin=10,vmax=60 
     
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.ylim(fl,fu)
    cbar = plt.colorbar()
    cbar.set_label("SPL [dB/Hz]")
    plt.tight_layout()
    
    out = fig
    return out

def EQ_of_white_based_BBcurve (color_noise, Fs, eq_curve):
    """
    This definition equalizes a colored noise.
    Appliying a 1/3 octave-band filter bank.
    
    Inputs
    ------
    color_noise : white noise as vector [pascals]
    Fs : sample rate [sam/second]
    eq_curve : It is the frecuency profile [PSD amplitude]
    
    Returns
    ------
    color_noise_EQ_BB : color noise equalized 
    
    """
    Ndft= 2**16
    
    psd_color_noise , F_data = calc_PSDs(color_noise, Fs, Ndft= Ndft, WINDOW='hann',p_ref   = 20e-6) #freq and PSD amplitude of the input
    
    
    filtered_noise_by_band = [] # saving the noice filtered on each band
    ID_f_Gain = []
    f_index_to_search = np.arange(0, F_data.shape[0], 1)
    
    #---- central frecuencies for the 1/12-octave band filter bank NOMINAL
    #https://www.nti-audio.com/en/support/know-how/fractional-octave-band-filter
    fc= [ 20.5, 21.8, 23, 24.4, 25.9, 27.4, 29, 30.7, 32.5, 34.5, 36.5, 38.7, 41, 43.4,
         46, 48.7, 52, 55, 58, 61, 65, 69, 73, 77, 82, 87, 92, 97, 103 ,109, 115, 122, 
        130, 137, 145, 154, 163, 173, 183, 194, 205, 218 ,230, 244, 259, 274, 290, 307, 
         325, 345, 365, 387, 410, 434, 460, 487, 520, 550, 580, 610, 650, 690, 730, 770, 
         820, 870, 920, 970, 1030, 1090, 1150, 1220, 1300, 1370, 1450, 1540, 1630, 1730, 
       1830, 1940, 2050, 2180, 2300, 2440, 2590, 2740, 2900, 3070,  3250, 3450, 3650, 
         3870, 4100, 4340, 4600, 4870, 5200, 5500, 5800, 6100, 6500, 6900, 7300, 7700, 8200,
        8700, 9200, 9700, 10300, 10900, 11500, 12200, 13000, 13700, 14500, 15400, 16300,
         17300, 18300, 19400] # 20.5, 21.8, 23, 24.4, 25.9, 27.4, 29,
    #---- filter generation
    nyq = 0.5 * Fs
    
    for f0 in fc:
        
        freq_d = f0 / np.power(2, (1/2)*(1/12))
        freq_u = f0 * np.power(2, (1/2)*(1/12))
        low = freq_d / nyq
        high = freq_u / nyq
        sos= ss.butter(4, [low, high], btype='bandpass', output='sos')
        filtered_noise = ss.sosfilt(sos, color_noise)
        # fgain=1
        
        if f0 <205:
            fgain = 0.4
        elif 206 <= f0 <= 500:
            fgain = 0.8
        elif 501 <= f0 <= 770:
            fgain = 0.8
        elif 771 <= f0 <= 1300:
            fgain = 0.8
        elif 1301 <= f0 <= 3000:
            fgain = 0.8
        elif 3001 <= f0 <= 4000:
            fgain = 0.5
        elif 401 <= f0 <= 5000:
            fgain = 0.5
        else: 
            fgain=0.5 
        #print(f'f to search {f0}')    
        #---- Vector of Gaind for equalization
        # to find the nearest freq
        near_freq_index =  abs(F_data - f0)
        # Index of the frequency to appli the gain.
        ID = int(f_index_to_search[near_freq_index==min(near_freq_index)][0]) # idex ) in the case thee are two frequencies
        #print(f'index to search {ID}')
        ID_f_Gain.append(ID)
        
       # Gain_EQ = psd_color_noise[ID] - eq_curve[ID]  
        
        filtered_noise = eq_curve[ID]/10 * filtered_noise * fgain * 2 #eq_curve[ID]/10*np.log10((abs(filtered_noise[ID])/20e-6)) / 10 #2.5 
        
        #filtered_noise = (10**(Gain_EQ/20)*20e-6) * filtered_noise
        
        filtered_noise_by_band.append(filtered_noise)
        Ndft = 2**16
        BBnoise_sinth_psd, F_data_colorEQnoise  = calc_PSDs(filtered_noise, Fs, Ndft= Ndft, WINDOW='hann',p_ref   = 20e-6)
         
        #plt.plot(F_data, BBnoise_sinth_psd)
        
        
    color_noise_EQ_BB= np.sum(np.array(filtered_noise_by_band), axis=0)
    
    return color_noise_EQ_BB

def EQ_of_white_based_BBcurve_2 (White_bp, STD, Fs, Gain_curve):
    """
    This definition equalizes a colored noise.
    Appliying a 1/3 octave-band filter bank.
    
    Inputs
    ------
    White_bp : white noise as vector [pascals]
    STD: WHite noise STD
    Fs : sample rate [sam/second]
    Gain_curve : It is the frecuency profile [PSD amplitude]
    
    Returns
    ------
    color_noise_EQ_BB : color noise equalized 
    
    """
    Ndft = 2**16
    #Gain vectors from Gain_curve for pascals ration
    Eq_G = np.sqrt((10**(Gain_curve/10))*(20e-6)**2)
    
    psd_color_noise , F_data = calc_PSDs(White_bp, Fs, Ndft= Ndft, WINDOW='hann',p_ref   = 20e-6) #freq and PSD amplitude of the input 
    
    #Data vector from signal curve for pascal ratio
    Eq_D = np.sqrt((10**(psd_color_noise/10))*(20e-6)**2)
    
    filtered_noise_by_band = [] # saving the noice filtered on each band
    eq_curve_by_band = [] # saving the frequency response by band
    ID_f_Gain = []
    f_index_to_search = np.arange(0, F_data.shape[0], 1)
    
    # plt.figure(figsize=(7,3.5))
    #---- central frecuencies for the 1/12-octave band filter bank NOMINAL
    #https://www.nti-audio.com/en/support/know-how/fractional-octave-band-filter
    fc= [11,12,13,14,15,16,17,18, 20.5, 21.8, 23, 24.4, 25.9, 27.4, 29, 30.7, 32.5, 34.5, 36.5, 38.7, 41,
      43.4, 46, 48.7, 52, 55, 58, 61, 65, 69, 73, 77, 82, 87, 92, 97, 103, 109,
      115, 122, 130, 137, 145, 154, 163, 173, 183, 194, 205, 218, 230, 244, 259, 274, 290, 307, 
      325, 345, 365, 387, 410, 434, 460, 487, 520, 550, 580, 610, 650, 690, 730, 770, 820, 870,
      920, 970, 1030, 1090, 1150, 1220, 1300, 1370, 1450, 1540, 1630, 1730, 1830, 1940, 2050, 2180,
      2300, 2440, 2590, 2740, 2900, 3070, 3250, 3450, 3650, 3870, 4100, 4340, 4600, 4870, 5200, 5500,
      5800, 6100, 6500, 6900, 7300, 7700, 8200, 8700, 9200, 9700, 10300, 10900, 11500, 12200, 13000,
      13700, 14500, 15400,16300,17300, 18300, 19400]
    # fc = [21.1, 23.7, 26.6, 29.9, 33.5, 37.6, 42.2, 47.3, 53, 60, 67, 75,
    #         84, 94, 106, 119, 133, 150, 168, 188, 211, 237, 266, 299, 335, 376, 422,
    #         473, 530, 600, 670, 750, 840, 940, 1060, 1190, 1330, 1500, 1680, 1880,
    #         2110, 2370, 2660, 2990, 3350, 3760, 4220, 4730, 5300, 6000, 6700, 7500,
    #         8400, 9400, 10600, 11900, 13300, 15000, 16800, 18800]
    # plt.figure()
    #---- filter generation
    nyq = 0.5 * Fs
    
    DelF = F_data[1]
    for Eq_G_index, f0 in enumerate(fc):
        
        
        freq_d = f0 / np.power(2, (1/2)*(1/12))
        freq_u = f0 * np.power(2, (1/2)*(1/12))
        low = freq_d / nyq
        high = freq_u / nyq
        # sos = ss.butter(3, [low, high], btype='bandpass', output='sos')
        sos = ss.bessel(N=3,
                    Wn=[low, high],
                    btype='bandpass',
                    output='sos',
                    norm='phase') 
        filtered_noise = ss.sosfilt(sos, White_bp)
        
           
        #---- Vector of Gaind for equalization
        # to find the nearest freq
        near_freq_index =  abs(F_data - f0) 
        # Index of the frequency to appli the gain.
        ID = int(f_index_to_search[near_freq_index == min(near_freq_index)][0]) # idex ) in the case thee are two frequencies
        #print(f'index to search {ID}')
        fgain = Eq_G[ID]
        
        ID_f_Gain.append(ID)
        
       # Gain_EQ = psd_color_noise[ID] - eq_curve[ID]  
        
        filtered_noise =  fgain * filtered_noise * Ndft 
        
        #filtered_noise = (10**(Gain_EQ/20)*20e-6) * filtered_noise
        
        filtered_noise_by_band.append(filtered_noise)
        
        BBnoise_sinth_psd, F_data_colorEQnoise  = calc_PSDs(filtered_noise, Fs, Ndft= Ndft, WINDOW='hann',p_ref   = 20e-6)
        
        eq_curve_by_band.append(BBnoise_sinth_psd)
         
        # plt.semilogx(F_data, BBnoise_sinth_psd)
        # plt.xlabel('$f/f_0$')
        # plt.ylabel('Gain [dB]')
        # plt.xlim(0.2,50)
        # plt.ylim(-1,1)
        # plt.tight_layout()
        # plt.savefig("freqBBfilterbank.svg")
        # plt.show()
              
    color_noise_EQ_BB = np.sum(np.array(filtered_noise_by_band), axis=0)#*0.8 # 0.9 is a smooth gain factor
    color_noise_EQ_BB = filter_data_hp (color_noise_EQ_BB, Fs, filter_order=3, fc=10, btype='highpass') 
    return color_noise_EQ_BB, eq_curve_by_band, F_data_colorEQnoise

def EQ_of_Recorded_based_BBcurve (White_bp, STD, Fs, Gain_curve):
    """
    This definition equalizes a colored noise.
    Appliying a 1/3 octave-band filter bank.
    
    Inputs
    ------
    White_bp : white noise as vector [pascals]
    STD: WHite noise STD
    Fs : sample rate [sam/second]
    Gain_curve : It is the frecuency profile [PSD amplitude]
    
    Returns
    ------
    color_noise_EQ_BB : color noise equalized 
    
    """
    Ndft = 2**16
    #Gain vectors from Gain_curve for pascals ration
    Eq_G = np.sqrt((10**(Gain_curve/10))*(20e-6)**2)
    
    psd_color_noise , F_data = calc_PSDs(White_bp, Fs, Ndft= Ndft, WINDOW='hann',p_ref   = 20e-6) #freq and PSD amplitude of the input 
    
    #Data vector from signal curve for pascal ratio
    Eq_D = np.sqrt((10**(psd_color_noise/10))*(20e-6)**2)
    
    filtered_noise_by_band = [] # saving the noice filtered on each band
    ID_f_Gain = []
    f_index_to_search = np.arange(0, F_data.shape[0], 1)
    
    #---- central frecuencies for the 1/12-octave band filter bank NOMINAL
    #https://www.nti-audio.com/en/support/know-how/fractional-octave-band-filter
    fc= [6,10,15,18,20.5, 21.8, 23, 24.4, 25.9, 27.4, 29, 30.7, 32.5, 34.5, 36.5, 38.7, 41,
      43.4, 46, 48.7, 52, 55, 58, 61, 65, 69, 73, 77, 82, 87, 92, 97, 103, 109,
      115, 122, 130, 137, 145, 154, 163, 173, 183, 194, 205, 218, 230, 244, 259, 274, 290, 307, 
      325, 345, 365, 387, 410, 434, 460, 487, 520, 550, 580, 610, 650, 690, 730, 770, 820, 870,
      920, 970, 1030, 1090, 1150, 1220, 1300, 1370, 1450, 1540, 1630, 1730, 1830, 1940, 2050, 2180,
      2300, 2440, 2590, 2740, 2900, 3070, 3250, 3450, 3650, 3870, 4100, 4340, 4600, 4870, 5200, 5500,
      5800, 6100, 6500, 6900, 7300, 7700, 8200, 8700, 9200, 9700, 10300, 10900, 11500, 12200, 13000,
      13700, 14500, 15400,16300,17300, 18300, 19400]
    
    # fc= [10.0  , 10.6  ,11.2,  11.9,  12.6,  13.3,  14.1,  15.0,  15.8,  16.8,  17.8,  18.8,
    #      20.0 , 21.2, 22.4,  23.7,  25.2,  26.7,  28.3,  30.0,  31.8,  33.7,  35.6,  37.8,  
    #     40.0,  42.4,  44.8,  47.5,  50.3,  53.4, 56.6,  60.0,  63.5,  67.2,  71.0,  75.0,
    #     79.4,  84.1,  89.1,  94.4,  100,  106,  112,  119,  126,  133,  141,  150,  
    #     158,  168,  178,  188,  200,  212,  224,  238,  252,  267,  283,  300, 
    #     318,  337,  356, 378,  400,  424,  448,  475,  503,  534,  566,  600,  
    #     635,  672,  710,  750,  794,  841,  891,  944,  1000, 1060,  1120,  1190,  
    #     1260,  1330,  1410,  1500,  1580,  1680,  1780,  1880,  2000,  2120,  2240,  2380 , 
    #     2520,  2670,  2830,  3000,  3180,  3370,  3560,  3780,  4000,  4240,  4480,  4750,  
    #     5030,  5340,  5660,  6000,  6350,  6720, 7100,  7500,  7940,  8410,  8910,  9440,  
    #     10000,  10600, 11200,  11900,  12600,  13300,  14100,  15000,  15800,  16800,  17800,  18800,  20000]
    # fc = [21.1, 23.7, 26.6, 29.9, 33.5, 37.6, 42.2, 47.3, 53, 60, 67, 75,
    #         84, 94, 106, 119, 133, 150, 168, 188, 211, 237, 266, 299, 335, 376, 422,
    #         473, 530, 600, 670, 750, 840, 940, 1060, 1190, 1330, 1500, 1680, 1880,
    #         2110, 2370, 2660, 2990, 3350, 3760, 4220, 4730, 5300, 6000, 6700, 7500,
    #         8400, 9400, 10600, 11900, 13300, 15000, 16800, 18800]
    # plt.figure()
    #---- filter generation
    nyq = 0.5 * Fs
    
    DelF = F_data[1]
    for Eq_G_index, f0 in enumerate(fc):
        
        
        freq_d = f0 / np.power(2, (1/2)*(1/14))
        freq_u = f0 * np.power(2, (1/2)*(1/14))
        low = freq_d / nyq
        high = freq_u / nyq
        sos= ss.bessel(4, [low, high], btype='bandpass', output='sos')
        filtered_noise = ss.sosfilt(sos, White_bp)*0.5
        
           
        #---- Vector of Gaind for equalization
        # to find the nearest freq
        near_freq_index =  abs(F_data - f0) 
        # Index of the frequency to appli the gain.
        ID = int(f_index_to_search[near_freq_index == min(near_freq_index)][0]) # idex ) in the case thee are two frequencies
        #print(f'index to search {ID}')
        fgain = Eq_G[ID]
        
        ID_f_Gain.append(ID)
        
       # Gain_EQ = psd_color_noise[ID] - eq_curve[ID]  
        
        filtered_noise =  fgain * filtered_noise * Ndft 
        
        #filtered_noise = (10**(Gain_EQ/20)*20e-6) * filtered_noise
        
        filtered_noise_by_band.append(filtered_noise)
        
        BBnoise_sinth_psd, F_data_colorEQnoise  = calc_PSDs(filtered_noise, Fs, Ndft= Ndft, WINDOW='hann',p_ref   = 20e-6)
         
        # plt.semilogx(F_data, BBnoise_sinth_psd)
        # plt.xlabel('Frequency [Hz]')
        # plt.xlim(10,20000)
              
    color_noise_EQ_BB= np.sum(np.array(filtered_noise_by_band), axis=0)
    
    return color_noise_EQ_BB


# %% MODULATIONS
###
########
def AModulation(carrier, modulator, AMdepth):
    """
    This definition is the Aplication of amplitude modulation in the carring signal
    
    Parameters
    ----------
    carrier : SIgnal to be modulated : Array
    modulator : Signal that modulates : Periodic signal array
    AMdepth : Maximum AM depth : constant, periodic or aperiodical

    Returns
    -------
    AM_modulated_SIgnal : AMplitude Modulated siganl

    """
    AM_modulated_SIgnal = carrier * (1 + AMdepth*modulator)
    return AM_modulated_SIgnal

def GenAMdepth(t, max_AMdepth, Q, fs = 50000, time_var="constant", slope="none"):
    """
    This definition generate the AMdepth w/wo time variations 
    
    Parameters
    ----------
    t = time vector
    fs = sample rate in Hz
    max_AMdepth : Maximum AM depth: constant or array. 
                 max_AMdepth is limeted to 0.9 to ensure unmodulated signal and overmodulation (WhiteP) 
    time_var : string : "constant" or "periodic" or "exp+" or "exp-"
    Q :   variability rate in Hz (recomend low values) 
    slope: +, -, or none

    Returns
    -------
    AMdepth : Maximum AM depth : constant, periodic or aperiodical

    """

    #AMdepth = max_AMdepth # constant
    if max_AMdepth > 0.9: # to limit overmoldulation and variance 0 cases
        max_AMdepth = 0.9
    #print(max_AMdepth)
    slp = max_AMdepth/(t.shape[0]/fs)*t if slope== "+" else (max_AMdepth/(t.shape[0]/fs)*-t)+max_AMdepth if slope== "-" else 1 if slope == "none" else 0
    slp = slp/np.max(slp) # normalized [0 to  max_AMdepth] in length of t
    
    if time_var=="constant":
        AMdepth = slp * max_AMdepth
        
    elif time_var=="periodic":
        m_var = max_AMdepth + max_AMdepth * np.sin(2 * np.pi * Q * t) # periodic variations of m
        m_var = m_var + abs(np.min(m_var)) # variations [0, max_AMdepth ]
        AMdepth = slp * (m_var)*(max_AMdepth/np.max(m_var)) # normalized
        
    # elif time_var=="aperiodic":
    #     m_var = max_AMdepth + 0.5*max_AMdepth*(np.sin(2*Q*t) + np.sin(2 * np.pi * Q * t) + np.sin(2 * -np.e * Q * t)) # aperiodic variations of m
    #     m_var = m_var + abs(np.min(m_var))+0.2 # variations [0, max_AMdepth ]
    #     AMdepth = slp * (m_var) * (max_AMdepth/np.max(m_var))  # normalized 
        
    elif time_var=="aperiodic":
        m_var = (np.sin(2*Q*t) + np.sin(2 * np.pi * Q * t) + np.sin(2 * -np.e * Q * t)) # aperiodic variations of m or beta
        m_var = (m_var + abs(np.min(m_var)))/np.max(m_var + abs(np.min(m_var)))  # normalized
        AMdepth = slp * max_AMdepth* (m_var) 
    
    elif time_var == "sigmoid+":
        lt =( t.shape[0]/fs)/4
        m_var = max_AMdepth/(1+pow(math.e,math.e*(-t+(max_AMdepth-lt+math.e))))
        m_var = m_var + abs(np.min(m_var))
        AMdepth = slp * (m_var) * (max_AMdepth/np.max(m_var))  # normalized 
        
    elif time_var == "sigmoid-":
        lt =( t.shape[0]/fs)/4
        m_var = -max_AMdepth/(1+pow(math.e,math.e*(-t+(max_AMdepth-lt+math.e))))
        m_var = m_var + abs(np.min(m_var))
        AMdepth = slp * (m_var) * (max_AMdepth/np.max(m_var))  # normalized
        
    return AMdepth

def GenFMbeta(t, Q, fm=50, fs = 50000,  max_Deltaf=300, time_var="constant", slope="none"):
    """
    This definition generate the FM modulation index beta w/wo time variations 
    
    Parameters
    ----------
    fm = modulation frequency
    fs = sample rate in Hz
    max_Deltaf : Maximum Delta freq. It is peak frequency deviation: constant or array. 
                 max_Deltaf is limeted to 1.5 to ensure unmodulated signal and overmodulation (WhiteP) 
    time_var : string : "constant" or "periodic" or "exp+" or "exp-"
    Q :   variability rate in Hz (recomend low values) 
    slope: +, -, or none

    Returns
    -------
    AMdepth : Maximum AM depth : constant, periodic or aperiodical

    """

    #max_Deltaf # constant
    if max_Deltaf > 700: # to limit overmoldulation and variance 0 cases
        max_Deltaf = 700
    max_FMbeta = max_Deltaf/fm # definition in Lawrence
    #print(max_FMbeta)
    
    slp = max_FMbeta/(t.shape[0]/fs)*t if slope== "+" else (max_FMbeta/(t.shape[0]/fs)*-t)+max_FMbeta if slope== "-" else 1 if slope == "none" else 0
    slp = slp/np.max(slp) # normalized [0 to  max_AMdepth] in length of t
    
    if time_var=="constant":
        FMbeta = slp * max_FMbeta
        
    elif time_var=="periodic":
        m_var = max_FMbeta + max_FMbeta * np.sin(2 * np.pi * Q * t) # periodic variations of m  
        m_var = m_var + abs(np.min(m_var)) # variations [0, max_AMdepth ]
        FMbeta = slp * (m_var)*(max_FMbeta/np.max(m_var)) # normalized
        
    # elif time_var=="aperiodic":
    #     m_var = max_FMbeta + max_FMbeta*(np.sin(2*Q*t) + np.sin(2 * np.pi * Q * t) + np.sin(2 * -np.e * Q * t)) # aperiodic variations of m
    #     plt.figure()
    #     plt.plot(m_var)
    #     m_var = m_var + abs(np.min(m_var))+0.2 # variations [0, max_AMdepth ]
    #     FMbeta = slp * (m_var) * (max_FMbeta/np.max(m_var))  # normalized
    #     # Plot the original and modulated noise
    
    elif time_var=="aperiodic":
        m_var = (np.sin(2*Q*t) + np.sin(2 * np.pi * Q * t) + np.sin(2 * -np.e * Q * t)) # aperiodic variations of m or beta
        m_var = (m_var + abs(np.min(m_var)))/np.max(m_var + abs(np.min(m_var)))  # normalized
        FMbeta = slp * max_FMbeta* (m_var) 
        
        
        # np.max((np.sin(2*Q*t) + np.sin(2 * np.pi * Q * t) + np.sin(2 * -np.e * Q * t)))
        # plt.figure()
        # plt.plot(m_var)
        # m_var2 = m_var + abs(np.min(m_var)) # variations [0, max_AMdepth ]
        # plt.plot(m_var2)
        # FMbeta = slp * (m_var) * (max_FMbeta/np.max(m_var))  # normalized
        # plt.plot(FMbeta)
        # Plot the original and modulated noise
    
    elif time_var == "sigmoid+":
        lt =( t.shape[0]/fs)/4
        m_var = max_FMbeta/(1+pow(math.e,math.e*(-t-Q/fs-lt+math.e)))
        m_var = m_var + abs(np.min(m_var))
        FMbeta = slp * (m_var) * (max_FMbeta/np.max(m_var))  # normalized 
        
    elif time_var == "sigmoid-":
        lt =( t.shape[0]/fs)/4
        m_var = -max_FMbeta/(1+pow(math.e,math.e*(-t-Q/fs-lt+math.e)))
        m_var = m_var + abs(np.min(m_var))
        FMbeta = slp * (m_var) * (max_FMbeta/np.max(m_var))  # normalized
    
    return FMbeta, max_Deltaf

def GaussianMixModel_peaks (data_in, n_components=4, min_max=False, displot=True, magnitude="magnitude"):
    """
    This defitinion gets the mean and standar deviations on a 1D data set.
    It also includes the displot.

    Parameters
    ----------
    data_in : 1D data. array
    n_components : gaussian components , int
    min_max : booloan. if True, min and max values from means are retured instead of all the n components
    displot : boolean. If True, distribution plot is presented
    magnitude : string. Is the magnitude od the 1D array

    Returns
    -------
    Means_Deltas: retursn the dictionary organized as 
                    {'Means' : means,
                    'Deltas':standard_deviations}

    """
    
    
    gmm = GaussianMixture(n_components = n_components,random_state=31)
    gmm.fit(data_in.reshape(-1, 1)) #inF Reuqency
    
    
#   # MEANS and STD based on n_components
    means = list(np.squeeze(gmm.means_))
    standard_deviations = list(np.squeeze(gmm.covariances_**0.5)) 
    
    Means_Deltas = {'Means' : means,
                    'Deltas':standard_deviations}# dictionary 
    
    if min_max==True:
        minBPF = means.index(min(means))
        maxBPF = means.index(max(means))

        Means_Deltas = {'Means' : [means[minBPF],means[maxBPF]],
                        'Deltas':[standard_deviations[minBPF],standard_deviations[maxBPF]]}# dictionary

    # Plot the histogram of detected BPFs
    if displot==True:
        Dis_plot_F = sns.displot(data_in, kind="kde")

        plt.vlines(x=Means_Deltas['Means'], ymin=0, ymax=0.2, colors='k',linestyles='--')

        plt.xlabel(f'{magnitude}')
        plt.ylabel('Density')
        plt.title('Histogram of detected peaks in region X axis')
        plt.tight_layout()
        # plt.xlim(70,110)
        plt.show()
        
    return Means_Deltas

def find_closest_index(vector, target_value):
    # Ensure vector is a NumPy array
    vector = np.asarray(vector)
    
    # Calculate absolute differences
    diff = np.abs(vector - target_value)
    
    # Find index of minimum difference
    closest_idx = diff.argmin()
    
    return closest_idx


# %% Dedoppler effect adapted from Eric Grenwood - Penstate University
# ##########################################################################
def THdedop (time,pressure,a0,ttrack,Xtrack,Ytrack,Ztrack,Xmic,Ymic,Zmic,dfs,**kwargs):
    
    """
    %THdedop - time history de-Dopplerization of aircraft noise

    THdedop de-Dopplerizes pressure time history signals from stationary bservers
    for a moving source.

    The correction may optionally be applied to transform the signals to those
    measured by virtual observers traveling on the surface of a hemisphere set a
    fixed distance away from the tracking location, including the amplitude
    correction due to sphereical spreading.All inputs and outputs should be provided
    in any consistant set of dimensional coordinates.

    Parameters
    ----------
    time      NxS matrix of time of reception of ground based microphones
    pres      NxS matrix of acoustic pressures associated with time
    a0        speed of sound
    ttrack    vector of time associated with tracking data
    Xtk       tracking data X coordinate
    Ytk       tracking data Y coordinate
    Ztk       tracking data Z coordinate
    Xm        Size N vector of microphone X locations
    Ym        Size N vector of microphone Y locations
    Zm        Size N vector of microphone Z locations
    dfs       Desired sample rate of de-Dopplerized data
    radius    (optional) radius of virtual observers

    Returns
    -------
    dtime     emission or virtual observer reception time
    dpres     de-Dopplerized pressure or observer corrected pressure
    """
    mics = 1
    radius = kwargs.get('radius', None)
    #Calculate uniformly sampled emission time for assumed source location
    dtime = np.arange(ttrack[0],ttrack[-1],1/dfs)
    dpres = np.zeros((mics,dtime.size));
    
    for i in range(mics):
        #Calculate linear propagation distances from assumed source location
        r2 = interpolate.interp1d(ttrack,np.sqrt((Xtrack-Xmic[i])**2 +\
                                                    (Ytrack-Ymic[i])**2 +\
                                                        (Ztrack-Zmic[i])**2),
                                        kind='cubic',fill_value="extrapolate")(dtime)
        #Transform pressure to emission time vector
        dpres = interpolate.interp1d(time,pressure,kind='cubic',fill_value="extrapolate")(dtime+r2/a0)
        #correct for spherical spreading to virtual observers
        if radius:
            dpres = (r2/radius)*dpres
        if radius:
            #Adjust time of emission to time of reception on sphere surface
            dtime = dtime+radius/a0
        
    return dtime, dpres

## %% Atmospheric absorption and distance for backpropagation of the recordings
############################################################################
class AtmosphericAbsorptionFilter():
    '''
    Implements distance-dependent lowpass filter to simulate
    atmospheric absorption.
    '''
    def __init__(self,
                 freqs=np.geomspace(20, 24000),
                 temp=20.0,
                 humidity=80.0,
                 pressure=101.325,
                 n_taps=21,
                 fs=48_000,
                 depropa=False):
        '''
        Initialises GroundReflectionFilter.

        '''
        self._attenuation = self._alpha(freqs, temp, humidity, pressure)
        self.n_taps = n_taps
        '''Number of taps for FIR filter'''
        self.freqs = freqs
        '''array of frequencies used to evaluate frequency response'''
        self.fs = fs
        '''Sampling frequency [Hz]'''
        self.depropa = depropa
        '''if depropagation is applied, filters arecompensated'''


    def _alpha(self, freqs, temp=20, humidity=80, pressure=101.325, depropa=False):
        '''Atmospheric absorption curves calculated as per ISO 9613-1'''
        # calculate temperatre variables
        kelvin = 273.15
        T_ref = kelvin + 20
        T_kel = kelvin + temp
        T_rel = T_kel / T_ref
        T_01 = kelvin + 0.01

        # calculate pressure variables
        p_ref = 101.325
        p_rel = pressure / p_ref

        # calculate humidity as molar concentration of water vapour
        C = -6.8346 * (T_01 / T_kel) ** 1.261 + 4.6151
        p_sat_by_p_ref = 10 ** C
        h = humidity * p_sat_by_p_ref * p_rel

        # calcuate relaxataion frequencies of atmospheric gases
        f_rO = p_rel * (
            24 + 4.04e4 * h * (0.02 + h) / (0.391 + h)
        )

        f_rN = p_rel / np.sqrt(T_rel) * (
            9 + 280 * h * np.exp(-4.17 * (T_rel ** (-1/3) - 1))
        )

        # calculate alpha
        xc = 1.84e-11 / p_rel * np.sqrt(T_rel)
        xo = 1.275e-2 * np.exp(-2239.1 / T_kel) * (
            f_rO + (freqs**2 / f_rO)) ** (-1)
        xn = 0.1068 * np.exp(-3352 / T_kel) * (
            f_rN + (freqs**2 / f_rN)) ** (-1)

        alpha = freqs**2 * (xc + T_rel**(-5/2) * (xo + xn))
        
        if depropa==True:
            alpha = -1*alpha
            
        return 1 - alpha

    def filter(self, x, position, depropa):
        _, _, r = position
        '''For Back propagation _inv_sqr_attn = 1 * r**2 ; For Propagation  _inv_sqr_attn1 / r**2 
        line250: https://github.com/acoustics-code-salford/uas-sound-propagation/blob/main/uasevent/environment.py'''
        _inv_sqr_attn = 1 / r**2
        
        if depropa==True:
            _inv_sqr_attn = 1 * r**2 # For Back propagation 1 * r**2 # For Propagation 1 / r**2

        x = x * _inv_sqr_attn
        
        h = signal.firls(self.n_taps, self.freqs,
                         self._attenuation**r,
                         fs=self.fs)

        return signal.fftconvolve(x, h, 'same')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    