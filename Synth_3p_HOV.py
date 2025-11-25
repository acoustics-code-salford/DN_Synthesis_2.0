"""
Created on Thu May  9 15:33:33 2024

SFR Synths From Recordings
Starts with Hovering recordings analysis. backpropagation  (at and distance).
Then, Tonal and broadcomponent retrivement and finally recostruction.

THE code includes backpropagation at 1m radious from source appliying the code by Marc.
# @author: SES271 Carlos Ramos-Romero
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import DSP_filters as dspd
import plots_matlab
plots_matlab
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as ss
from scipy.signal import find_peaks, medfilt
import scipy.io.wavfile as wav
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import soundfile as sf
import math
import random
from scipy.ndimage import filters
import resampy
from itertools import product

np.random.seed(143)
random.seed(143)
plt.close('all')
#%% --- Constants
Fs = 50_000
preffsqr = (2e-5)**2
Ndft    = 2**16#2**16
df      = Fs/Ndft
p_ref   = 20e-6
recordings_folder = 'base_recordings'
WINDOW = 'hann'
""" Choice of synthesis parameters
# VS_Signal = [AM_FACTOR, Q_m, max_beta, Q_f, time_var_m] # Variables for synthesis
# VS_Signal = [0.1, 4, 4, 4, 'aperiodic'] # Variables for synthesis symilar to the recording
"""
m_fac = [0.01, 0.3, 0.7]
Q_fac = [1 , 3.5]
bet_fac = [0.05, 1, 5]
func_fact = ['periodic', 'aperiodic']
cases_list = list(product(m_fac, Q_fac, bet_fac, func_fact))

# cases_list = [[0.3, 3.5, 1, 'aperiodic']] # Best case from previous analysis

for cl in range (len(cases_list)):
    VS_Signal = cases_list[cl] # Variables for synthesis
    # NOVERLAP = 2**8 #2**12
    #%% --- INPUT ACOUSTIC DATA from hovering
    # Filename = "Ed_M3_10_H00_N_C_nw_ev2_M5.wav"
    Drone_Recs = [#1500 ["wav file", [low_range,high_rnge], n_har, n_rotors, ee, number ob blades] 
                ["Ed_3p_10_HH_N_C_nw_ev2_M5.wav",[170,250], 10, 4, 0.05, 2]
                ]
    n_f = 0

    #White_bb =  dspd.filter_data_bp (White_noise, Fs, fc_hihg_pass=20, fc_low_pass = 20000)
    n_har = Drone_Recs[n_f][2] # number of harmonics
    n_blades = Drone_Recs[n_f][5] # number of blades

    n_harmonic_tones = list( np.linspace(1,n_har+1,n_har+1))
    n_harmonic_tones.insert(0, 0.5)

    Filename = Drone_Recs[n_f][0]
    Filename = os.path.join(recordings_folder, Filename)
    Fs, data_time = wav.read(Filename)

    '''   Back propagated signal will rewrite the original data'''
    position = (0, 0, 9) # x,y,z position of the drone to be backpropagated to.
    absorption_filter = dspd.AtmosphericAbsorptionFilter()
    data_time = absorption_filter.filter(data_time, position, depropa=True)

    Freqs_and_Amplitudes = []

    for k in n_harmonic_tones:
        
        peak_zone_file = Drone_Recs[n_f][1]
        
        # Fs, data_time = wav.read(Filename)
        
        data_t = dspd.filter_data_bp(data_time, Fs, fc_hihg_pass=20, fc_low_pass = 20_000)
        
        psd_data , F_data = dspd.calc_PSDs(data_t, Fs, Ndft, WINDOW,p_ref   = 20e-6)
        data_t = data_t[5*Fs:15*Fs]
        delta_f_psd = F_data[1] # PSD definition
        
        # plt.figure(figsize=(18,7))
            
        Chunk_L = 50_000 # In number of samples fro time windowind analysis
        
        BPFs_to_synth = [] # saving the bfs detected of each chunk
        Amp_BPFs_to_synth =[]
        
        data_t = data_t[0:round(data_t.shape[0]/Chunk_L)*Fs] #perfect chunk
        Sh_Ov = int(Chunk_L/2) # 50% shift due overlap
        
        
        s_ini = 0
        for nw in  tqdm(range(int(np.round(np.round(data_t.shape[0]/Chunk_L)*2-1))), colour="red"):
            s_fin = s_ini + Chunk_L
            chunk_data_t = data_t[s_ini:s_fin]
            psd_data_chunk , F_data_chunk = dspd.calc_PSDs(chunk_data_t  , Fs, Ndft, WINDOW,p_ref   = 20e-6)
            plt.semilogx(F_data_chunk, psd_data_chunk,color="#74a3c7") #skyblue lines one per second
            
            s_ini = s_fin-Sh_Ov
        
        #%% BPF detection
        
        # f_down and f_up related with each Filename
            
            lowBPF = peak_zone_file[0]*k
            higBPF = peak_zone_file[1]*k
            ini_fx = np.argmin(np.abs(F_data_chunk-lowBPF)) # index of F_data
            end_fx = np.argmin(np.abs(F_data_chunk-higBPF))
            QQ_data_chunk = psd_data_chunk[ini_fx:end_fx] #segmpent of spectrogram for looking for peaks
            
            height = medfilt(QQ_data_chunk, kernel_size=21)
            peaks, _ = find_peaks(QQ_data_chunk, height=height+7, prominence=4)#M3
            
            
            peaks = peaks + ini_fx #peaks index in all F_data chunk
            
            BPFs_detected = F_data_chunk[peaks][0:4]
            BPFs_to_synth.append(BPFs_detected)
            
            Amp_BPFs_detected = psd_data_chunk[peaks][0:4]
            Amp_BPFs_to_synth.append(Amp_BPFs_detected)
        
            plt.plot(BPFs_detected, Amp_BPFs_detected,'+r', label ='peaks') #each peak in red
        
        plt.semilogx(F_data, psd_data,label='recording',alpha =0.8, color = "#3619df")
        
        # plt.title(Filename[0:5]+f'BPFs*{k}')
        plt.ylabel('SPL [dB]')
        plt.xlabel('Frequency [Hz]')
        plt.xlim(10,4_000)
        plt.ylim(50,90)
        
        
        #plot harmonics
        # vli_down = min(BPFs_detected)
        # vli_up = max(BPFs_detected)
        # plt.vlines(x=vli_down, ymin=0, ymax=70, colors='gray',linestyles='--')
        # plt.vlines(x=vli_up, ymin=0, ymax=70, colors='#FFC0CB',linestyles='--')
        
        # plt.vlines(x=np.array(peak_zone_file)*k, ymin=0, ymax=70, colors='k',linestyles='-')
    #%% Parameters for FM from peakfinder
        # ---- IN FREQ
        All_BPF_detected = np.hstack([i.ravel() for i in BPFs_to_synth]) #squeeze the list compilation
        # ---- IN AMP
        All_Amp_BPFs_detected = np.hstack([i.ravel() for i in Amp_BPFs_to_synth]) #squeeze the list compilation
        
        n_rotors = Drone_Recs[n_f][3]
        fs_delta = dspd.GaussianMixModel_peaks(All_BPF_detected, n_components=n_rotors, \
                                            min_max=False, displot=False, magnitude ="Frequency [Hz]")
        amps_delta=[]
        indx_amp=[]
        for f_indx in range(len(fs_delta['Means'])):
            freq_Amp_indx = dspd.find_closest_index(F_data, fs_delta['Means'][f_indx]+F_data[1])
            indx_amp.append(freq_Amp_indx)
            
            amps_del = psd_data[freq_Amp_indx]/F_data[1]
            amps_delta.append(amps_del)

            
        # amps_delta = dspd.GaussianMixModel_peaks(All_Amp_BPFs_detected, n_components=n_rotors, \
        #                                           min_max=False, displot=False, magnitude ="Amplitude [dB]")
        
        F_A = [fs_delta['Means'], fs_delta['Deltas'], amps_delta] 
    # ALL the peak amplitudes andd frequencies    
        Freqs_and_Amplitudes.append(F_A) 

    BPFs_to_add = Freqs_and_Amplitudes[1][0]
    Delta_f_FM = Freqs_and_Amplitudes[1][1]
    AAdb_mean = Freqs_and_Amplitudes[1][2]

    #%% --- limit ti 5 seconds
    # data_t = data_t[1*Fs:8*Fs]

    #%% --- BROADBAND EXTRACTION
    # --- clean low-end to avoid spurius data at low frequencies< 50 Hz
    data_t_hp = dspd.filter_data_hp (data_t, Fs, filter_order=3,
                                            fc=10, btype='highpass') # HIghpass for cleaning the signal. cutoff 50Hz
    psd_data_hp , F_data = dspd.calc_PSDs(data_t_hp,Fs, Ndft, WINDOW, p_ref = 20e-6)

    #---- Notch filtering bank - based on each BPFs and their harmonics
    BPFs =  Freqs_and_Amplitudes[1][0]
    # BPFs = bpfs_delta['BPFs']

    n_har = n_har #10#15 # number of harmonics
    Q_ini = 28 #20 # sarting Q factor 
    Q_step = 8 #3 #steps in Q factor for higher frequency notch

    Q = list(np.arange(Q_ini, Q_ini + Q_step*n_har, Q_step))
    FF = (np.ones([n_har,len(BPFs)])*BPFs) # Array of [n_harmonics, BPFs]

    data_t_notched  = np.copy(data_t_hp)

    # Appying notch on EACH BPF
    for nh in range(n_har):
        for notchf in range(len(BPFs)):
            FF[nh, notchf] = BPFs[notchf] * (nh+1)
            data_t_notched = dspd.notch_filter(data_t_notched, BPFs[notchf]*(nh+1), Q[nh], Fs)   
            
    psd_data_notched , F_data = dspd.calc_PSDs(data_t_notched,Fs, Ndft, WINDOW, p_ref = 20e-6)

    #----Bb_noise Midfiltering for Pwelch of notch filtering 
    psd_data_notched_midfiltered = ss.medfilt(psd_data_notched , kernel_size = 101)#81best
     
    # If flyover operation sis simulated, BPFs will be different.
    # ---- BPFs
    # ---- CHange the BPFs when UAS moves forward.
    ope_synth = "hover"
    velo_uas = "00" # m/s
    print(BPFs_to_add)
    
    synth_FO = True
    if synth_FO == True:
        ope_synth = "flyover"
        velo_uas = "15" # m/s
        Freqs_and_Amplitudes[1][0] = [177*1.1, 243*1.1, 205*1.2, 222*1.2] #[177, 243, 205, 222]"""hovering BPFs"""
        Freqs_and_Amplitudes[0][0] = list(np.array(Freqs_and_Amplitudes[1][0])/n_blades)


    # %% --- ADITIVE Sinthesys
    # %%% ---- BB Noise method
    BB_method = 'white' #['white', 'xcor']

    if BB_method == 'white':
        #---- Color Noise from White Noise
        STD = np.max(data_t_hp)
        White_noise = 1 * np.random.normal(0, STD, size=len(data_t_hp))
        N = len(data_t_hp)
        t  = 1/Fs * np.arange(N)
        #f  = Fs/N * np.arange(N)
        
            # BANDPASS 40 - 10000
        White_bb =  dspd.filter_data_bp (White_noise, Fs, fc_hihg_pass=5, fc_low_pass = 20000)
        psd_White_bb , F_data = dspd.calc_PSDs(White_bb,Fs, Ndft, WINDOW,p_ref   = 20e-6)
        
        # Gain should be applied to white noise
        Gain_curve =  psd_data_notched_midfiltered - psd_White_bb 
        
        # Or Gain could be applied to recordings to compare
        Gain_curve_to_rec =  psd_data_notched_midfiltered - psd_data_hp
        
        #---- Equalization of RECORDED noise based on EQ curve
        Recorded_BB = dspd.EQ_of_Recorded_based_BBcurve (data_t_hp, np.max(data_t_hp)*10, Fs, Gain_curve_to_rec)
        psd_Recorded_BB , F_data = dspd.calc_PSDs(Recorded_BB,Fs, Ndft, WINDOW,p_ref = 20e-6)
        
        
        #---- Equalization of WHITE NOISE based on Color Noise mid ilter EQ curve from recording
        White_noise_EQ_BB, eq_curve_by_band, F_data_colorEQnoise = dspd.EQ_of_white_based_BBcurve_2 (White_bb, STD, Fs, Gain_curve)
        psd_White_noise_EQ_BB , F_data = dspd.calc_PSDs(White_noise_EQ_BB,Fs, Ndft, WINDOW,p_ref = 20e-6)
        
        BB_to_wav = "white_eq"     #"recorded" or "white_eq"
        BB_data = Recorded_BB if BB_to_wav == "recorded" else (White_noise_EQ_BB if BB_to_wav == "white_eq" else 0)
        # generate audio file  from white equilized or recording
        #sf.write(Filename[3:13]+f'{BB_to_wav}'+'.wav',  BB_data[Fs::], Fs,subtype='FLOAT') 
            
        
        #---- No-Modulated Low-Mid and Hi broad noise
        fc_modulation_limit = 1500 #10 * np.max([BPFs]) #HF broad band noise 10 times highest BPFs
        
        LM_BB_noise = 1 * dspd.filter_data_lp (White_noise_EQ_BB, Fs, filter_order=3,
                                            fc=fc_modulation_limit, btype='lowpass')
        psd_LM_BB_noise  , F_data = dspd.calc_PSDs(LM_BB_noise ,Fs, Ndft, WINDOW,p_ref = 20e-6)
        
        HF_BB_noise = 1 * dspd.filter_data_hp(White_noise_EQ_BB, Fs, filter_order=2,
                                        fc=fc_modulation_limit-300, btype='highpass')
        psd_HF_BB_noise  , F_data = dspd.calc_PSDs(HF_BB_noise ,Fs, Ndft, WINDOW,p_ref = 20e-6)
        
    # %%% ---- BB AMPLITUDE MODULATION
        # ---- Generate AM Modulator
        fm_AM_1 = sum(BPFs[0:2])/2#min(BPFs)#sum(BPFs) / len(BPFs)#BPFs[0] # frecuency of amplitud modulation BPF 1
        fm_AM_2 = sum(BPFs[2::])/2#max(BPFs)#BPFs[1] # frecuency of amplitud modulation BPF 2

        # AM_modulator = (np.sin(2 * np.pi * fm_AM_1*t) + np.sin(2 * np.pi * fm_AM_2 * t))/2 #lineal conbination of Bpf1 and BPF2
        #                                                                                 # to avoid overmoludation I tookthe average
        AM_modulator = (1.3*np.sin(2 * np.pi * fm_AM_1*t) + 1.3*np.sin(2 * np.pi * fm_AM_2 * t)+
                        0.2*np.sin(2 * np.pi * fm_AM_1*2*t) + 0.2*np.sin(2 * np.pi * fm_AM_2*2 * t)
                    
                    ) #lineal conbination of Bpf1 and BPF2
                    #0.4*np.sin( np.pi * fm_AM_1*t) + 0.4*np.sin( np.pi * fm_AM_2 * t)+
                                                                                    # to
    #---- Generate AMplitude modulation depth
        # time_var_m = "aperiodic" # tvf time variant function
        time_var_m = VS_Signal[3]
        slope_m = "none"
        Q_m = VS_Signal[1]
        # Q_m = 4
        AM_FACTOR = VS_Signal[0]
        # AM_FACTOR = 0.4
        AMdepth = dspd.GenAMdepth(t, fs=Fs, max_AMdepth=AM_FACTOR, time_var=time_var_m, Q=Q_m, slope=slope_m)

        #---- Modulate the signal
        # HF_modulated_BB_noise = dspd.AModulation(HF_BB_noise, AM_modulator, AMdepth)
        
        #---- Total BB
        GLfBB = 1
        GHfBB = 0.8
        Total_BB =   GLfBB * LM_BB_noise + GHfBB * HF_BB_noise # HF_modulated_BB_noise # 
        #---- Modulate the signal
        Total_BB = dspd.AModulation(Total_BB , AM_modulator, AMdepth)
        # sf.write(Filename[3:13]+f'BB_{time_var_m}{slope_m}'+'.wav',  Total_BB[Fs::], Fs,subtype='FLOAT')
        psd_Total_BB  , F_data = dspd.calc_PSDs(Total_BB ,Fs, Ndft, WINDOW,p_ref = 20e-6)
        
    plt.figure(figsize=(10,5)) 
    plt.semilogx(F_data, psd_data,label='recording',alpha =0.8, color ='blue')
    plt.semilogx(F_data, psd_Total_BB,label='BB',alpha =0.8, color ='black')
    plt.title(Filename[0:5]+f'BPFs*{k}')
    plt.ylabel('SPL [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.xlim(10,20000)
    plt.ylim(30,80)    
    # %%% --- TONAL components and FREQUENCY MODULATION
    # plt.close("all")

    # ---- CHange the BPFs when UAS moves forward.

    BPFs_to_add =  Freqs_and_Amplitudes[1][0]
    Delta_f_FM =  Freqs_and_Amplitudes[1][1]
    # print(BPFs_to_add)
    # print(Delta_f_FM)
    time_var_beta = np.copy(time_var_m) # linking am with fm
    slope_beta = np.copy(slope_m)
    
    # ---- Total SHAFT, BPFS mad HARMONICS

    # Q_f = 4 #1.5#15 works fine for both drones
    Q_f = VS_Signal[1] #equal to Q_m for be syncrinized btoh types of modulations

    HH_Tone_BPFs = []
    HH_Modulated_Tone_BPFs = []

    # ---- Generate modulated tone at HArmonics of BPFs 
    plt.figure(figsize=(7,3.5)) # all the harmonics contributions
    for i_har in range(len(Freqs_and_Amplitudes)):
        iH_Tone_BPFs=[]
        iH_Modulated_Tone_BPFs=[]
        
        # if i_har==0:
        #     fac=0.75
        # else:
        #     fac=1
        
        # ---- Generate tone related with rotors
        for n_bpf, bpf in enumerate(BPFs_to_add):
            
            #SINTHETIC ARMONICS BASED ON BPFs VALUES of HARMONIC ORDER h_ord
            SYNTH_TONES = True
            if SYNTH_TONES == True:
                h_ord = 1/n_blades if i_har == 0 else i_har #take the harmonic order bvalue
                F_h = Freqs_and_Amplitudes[1][0][n_bpf] * h_ord #harmonic frequency
            else:
                #SINTHETIC ARMONICS BASED PURELLY ON RECORDING
                F_h = Freqs_and_Amplitudes[i_har][0][n_bpf]
            
            del_f = Freqs_and_Amplitudes[i_har][1][n_bpf] #harmonic frequency variation
            A_h = Freqs_and_Amplitudes[i_har][2][n_bpf] #* fac #harmonic amplitude [dB]
            ph_r = random.uniform(0, 2*np.pi) #random phase for each rotor
            
            Abpf = 10**((A_h)/20)*(20e-6) # mean amplitudes of detected frecuency, in pascalls
            
            # max_beta = 4
            max_beta = VS_Signal[2]
            fm = 2.7*del_f/max_beta #3*( del_f/max_beta)# Hz 
            
            FM_FACTOR = del_f*2#del_f*2 # maximum beta factor
            # print(FM_FACTOR)
            FMbeta, max_Deltaf = dspd.GenFMbeta(t, fm = fm , fs = 50_000, max_Deltaf = FM_FACTOR, time_var=time_var_beta, Q=Q_f, slope=slope_beta)
            FM_modulator = FMbeta * np.cos(2*np.pi * fm * t)
            # %%% SPECTRAL BROADENING by phase modulation
            #percent RIZZY https://ntrs.nasa.gov/api/citations/20170005762/downloads/20170005762.pdf
            ee = Drone_Recs[n_f][4] / 10 ##ee
            PM = (2*np.pi*ee)*np.cos(2 * np.pi * fm *t)

            T_h = Abpf * np.sin(2*np.pi * F_h * t + ph_r) # Pure Tone
            T_h_mod =  0.5* Abpf * np.cos(2*np.pi * F_h * t + FMbeta * np.sin(2*np.pi* fm *t) + PM  + ph_r) #Tone modulated in frequency and rotor-phase
            iH_Tone_BPFs.append(T_h)
            iH_Modulated_Tone_BPFs.append(T_h_mod)
            
        # %%%    
        # ---- Total i_Harmonic signals
        Total_iH_Tone_BPFs = np.sum(np.array(iH_Tone_BPFs),axis=0)
        Total_iH_Modulated_Tone_BPFs = np.sum(np.array( iH_Modulated_Tone_BPFs),axis=0)

        psd_Total_iH_Tone_BPFs  , F_data = dspd.calc_PSDs(Total_iH_Tone_BPFs ,Fs, Ndft, WINDOW,p_ref = 20e-6)
        psd_Total_iH_Modulated_Tone_BPFs  , F_data = dspd.calc_PSDs(Total_iH_Modulated_Tone_BPFs ,Fs, Ndft, WINDOW,p_ref = 20e-6)
        # 
        HH_Tone_BPFs.append(Total_iH_Tone_BPFs)
        HH_Modulated_Tone_BPFs.append(Total_iH_Modulated_Tone_BPFs)
        
        # plt.semilogx(F_data,  psd_Total_iH_Modulated_Tone_BPFs)

    #     plt.ylabel('SPL [dB]')
    #     plt.xlabel('$f/f0$')
    #     plt.xlim(0.2,50)
    #     plt.ylim(50,90)
    # plt.tight_layout()
    # plt.savefig("F3_c_rev.svg", format="svg")
        
    # ---- Total H_Harmonic signals
    Total_HH_BPF = np.sum(np.array(HH_Tone_BPFs),axis=0)
    Total_HH_Modulated_Tone_BPFs = np.sum(np.array(HH_Modulated_Tone_BPFs), axis=0)    

    psd_Total_HH_BPF , F_data = dspd.calc_PSDs (Total_HH_BPF , Fs, Ndft, WINDOW,p_ref = 20e-6)
    psd_Total_HH_Modulated_Tone_BPFs , F_data = dspd.calc_PSDs (Total_HH_Modulated_Tone_BPFs , Fs, Ndft, WINDOW,p_ref = 20e-6)

    # 
    # %% Total Sinthesised Noise


    Total_Tonal = Total_HH_Modulated_Tone_BPFs.copy()
    # Total_Tonal.shape[0]/Fs

    GBB = 1
    GTT = 1

    Total_Synth_Noise =   1*(GBB * Total_BB + GTT * Total_Tonal)
    Total_Synth_Noise = dspd.filter_data_bp (Total_Synth_Noise, Fs, fc_hihg_pass=10, fc_low_pass = 20000) # HIghpass for cleaning the signal. cutoff 50Hz


    psd_Total_Synth_Noise, F_data = dspd.calc_PSDs(Total_Synth_Noise  ,Fs, Ndft, WINDOW,p_ref = 20e-6)

    New_wav_name = Filename[3:12]+'_AM_'+str(AM_FACTOR)+'_fmF_'+str(FM_FACTOR)+'_fm_'+str(fm)+'_Q_'+time_var_m+'_S_'+slope_m

    plt.figure(figsize=(7,3.5))
    plt.semilogx(F_data/98.3, psd_data_hp,label ='Recorded', color='blue')
    # plt.semilogx(F_data/98.3,psd_data_notched, label = 'Notched', color='gray')
    # plt.semilogx(F_data/98.3,  psd_data_notched_midfiltered ,':r',label='Median filter')
    # plt.semilogx(F_data/98.3, psd_White_noise_EQ_BB,label ='Filtered white-noise')
    # plt.semilogx(F_data, psd_Total_HH_BPF,label ='pure tones', color='black')
    #plt.semilogx(F_data, psd_Total_HH_Modulated_Tone_BPFs,label ='modulated tones', color='green')
    plt.semilogx(F_data/98.3, psd_Total_Synth_Noise, 'r',label ='total Synth')
    # plt.semilogx(F_data, psd_Total_HH_BPF,':',label ='Pure tones')
    # plt.semilogx(F_data/98.3, psd_Total_BB,label='Synth. Broadband noise', color ='green')

    # plt.semilogx(F_data/98.3, Gain_curve,':k' ,label='gains for Wnoise')
    # plt.semilogx(F_data/98.3,  psd_data_notched_midfiltered ,'--g',label='midfilter')

    # plt.title(Filename[0:5])
    plt.ylabel('SPL [dB]')
    plt.legend()
    # plt.xlabel('Frequency [Hz]')
    # plt.xlim(10,10000)
    plt.xlabel('$f/f0$')
    plt.xlim(0.2,50)
    plt.ylim(50,90)
    plt.tight_layout()
    # plt.savefig("FigBB_c_rev.svg", format="svg")

    # %%% Folder for saving audio files
    #########

    out_folder = 'Out_AUDIOS'
    out_subfolder = Filename[19:28]+f'_{ope_synth}_{velo_uas}_AM'+f'_{AM_FACTOR}'+'_Q'+f'_{Q_m}'+'_be'+f'_{max_beta}'+'_'+time_var_m
    ffolder = os.path.join(out_folder, out_subfolder)
    if not os.path.exists(ffolder):
        os.makedirs(ffolder)
        
    if synth_FO == False:
        """If synthesised signal is a hovering, propagation will be done here,
        If synthesised signal is a flyover, propagation will be done by Marc's code"""
        BB_signal = absorption_filter.filter(GBB * Total_BB[Fs::], position, depropa=False)    
        TT_signal = absorption_filter.filter(GTT * Total_Tonal[Fs::], position, depropa=False) 
        TOT_signal = absorption_filter.filter(Total_Synth_Noise[Fs::], position, depropa=False) 
        data_t = absorption_filter.filter(data_t_hp[Fs::], position, depropa=False) 
    else: 
        BB_signal = GBB * Total_BB[Fs::]   
        TT_signal = GTT * Total_Tonal[Fs::]
        TOT_signal = Total_Synth_Noise[Fs::]
        data_t = data_t_hp[Fs::]
        
    sf.write(ffolder +'/' + "BroadBand_"+ out_subfolder +'.wav', BB_signal, Fs,subtype='FLOAT')
    sf.write(ffolder +'/' + "Tonal_"+     out_subfolder +'.wav', TT_signal, Fs,subtype='FLOAT')
    sf.write(ffolder +'/' + "Total_"+     out_subfolder +'.wav', TOT_signal, Fs,subtype='FLOAT')
    sf.write(ffolder +'/' + "Recorded"+'.wav',  data_t, Fs,subtype='FLOAT')

    # Set the input folder, output folder, and the new sample rate
    downsample = False
    if downsample == True:
        n_fs = 48_000 #new sample rate
        input_folder = ffolder
        output_folder = ffolder+ '\\downsampled'
        # Process each WAV file in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith('.wav'):
                input_path = os.path.join(input_folder, filename)
        
                # Read the original wav file
                original_sample_rate, audio_data_synth = wav.read(input_path)
        
                # Downsample the audio data
                audio_data_synth_downsampled = resampy.resample(audio_data_synth, original_sample_rate, n_fs)
        
                # Save the downsampled audio to a new file
                # sf.write(ffolder +'/downsampled/'+filename   ,  audio_data_synth_downsampled, n_fs,subtype='FLOAT')
    # %%  --- PLOTS AM
    # Plot the original and modulated noise
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(t, HF_BB_noise)
    plt.title('HFBBnoise')
    plt.ylabel('Pressure (p)')

    plt.subplot(4, 1, 2)
    if not isinstance(AMdepth,  np.ndarray):
        plt.hlines(AMdepth, 0, t.max(), 'r')
    else: 
        plt.plot(t,AMdepth, 'r')
    plt.title('AM depth')
    plt.ylabel('$m$')

    plt.subplot(4, 1, 3)
    plt.plot(t, AM_modulator)
    plt.title(' AM Modulator')
    plt.ylabel('Amplitude env.')

    plt.subplot(4, 1, 4)
    plt.plot(t, HF_BB_noise)
    plt.title('Amp Modulated HFBBnoise')
    plt.ylabel('Pressure (p)')

    plt.tight_layout()
    # plt.show()
    # %%  --- PLOTS FM
    # Plot the original and modulated noise
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(t, Total_iH_Modulated_Tone_BPFs)
    plt.title('BPFs')
    plt.ylabel('Pressure (p)')

    plt.subplot(4, 1, 2)
    if not isinstance(FMbeta,  np.ndarray):
        plt.hlines(FMbeta, 0, t.max(), 'r')
    else: 
        plt.plot(t,FMbeta, 'r')
    plt.title('FMbeta')
    plt.ylabel(r'$\beta$')

    plt.subplot(4, 1, 3)
    plt.plot(t, FM_modulator)
    plt.title('FM Modulator')
    plt.ylabel('$\Delta fm$')

    plt.subplot(4, 1, 4)
    plt.plot(t, Total_HH_Modulated_Tone_BPFs)
    plt.title('Freq Modulated BPFs')
    plt.ylabel('Pressure (p)')

    # Manually share the x-axis among all subplots


    plt.tight_layout()
    # plt.show()

    # %%important to wav generation
    print(f'Q_m = {Q_m}')
    print(f'GBB = {GBB}')
    print(f'GLfBB = {GLfBB}')
    print(f'GHfBB = {GHfBB}')
    print(f'GTT = {GTT}')
    print(f'BPF_mean= {sum(BPFs) / len(BPFs)}')


    # # %%  --- PLOT m
    # # Create a figure and axis
    # from matplotlib.ticker import FuncFormatter
    # import matplotlib as mpl
    # mpl.rcParams['text.usetex'] = True
    # mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    # fig, ax1 = plt.subplots(figsize=(10, 5))

    # # Plot the AM depth on the first y-axis (left side)
    # line1, = ax1.plot(t[0:int(2.5*Fs)], AMdepth[0:int(2.5*Fs)], '--r', label=f'$m$ = {AM_FACTOR}')
    # # ax1.set_title('AM Depth and FM Modulation Index')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel(r'$m_{\text{var}}$', color='r') # AM depth y-axis label
    # ax1.tick_params(axis='y', labelcolor='r')  # Set y-axis tick color to match the line color
    # # Set 5 y-ticks for the first y-axis
    # # ax1.set_yticks(np.linspace(0, max(AMdepth[0:2.5*Fs]), 4))
    # # Format y-ticks to show two decimal places for the second y-axis
    # ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
    # ax1.grid(False)
    # # Turn off the grid

    # # Create a second y-axis sharing the same x-axis
    # ax2 = ax1.twinx()
    # line2, = ax2.plot(t[0:int(2.5*Fs)], FMbeta[0:int(2.5*Fs)], '-b', label=fr'$\beta$ = {max_beta}')
    # ax2.set_ylabel(r'$\beta_{\text{var}}$', color='b')  # FM modulation index y-axis label
    # ax2.tick_params(axis='y', labelcolor='b')  # Set y-axis tick color to match the line color
    # # Set 5 y-ticks for the second y-axis
    # ax2.set_yticks(np.linspace(0, max(FMbeta[0:int(2.5*Fs)]), 4))
    # # Format y-ticks to show two decimal places for the second y-axis
    # ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
    # ax2.grid(False)


    # # Make the plot frame (spines) thinner
    # for spine in ax1.spines.values():
    #     spine.set_linewidth(0.4)
    # for spine in ax2.spines.values():
    #     spine.set_linewidth(0.4)
    # # Combine legends from both axes into a single box
    # lines = [line1, line2]
    # labels = [line.get_label() for line in lines]
    # ax1.legend(lines, labels, loc='upper right')

    # # Adjust layout
    # fig.tight_layout()
    # plt.show()
    # plt.savefig("m_beta_timevar.svg", format="svg")