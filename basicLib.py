import os,sys
import math
import numpy as np
import h5py
import csv
import cmath
from stockwell import st
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft
import pywt

### Funcoes que geram as combinações para os rotulos dos sinais
#a matriz de combinação é a forma canonica de representar os dados
def genComb(v,i,my_one,N,K,max_p,l_comb): #funcao que gera a matriz de combinacoes de N aparelhos K a K
    if my_one < K:
        for j in range(max_p):
            v[i+j]=1
            genComb(v,(i+j+1),my_one+1,N,K,max_p-j,l_comb)
            v[i+j]=0
    else:
        for j in range(max_p):
            v[i+j]=1
            l_comb.append(np.array(v,dtype = np.float64))
            v[i+j]=0

def get_fft_values(y_values,N):
    #f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return fft_values

def combMatrix(min,N):#N representa o numero total de elementos a serem combinados
    comb_M = np.zeros((1,N),dtype = np.float64)
    l=[]
    v=[0 for j in range(N)]
    genComb(v,0,1,N,min,(N-min+1),l)
    comb_M = np.array(l)
    '''
    for i in range(min,N+1):
        l=[]
        v=[0 for j in range(N)]
        genComb(v,0,1,N,i,(N-i+1),l)

        comb_M=np.append(comb_M ,np.array(l,dtype = np.float64),0)
    comb_M=comb_M[1:]
    '''
    return comb_M

### Funcoes dde transformadas utilizadas nos sinais


def dft_n_coeff(v,n_coeff):
    N=v.shape[0]
    kvec=np.arange(N)
    angle = -1j*2*cmath.pi*kvec*n_coeff*(1/N)
    vec_e = np.exp(angle)
    A = v*vec_e
    dft_c =np.sum(A)/N
    return dft_c

def dft_transform(v):
    l=[]
    for i in range(v.shape[0]):
        coef=dft_n_coeff(v,i)
        l.append(coef)
    dft_sig=np.array(l)
    return dft_sig

def dft_ln_transform(v,ln):
    l=[]
    for i in range(ln.shape[0]):
        coef=dft_n_coeff(v,ln[i])
        l.append(coef)
    dft_sig=np.array(l)
    return dft_sig

def ift_n_coeff(v,n_coeff):
    N=v.shape[0]
    kvec=np.arange(N)
    angle = 1j*2*cmath.pi*kvec*n_coeff*(1/N)
    vec_e = np.exp(angle)
    A = v*vec_e
    dft_c =np.sum(A)
    return dft_c

def ift_transform(v):
    l=[]
    for i in range(v.shape[0]):
        coef=ift_n_coeff(v,i)
        l.append(coef)
    dft_sig=np.array(l)
    return dft_sig

#conjunto de transformada discreta S
def dt_st(sig):
    N = sig.shape[0]
    n_v = np.arange(1,N)
    m_v = np.arange(N)
    j_v = np.arange(N)
    n_m = np.arange((2*N-1))

    dft_coeffs_v = dft_ln_transform(sig,n_m)
    S_rows=[]

    S_rows.append(np.full(N,sig.mean(),dtype=complex ) )

    for n in n_v:
        index=m_v+n
        dft_v=dft_coeffs_v[index]
        H_coeffs = np.empty((N,N),dtype=complex)
        H_coeffs[:]=dft_v
        ###
        exp1 = -2*((cmath.pi**2)*(m_v**2)/(n**2))
        e1 = np.exp(exp1)
        e1M = np.empty((N,N))
        e1M[:]=e1

        exp2 = (2j*cmath.pi*m_v)/N
        exp2M = np.empty((N,N),dtype=complex)
        exp2M[:] = exp2
        jM=np.empty((N,N))
        jM[:]=j_v
        exp2M = exp2M.T*jM
        e2M = np.exp(exp2M)

        S_aux = H_coeffs.T*e1M.T*e2M
        row = np.sum(S_aux,0)
        S_rows.append(row)


    ST_matrix = np.vstack(S_rows)
    return ST_matrix

def dt_ist(SM):
    N = SM.shape[0]
    k_v = np.arange(N)
    n_v =np.arange(N)

    S_j = np.sum(SM,1)
    SM_j = np.empty((N,N),dtype=complex)
    SM_j[:] =S_j


    power_exp = (2j*cmath.pi/N)*n_v
    M_power = np.empty((N,N),dtype=complex)
    M_power[:]=power_exp
    K_index = np.empty((N,N))
    K_index[:]=k_v
    M_power = M_power.T*K_index
    EulerM = np.exp(M_power)

    prevM=SM_j.T*EulerM
    invS = np.sum(prevM,0)/N
    return invS
#conjunto de transformada T
def dt_tt(Sm):
    N = Sm.shape[0]
    n = np.arange(N)
    ME = np.empty((N,N),dtype=complex)
    l_rowTT=[]
    for k in range(N):
        angle = ((2*cmath.pi*n*k)/N)
        re = np.cos(angle,dtype = complex)
        im = np.sin(angle)
        re.imag = im
        #e_v = np.exp(((2j*cmath.pi*n*k)/N) )
        ME[:]=re
        auxTT=Sm*ME.T
        l_rowTT.append(np.sum(auxTT,0))
    TT_m = np.vstack(l_rowTT)
    return TT_m

def dt_itt(TT):
    sig = np.sum(TT,1)
    return sig

def feat_extract(sig):
    #SM = dt_st(sig)
    #TM =dt_tt(SM)
    size=sig.shape[0]

    SM =st.st(sig,0,size-1)

    SMA = np.abs(SM)
    N = SMA.shape[0]
    squared_SMA = SMA**2
    Tcc=[np.amax(SMA,0), np.amin(SMA,0), np.mean(SMA,0), np.std(SMA,0), np.sqrt((1/N)*np.sum(squared_SMA,0)) ]
    Fcc=[np.amax(SMA,1), np.amin(SMA,1), np.mean(SMA,1), np.std(SMA,1), np.sqrt((1/N)*np.sum(squared_SMA,1)) ]
    Tcc = np.vstack(Tcc)
    Fcc = np.vstack(Fcc)
    #Tdiag = TM.diag()

    staVec = np.reshape(SMA,(SMA.shape[0]*SMA.shape[1]))

    f_1=np.amax(np.amax(staVec)+np.amin(staVec))
    f_4 = np.std(staVec)

    f_6 = np.amax(Tcc[0])+np.amin(Tcc[0])
    f_19 = np.sqrt( (1/N)* np.sum(Tcc[2]**2))
    f_20 = np.amax(Tcc[3])+np.amin(Tcc[3])
    f_23= np.std(Tcc[3])


    f_35 = np.sqrt(np.sum(Fcc[1][1:]**2))/(Fcc[1][0])
    f_38 = np.mean(Fcc[1])
    f_39 = np.std(Fcc[1])
    f_40 = np.sqrt( (1/N)*np.sum(Fcc[1]**2) )
    f_41 = np.sqrt(np.sum(Fcc[2][1:]**2))/(Fcc[2][0])
    f_50 = np.std(Fcc[3])
    f_51 = np.sqrt((1/N)*np.sum(Fcc[3]**2 ))

    t_features=[f_1,f_4 ,f_6,f_19,f_20,f_23]
    f_features=[f_35,f_38,f_39,f_40,f_41,f_50,f_51]

    features = np.append(np.array(t_features),np.array(f_features))

    #features = np.append(Tcc,Fcc,0)
    #features = np.reshape(features,(features.shape[0]*features.shape[1]))

    return features

def getDset_features(dset):
    N = dset.shape[0]
    l_dset=[]
    for i in range(N):
        print("Index: ",i)
        feat=feat_extract(dset[i])
        #print(feat.shape)
        l_dset.append(feat)
    #featDset = np.vstack(dset_features)
    featDset=np.vstack(l_dset)
    return featDset

def plotSt(S,sig,t):
    S_abs = np.abs(S)
    #TT = dt_tt(S)
    #TT_abs = np.abs(TT)
    #tdiag = np.diag(TT_abs)
    N = S_abs.shape[0]
    extent = (t[0], t[-1], 0, N-1)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title("signal P")
    ax[0].plot(t, sig)
    ax[0].set(ylabel='amplitude')
    ax[1].set_title("Transformada S")
    ax[1].imshow(S_abs, origin='lower', extent=extent)
    #ax[1].matshow(s, origin='lower', extent=extent)

    ax[1].axis('tight')
    ax[1].set(ylabel='frequency (Hz)')
    #ax[2].set_title("Transformada T")
    #ax[2].plot(t,tdiag )
    #ax[2].set(xlabel='time (s)')


#977.299.498-49
