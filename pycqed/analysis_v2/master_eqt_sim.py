# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:54:50 2019

@author: Kerstin
"""


get_ipython().run_line_magic('matplotlib', 'inline')





from IPython.display import Image





#Import
import os
import h5py
import matplotlib as plt 
import numpy as np 
import scipy as sp
import time
import qutip as qt

"""
The goal of this file is to define some important functions to simulate the result 
of master equation when the sequence of manipulation if given.
"""

#Here I define the time evolution for N-qubits.
#T1 is the life time of each qubit
#T2s is the dephasing rate of each qubit
#N is the number of qubits. Initial N = 0
N = 0 #N =4 right now. The concrete number of qubits should be defined the the function compare_3() in the file tda_mes


def time_evole(H, t_gate, rho_sim, T1, T2s, N):
    """
    We define the time evolution in which c_ops != 0 (non-ideal time evolution)
    """
    #First we initial the list of c_ops: c_ops_1 is related to Lifetime of qubits and _ops_2 is related to Dephasing of qubits
    c_ops_1 = []
    c_ops_2 = []
    #Second we define the 2 types of N N-dimension tensors using 2 Recursion which will fullfil c_ops_1/2
    for j in range(N):
        if j == 0:
            tensor_1 = qt.destroy(2)
            tensor_2 = qt.sigmaz()
        else:
            tensor_1 = qt.identity(2)
            tensor_2 = qt.identity(2)
        for i in range(1,N):
            if i == j:
                tensor_1 = qt.tensor(tensor_1, qt.destroy(2))
                tensor_2 = qt.tensor(tensor_2, qt.sigmaz())
            else:
                tensor_1 = qt.tensor(tensor_1, qt.identity(2))
                tensor_2 = qt.tensor(tensor_2, qt.identity(2))
        c_ops_1.append( np.sqrt(1/T1[j])*tensor_1 )
        c_ops_2.append( np.sqrt((1/T2s[j] - 1/(2*T1[j]))/2)*tensor_2 )
    #Now we add the 2 c_ops to the total one
    c_ops = c_ops_1 + c_ops_2
    e_ops = [] #No expectation values are we are interested in the final state
    tlist = np.linspace(0,t_gate,100) #We solve for these times
    output = qt.mesolve(H, rho_sim, tlist, c_ops, e_ops, qt.Options(store_states=True)) #here we solve master equation
    return output.states[-1] #Output final state of rho_sim



def time_evole_ideal(H, t_gate, rho_ideal, T1, T2s, N):
    """
    Now, we define the ideal time evolution where c_ops = 0 (no noise)
    """
    c_ops = [] #No collapse
    e_ops = [] #No expectation values are we are interested in the final state
    tlist = np.linspace(0,t_gate,100) #We solve for these times
    output = qt.mesolve(H, rho_ideal, tlist, c_ops, e_ops, qt.Options(store_states=True)) #here we solve master equation
    return output.states[-1] #Output final state of rho_ideal




def rho_0(N):
    """
    Now we make a Bell State starting in the |0> state for N qubits.
    Initial state we want: rho = rho_ideal = rho_0
    """
    qb1_state = qt.basis(2,0) #Start in the 0 state of the first qubit
    qb1_state = qb1_state * qb1_state.dag() #Make to density matrix
    rho_0 = qb1_state
    #Adding the next (N-1) qubits to rho
    for i in range(1,N):
        qb_state = qt.basis(2, 0)
        qb_state = qb_state * qb_state.dag()
        rho_0 = qt.tensor(rho_0, qb_state) #Bell State |00...0><00...0|
    return rho_0





#Def 0: We define a dictionary to save all our time - parameters: gate times for all gates, waiting times for before/after two-qubit gates etc
times = {}

#Unit of time is us
T1 = [] #Lifetime of the qubits
T2s = [] #Dephasing rates of qubits. We must have T2s[i] < 2*T1[i]
t_xy = 0.040 #All Y and X gates: 40 ns 
t_before = 0.100 #Waiting time before 2 qubits interaction in CZ manipulation
t_cz =  0.105 #Controlled-phase gate time is 105 ns in CZ manipulation
t_after = 0.030 #Waiting time after 2 qubits interaction in CZ manipulation

times['T1'] = T1
times['T2s'] = T2s
times['t_xy'] = t_xy
times['t_before'] = t_before
times['t_cz'] = t_cz
times['t_after'] = t_after



#Def 1:We define the 5 sorts of important matrices
def gates(N):
    '''
    We define the useful tensorsx,y,z for Axis rotation manipulation and identity matrice for relax manipulation
    We save the four sorts matrix in gates function
    '''
    #we define the 3N matrices tensor tensorsx[j] =: 1*1*...*0.5sigmax*...*1 We generate 3 lists to save them
    #These matrixes are important for axis rotation
    tensorsx = []
    tensorsy = []
    tensorsz = []
    for j in range(N):
        if j == 0:
            tensorsx.append(qt.sigmax()/2)
            tensorsy.append(qt.sigmay()/2)
            tensorsz.append(qt.sigmaz()/2)
        else:
            tensorsx.append(qt.identity(2))
            tensorsy.append(qt.identity(2))
            tensorsz.append(qt.identity(2))
        for i in range(1,N):
            if i ==j:
                tensorsx[j] = qt.tensor(tensorsx[j], qt.sigmax()/2 )
                tensorsy[j] = qt.tensor(tensorsy[j], qt.sigmay()/2 )
                tensorsz[j] = qt.tensor(tensorsz[j], qt.sigmaz()/2 )
            else: 
                tensorsx[j] = qt.tensor(tensorsx[j], qt.identity(2) )
                tensorsy[j] = qt.tensor(tensorsy[j], qt.identity(2) )
                tensorsz[j] = qt.tensor(tensorsz[j], qt.identity(2) )
    #we define the identity matrix 1*1*...*1 :=identity
    #This matrix is important for relax process
    identity = qt.identity(2)
    for j in range(1,N):
        identity = qt.tensor(identity, qt.identity(2))
    return tensorsx, tensorsy, tensorsz, identity
            
def interact(value):
    '''
    We define a function which can generate interaction matrix between any 2 qubits, interact(value) := 1*...*|1><1|*...*|1><1|*...1
    These matrices are important for CZ manipulation, for example: 'CZ qb4 qb1'
    '''
    #Extract the indexes of the 2 interaction qubits
    p = int(value[-1]) -1 
    q = int(value[-5]) -1
    #We arrange p and q so that p<q
    if q<p:
        p, q = q, p
    #We define z := |1><1| state
    e = qt.basis(2,1)
    z = e * e.dag()
    #We start to generate the interaction matrix
    if p == 0:
        interact = z
    else:
        interact = qt.identity(2)
    for i in range(1, N):
        if i == q or i == p:
            interact = qt.tensor(interact, z)
        else:
            interact = qt.tensor(interact, qt.identity(2))
    return interact


#Def 2: We define the 4 sorts of important manipualtion: CZ and 3 Axis rotations
def cz(value, rho_sim, rho_ideal, times, N):
    '''
    This function calculate rho_sim and rho_ideal after CZ manipulation between any 2 qubits.
    CZ: relax 100 ns + interaction between 2 qubits + relax 30 ns
    '''
    #Time-parameters are in the dictionary times{}
    T1 = times['T1']
    T2s = times['T2s']
    t_xy = times['t_xy']
    t_before = times['t_before']
    t_cz = times['t_cz']
    t_after = times['t_after']
    
    #relax: we wait for 100 ns with no control applied. t_before = 100ns
    amp = 0
    H = amp * gates(N=N)[3] #The density matrix of N identity(2) states has defined above as identity 
    rho_sim = time_evole(H, t_gate=t_before, rho_sim=rho_sim, T1=T1, T2s=T2s, N=N )
    rho_ideal = time_evole_ideal(H, t_gate=t_before, rho_ideal=rho_ideal, T1=T1, T2s=T2s, N=N)
    #interaction: Controlled phase gate for any 2 qubits. t_cz = 105ns
    amp = np.pi/(t_cz) #Amplitude needed for a picking up a phase of -1 in time t_gate
    H = amp * interact(value=value) #We call the "interact" function to generate interaction matrix
    rho_sim = time_evole(H, t_gate=t_cz, rho_sim=rho_sim, T1=T1, T2s=T2s, N=N) 
    rho_ideal = time_evole_ideal(H, t_gate=t_cz, rho_ideal=rho_ideal, T1=T1, T2s=T2s, N=N)
    #relax 30 ns. t_after = 30ns
    amp = 0
    H = amp * gates(N=N)[3]
    rho_sim = time_evole(H, t_gate=t_after, rho_sim=rho_sim, T1=T1, T2s=T2s, N=N )
    rho_ideal = time_evole_ideal(H, t_gate=t_after, rho_ideal=rho_ideal, T1=T1, T2s=T2s, N=N)
    return rho_sim, rho_ideal


def extract_angle(value):
    '''
    This function print out the angle information from an arbitrary rotation. It's important for Axis Rotation X, Y ,Z
    For example: value = 'Y90 qb1' or 'Y180 qb2' or 'mX90s qb4' or 'Z32 qb2' or 'mY172s qb7' or 'Y2s qb7'
    '''
    #We find out at which place the angle starts. Then we find out at which place the angle ends.
    start = 0 #We initial the start place and the end place.
    end = 0
    if 'm' in value:
        start = 2
    else:
        start = 1
    if 's' in value:
        end = value.index('s')
    else:
        end = value.index(' ')
    angle = int(value[start:end]) * np.pi / 180  #Interpret the angle in the unit of pi
    if 'm' in value:
        return -angle
    else:
        return angle


def zrot(value, rho_sim, rho_ideal, tensorsz):
    '''
    This function print out rho_sim and rho_ideal after zero-time gate manipulation (Z rotation) 
    For example: value = 'Z172 qb3' or 'mZ90 qb6'
    '''
    angle = extract_angle(value)
    U = (-1j * tensorsz[ int(value[-1]) -1 ] * angle).expm()
    rho_sim = U * rho_sim * U.dag()
    rho_ideal = U * rho_ideal * U.dag()
    return rho_sim, rho_ideal

def xrot_h(value, tensorsx, times):
    '''
    This function calculate the Hamiltonian for X rotation
    '''
    angle = extract_angle(value)
    t_xy = times['t_xy']
    amp = angle/(t_xy)
    add_H = amp * tensorsx[int(value[-1]) -1 ]
    return add_H

def yrot_h(value, tensorsy, times):
    '''
    This function calculate the Hamiltonian for X rotation
    '''
    angle = extract_angle(value)
    t_xy = times['t_xy']
    amp = angle/(t_xy)
    add_H = amp * tensorsy[int(value[-1]) -1 ]
    return add_H
    


#Def 3: We extract important information of the experiments, in order to prepare for the simulation
def manip_extract(path):
    """
    We get the sequence of manipulation "manip" from a path (from a file) 
    """
    #We open the 'PrepSeq.txt' from the path
    files = os.listdir(path)
    for f in files:
        if f.endswith('.txt'):
            txt = path + '\\' + f
    with open(txt, 'r') as f:
        data= f.readlines()[-1] #We extact the last line of the PrepSeq.txt which contain the manipulation information
                                #For example: data = "prep_sequence = ['Y90 qb1', 'CZ qb2 qb1', 'mY90s qb2', 'X180s qb1']"
    #To convert a string to list, we take the following steps
    data = data[data.find('[')+1 : -1] #We extract the string information between [ and ]
    manip = data.split(",") #We split the total string to each single manipulation
    for i in range(len(manip)): #Now we remove ' and slash to nake each single string Standard
        if i == 0:
            manip[i] = manip[i].replace("'","")
        else:
            manip[i] = manip[i].replace(" '","") 
            manip[i] = manip[i].replace("'","")
    return manip

def t1t2s_extract(path, qubits):
    """
    We get the time-parameters T1 & T2s for each qubits from a path (from a file) 
    """
    #We open the document '.hdf5' from the path
    files = os.listdir(path)
    for f in files:
        if f.endswith('.hdf5'):
            hdf = path + '\\' + f
    hdf = h5py.File(hdf,'r')
    T1 = []
    T2s = []
    for qb in qubits:
        key = 'Instrument settings/'+str(qb)
        T1.append((float(hdf[key].attrs['T1']))/10**(-6))
        T2s.append((float(hdf[key].attrs['T2_star']))/10**(-6))
    return T1, T2s
    

#Def 4: We define arbitrary manipulation for manip(input a list)
def manipulate(manip, rho_sim, rho_ideal, times, H0, N):
    '''
    This function calculates arbitrary manipulation and prints the final rho_sim & rho_ideal out.
    CZ between 2 qubits: rho = cz(...)
    Axis rotations for qb j: H = amp * tensorsm[j]     All Y and X gates: 40 ns ; All Z gates: zrot()
    For example: manip = ['Y90 qb1', 'Y180 qb2', 'mX90s qb4', 'Z32 qb2', 'mY172s qb7', 'Y2s qb7',  'CZ qb2 qb1',  'Z90 qb2', 'mZ90s qb2']
    '''
    #Matrixes and time-parameters are in the dictionaries gates{} and times{}
    tensorsx, tensorsy, tensorsz, identity = gates(N)
    T1 = times['T1']
    T2s = times['T2s']
    t_xy = times['t_xy']
    t_before = times['t_before']
    t_cz = times['t_cz']
    t_after = times['t_after']
 
    
    H = H0 #Initial the Hamiltonian
    
    manip.append(' ') #So that we don't need to discuss the end, and the 's' has nothing to do with the real manipualtion
    
    index = 0
    while index < len(manip)-1:
        value = manip[index]
        future = manip[index+1] #To check if the next manipulation is simultaneous or not

        #3 cases for X, Y, Z rotation for each qb, in this case, we should discuss about "future"
        if 'C' not in value:
            if 'X' in value:
                H += xrot_h(value=value, tensorsx=tensorsx, times=times)
            elif 'Y' in value:
                H += yrot_h(value=value, tensorsy=tensorsy, times=times)
            elif 'Z' in value and 'C' not in value:
                rho_sim, rho_ideal = zrot(value=value, rho_sim=rho_sim, rho_ideal=rho_ideal, tensorsz=tensorsz) 
            #Now check if the next manipulation is simultaneous or not
            #if s: add H together, if not: calculate rho now
            if 's' in future:
                index += 1
            elif 's' not in future:
                rho_sim = time_evole(H, t_gate=t_xy, rho_sim=rho_sim, T1=T1, T2s=T2s, N=N) #We recall "time_evole" functions to calculate rho and rho_ideal
                rho_ideal = time_evole_ideal(H, t_gate=t_xy, rho_ideal=rho_ideal, T1=T1, T2s=T2s, N=N)
                H = H0 #Every time after we have already calculated rho, we must initial H
                index += 1
        
        #CZ case: In this case, we shouldn't discuss about future 
        elif 'C' in value:          
            rho_sim, rho_ideal = cz(value=value, rho_sim=rho_sim, rho_ideal=rho_ideal, times=times, N=N)
            index += 1
        
    return rho_sim, rho_ideal
