import pyads
import time
import datetime
import csv
import socket, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pi = np.pi

# Khau khuech dai
Kpx = 0.9
Kpy = 0.9
Kpz = 0.9

Kdx = 0 #0.001
Kdy = 0 #0.001
Kdz = 0 #0.001

preErr_x = 0
preErr_y = 0
preErr_z = 0

Sampling_time = 0.7    #sec
inv_Sampling_time = 1/Sampling_time

# input the desired trajectory
def trajectory():
    data = pd.read_csv("trajectory.csv", delimiter = "	")
    #xy.info()
    #xy.head(5)
    #xy.describe()
    data = data.values
    return data
    
def connect_to_plc():
    #import pyads
    # connect to plc and open connection
    #plc = pyads.Connection("192.9.66.203.1.1", 851)     # my twincat 
    plc = pyads.Connection("192.9.45.134.1.1", 851)      # lab twincat
    
    print("Connecting..")
    plc.open()
    print("Connected!")
    plc.read_device_info()
    return plc
#==== connect to laptop and camera
def connect_to_camera():
    #import socket
    #def Client():
    Adress=('192.9.44.243',5000)
    s = socket.socket()
    s.connect(Adress)
    return s

def get_tension(plc):
    T0, T1, T2, T3, T4, T5, T6, T7 = plc.read_by_name('MAIN.newton2', pyads.PLCTYPE_LREAL*8)
    return T0, T1, T2, T3, T4, T5, T6, T7

def angle_to_PLC(plc, A1, A2, A3, A4, A5, A6, A7, A8):          # send motor angle to twincat 
    plc.write_by_name('MAIN.Des',[A1, A2, A3, A4, A5, A6, A7, A8] , pyads.PLCTYPE_LREAL*8)

def get_position(s):
    
    #s.sendall('0'.encode('utf-8'))
    
    position = s.recv(4096)
    position = position.decode('utf-8')
    position_arr = position.split(":")
    
    X = float(position_arr[-3])
    Y = float(position_arr[-2])
    Z = float(position_arr[-1])
    
    # position = s.recv(2048)
    # position_arr = pickle.loads(position)
    # X = position_arr[0]
    # Y = position_arr[1]
    # Z = position_arr[2]
    
    return X, Y, Z

def circle_trajectory(r,n):
    for i in range(0, n):
        x_desired = r * np.cos(2*i*pi/n)
        y_desired = 0 * i
        z_desired = 445 + r * np.sin(2*i*pi/n)
    return x_desired, y_desired, z_desired

#circle_trajectory(150,100)

# calculate inverse kinematic of CDPRs
def inverse_kinematic(x, y, z):
    
    # Pulley position
    # P0 = np.array([-455, 540, 903])  
    # P1 = np.array([455, 540, 902])              
    # P2 = np.array([-455, -540, 877])    
    # P3 = np.array([455, -540, 901])     
    # P4 = np.array([-540, 465, 73.5])   # 455 or 465
    # P5 = np.array([540, 455, 73])       
    # P6 = np.array([-540, -455, 73])     
    # P7 = np.array([540, -455, 72]) 
    
    P0 = np.array([-455, 535, 898])  
    P1 = np.array([450, 535, 897])              
    P2 = np.array([-460, -545, 882])    
    P3 = np.array([450, -545, 896])     
    P4 = np.array([-545, 460, 70])   # 455 or 465
    P5 = np.array([535, 450, 70])       
    P6 = np.array([-545, -460, 70])     
    P7 = np.array([535, -455, 72]) 
    
    # P0 = np.array([-437.082, 524.3283, 864.7443])  
    # P1 = np.array([439.7277, 525.9461, 860.8701])              
    # P2 = np.array([-435.143, -521.906, 839.1691])    
    # P3 = np.array([437.8987, -523.83, 859.4645])     
    # P4 = np.array([-535.048, 466.8895, 79.72492])    
    # P5 = np.array([531.9962, 451.4305, 80.32775])       
    # P6 = np.array([-543.043, -463.811, 71.49693])     
    # P7 = np.array([529.0567, -448.63, 84.36402]) 
    
    # connection point between EEs and cable
    e0 = np.array([-130/2, 130/2, -130/2])
    e1 = np.array([130/2, 130/2, -130/2])
    e2 = np.array([-130/2, -130/2, -130/2])
    e3 = np.array([130/2, -130/2, -130/2])
    e4 = np.array([-130/2, 130/2, 130/2])
    e5 = np.array([130/2, 130/2, 130/2])
    e6 = np.array([-130/2, -130/2, 130/2])
    e7 = np.array([130/2, -130/2, 130/2])    
    
    # Center EEs position
    G = np.array([x, y, z])
     
    # center point of pulley
    L0v = P0 - e0 - G                      
    L1v = P1 - e1 - G
    L2v = P2 - e2 - G
    L3v = P3 - e3 - G 
    L4v = P4 - e4 - G
    L5v = P5 - e5 - G
    L6v = P6 - e6 - G
    L7v = P7 - e7 - G
    
    bxy0p = np.sqrt((L0v[0]**2) + (L0v[1]**2))
    bxy1p = np.sqrt((L1v[0]**2) + (L1v[1]**2))
    bxy2p = np.sqrt((L2v[0]**2) + (L2v[1]**2))
    bxy3p = np.sqrt((L3v[0]**2) + (L3v[1]**2))
    bxy4p = np.sqrt((L4v[0]**2) + (L4v[1]**2))
    bxy5p = np.sqrt((L5v[0]**2) + (L5v[1]**2))
    bxy6p = np.sqrt((L6v[0]**2) + (L6v[1]**2))
    bxy7p = np.sqrt((L7v[0]**2) + (L7v[1]**2))

    # pulley radius
    VL = np.array([13, 13, 13, 13, 13, 13, 13, 13])    
    
    bxy0 = bxy0p - VL[0]
    bxy1 = bxy1p - VL[1]
    bxy2 = bxy2p - VL[2]
    bxy3 = bxy3p - VL[3]
    bxy4 = bxy4p - VL[4]
    bxy5 = bxy5p - VL[5]
    bxy6 = bxy6p - VL[6]
    bxy7 = bxy7p - VL[7]
     
    bz0 = L0v[2]
    bz1 = L1v[2]
    bz2 = L2v[2]
    bz3 = L3v[2]
    bz4 = L4v[2]
    bz5 = L5v[2]
    bz6 = L6v[2]
    bz7 = L7v[2]
    
    MB0 = np.sqrt((bxy0**2) + (bz0**2))
    MB1 = np.sqrt((bxy1**2) + (bz1**2))
    MB2 = np.sqrt((bxy2**2) + (bz2**2))
    MB3 = np.sqrt((bxy3**2) + (bz3**2))
    MB4 = np.sqrt((bxy4**2) + (bz4**2))
    MB5 = np.sqrt((bxy5**2) + (bz5**2))
    MB6 = np.sqrt((bxy6**2) + (bz6**2))
    MB7 = np.sqrt((bxy7**2) + (bz7**2))
    
    # pulley radius
    rp = np.array([13, 13, 13, 13, 13, 13, 13, 13])    
    
    lf0 = np.sqrt((MB0**2) - (rp[0]**2))
    lf1 = np.sqrt((MB1**2) - (rp[1]**2))
    lf2 = np.sqrt((MB2**2) - (rp[2]**2))
    lf3 = np.sqrt((MB3**2) - (rp[3]**2))
    lf4 = np.sqrt((MB4**2) - (rp[4]**2))
    lf5 = np.sqrt((MB5**2) - (rp[5]**2))
    lf6 = np.sqrt((MB6**2) - (rp[6]**2))
    lf7 = np.sqrt((MB7**2) - (rp[7]**2))
    
    beta0 = np.arccos(lf0/np.sqrt((bxy0**2)+(bz0**2))) + np.arccos(bz0/(np.sqrt((bxy0**2)+bz0**2)))
    beta1 = np.arccos(lf1/np.sqrt((bxy1**2)+(bz1**2))) + np.arccos(bz1/(np.sqrt((bxy1**2)+bz1**2)))
    beta2 = np.arccos(lf2/np.sqrt((bxy2**2)+(bz2**2))) + np.arccos(bz2/(np.sqrt((bxy2**2)+bz2**2)))
    beta3 = np.arccos(lf3/np.sqrt((bxy3**2)+(bz3**2))) + np.arccos(bz3/(np.sqrt((bxy3**2)+bz3**2)))
    beta4 = np.arccos(lf4/np.sqrt((bxy4**2)+(bz4**2))) + np.arccos(bz4/(np.sqrt((bxy4**2)+bz4**2)))
    beta5 = np.arccos(lf5/np.sqrt((bxy5**2)+(bz5**2))) + np.arccos(bz5/(np.sqrt((bxy5**2)+bz5**2)))
    beta6 = np.arccos(lf6/np.sqrt((bxy6**2)+(bz6**2))) + np.arccos(bz6/(np.sqrt((bxy6**2)+bz6**2)))
    beta7 = np.arccos(lf7/np.sqrt((bxy7**2)+(bz7**2))) + np.arccos(bz7/(np.sqrt((bxy7**2)+bz7**2)))
    
    L0 = lf0 + beta0 * rp[0]
    L1 = lf1 + beta1 * rp[1]
    L2 = lf2 + beta2 * rp[2]
    L3 = lf3 + beta3 * rp[3]
    L4 = lf4 + beta4 * rp[4]
    L5 = lf5 + beta5 * rp[5]
    L6 = lf6 + beta6 * rp[6]
    L7 = lf7 + beta7 * rp[7]
    
    # At initial point (0, 0, 445)
    L_ini = np.array([808.3703712079258, 807.7270439797596, 791.8879833545069, 807.0844533343285, 777.0800506441725, 772.2623734847709, 772.2623734847709, 772.8645696745367])
    
    a0 = round(-(((L0 - L_ini[0])*360)/((pi*64)+5)),3)
    a1 = round(-(((L1 - L_ini[1])*360)/((pi*64)+5)),3)
    a2 = round(-(((L2 - L_ini[2])*360)/((pi*64)+5)),3)
    a3 = round(-(((L3 - L_ini[3])*360)/((pi*64)+5)),3)
    a4 = round(-(((L4 - L_ini[4])*360)/((pi*64)+5)),3)
    a5 = round(-(((L5 - L_ini[5])*360)/((pi*64)+5)),3)
    a6 = round(-(((L6 - L_ini[6])*360)/((pi*64)+5)),3)
    a7 = round(-(((L7 - L_ini[7])*360)/((pi*64)+5)),3)
    
    return a0, a1, a2, a3, a4, a5, a6, a7
    #return L0, L1, L2, L3, L4, L5, L6, L7
    
#A1, A2, A3, A4, A5, A6, A7, A8 = inverse_kinematic(0, 0, 445)

plc = connect_to_plc()
s = connect_to_camera()

data = trajectory()
#fig = plt.figure()
#ax = plt.axes(projection = '3d')
#ax.plot3D(data[:,0], data[:,1], data[:,2])
#plt.plot(xy)

pos_list = [[0,0,0]]            #position
compute_time = []               #calculation time
err_list = [[0,0,0]]            #error
command_list = [[0,0,0]]        #control signal
result = [[0,0,0,0,0,0]]

for i in range(1, len(data)+1):
    start =  time.time()
    #update tension
    T1, T2, T3, T4, T5, T6, T7, T8 = get_tension(plc)
    
    #update position
    X, Y, Z = get_position(s)
    pos_list = np.append(pos_list, ([[X, Y, Z]]), axis = 0)
    
    X_Err = data[i-1,0] - X         #real error
    Y_Err = data[i-1,1] - Y
    Z_Err = data[i-1,2] - Z
    
    err_list = np.append(err_list,([[X_Err, Y_Err, Z_Err]]), axis = 0)
    
    result = np.append(result,([[X, Y, Z, X_Err, Y_Err, Z_Err]]), axis = 0)
        
    if i < len(data):
        
        Err_x =  X_Err
        Err_y =  Y_Err
        Err_z =  Z_Err
        
        pPart_x = Kpx * Err_x
        pPart_y = Kpy * Err_y
        pPart_z = Kpz * Err_z
        
        dPart_x = Kdx*(Err_x - preErr_x)*inv_Sampling_time
        dPart_y = Kdy*(Err_y - preErr_y)*inv_Sampling_time
        dPart_z = Kdz*(Err_z - preErr_z)*inv_Sampling_time
        
        Cmd_x = pPart_x + dPart_x
        Cmd_y = pPart_y + dPart_y
        Cmd_z = pPart_z + dPart_z
        
        preErr_x = Err_x
        preErr_y = Err_y
        preErr_z = Err_z
            
        #saturation
        if Cmd_x > 10*2.5 : Cmd_x = 10
        if Cmd_x < -10*2.5 : Cmd_x = -10
            
        if Cmd_y > 10*2.5 : Cmd_y = 10
        if Cmd_y < -10*2.5 : Cmd_y = -10
            
        if Cmd_z > 10*2.5 : Cmd_z = 10
        if Cmd_z < -10*2.5 : Cmd_z = -10    
    
        command_list = np.append(command_list, ([[Cmd_x, Cmd_y, Cmd_z]]), axis = 0)

        #A1, A2, A3, A4, A5, A6, A7, A8 = inverse_kinematic(data[i,0], data[i,1], data[i,2])
        A1, A2, A3, A4, A5, A6, A7, A8 = inverse_kinematic(data[i,0]+Cmd_x, data[i,1]+Cmd_y, data[i,2]+Cmd_z)
        
        
        angle_to_PLC(plc, A1, A2, A3, A4, A5, A6, A7, A8)
        
        if i == 35 or i == 155:
            time.sleep(1)
        
        t = time.time()-start
        compute_time = np.append(compute_time, [t], axis = 0)
    else:
        break
        
    #print(round(time.time()-start,3))
    time.sleep(0.5)

#s.close()

# data = trajectory()
# for i in range(1, len(data)+1):   
#     if i != len(data):
#         A1, A2, A3, A4, A5, A6, A7, A8 = inverse_kinematic(data[i,0], data[i,1], data[i,2])
        
#         angle_to_PLC(plc, A1, A2, A3, A4, A5, A6, A7, A8)
#     else:
#         break
#     time.sleep(0.02)

# # to 0
# A1, A2, A3, A4, A5, A6, A7, A8 = inverse_kinematic(-10, -10, 445)
# angle_to_PLC(plc, A1, A2, A3, A4, A5, A6, A7, A8)