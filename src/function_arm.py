#import numpy as np
import time
import urx
from math import *

rob = urx.Robot("192.168.1.101")
rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(2, (0, 0, 0.1))
time.sleep(0.2)

def arm(X,Y,Z,kind,pose0,acc,vel):
    global rob
    # print("X:",X)
    # print("Y:",Y)
    # print("Z:",Z)
    # print("kind:",kind)
    
    num=len(kind)
    # if num > 5:
    #     num=5
    for a in range(num):
        # if Z[a]>=-0.1 and Z[a]<=0.5 and X[a]>=-0.1 and X[a]<=0.1 and Y[a]>=-0.1 and Y[a]<=0.1:
        #     if Z[a] < 0.05:
        #         Z[a] = 0.05
        #     elif Z[a] > 0.2:
        #          Z[a] = 0.2
                 
        #     if X[a] < -0.1:
        #          X[a] = -0.1
        #     elif X[a] > 0.1:
        #          X[a] = 0.1
                
        #     if Y[a]<-0.1:
        #          Y[a] = -0.1
        #     elif Y[a] > 0.1:
        #          Y[a] = 0.1
                
            rob.movel_tool((0,0,0.05,0,0,0),acc,vel)
            time.sleep(0.1)
            rob.movel_tool((X[a],Y[a],0,0,0,0),acc,vel)
            time.sleep(0.1)
            rob.movel_tool((0,0,Z[a]-0.05,0,0,0),acc,vel)
            time.sleep(0.1)
            
            # if kind[a]==1 or kind[a]==0:
            #     car.watering(1)
            # if kind[a]==2:
            #     rob.movel_tool((0,0.045,0,0,0,0),acc,vel)
            #     car.watering(2)
            rob.movel_tool((0,0,-0.05,0,0,0),acc,vel)
            time.sleep(0.1)
            rob.movel(pose0,acc,vel)
            time.sleep(0.1)
    # return

def arm_movej(pose,acc,vel):
    global rob
    try:
        rob.movej(pose,acc,vel)
    except Exception as e:
        print(e)
        time.sleep(3)
        rob.movej(pose,acc,vel)
        
def arm_movel(pose,acc,vel):
    global rob
    try:
        rob.movel(pose,acc,vel)
    except Exception as e:
        print(e)
        time.sleep(3)
        rob.movel(pose,acc,vel)
        
def arm_movel_tool(pose,acc,vel):
    global rob
    try:
        rob.movel_tool(pose,acc,vel)
    except Exception as e:
        print(e)
        time.sleep(3)
        rob.movel_tool(pose,acc,vel)
    
def arm_getl():
    global rob    
    return [value for value in rob.getl()]

def arm_getpose():
    global rob    
    return rob.get_pos()

def arm_getpos():
    global rob    
    return rob.get_pose()

def rotvectorpy():
    global rob    
    return rob.rotvec2rpy()

def rv2rpy(rx,ry,rz):
  rpy = []
  theta = sqrt(rx*rx + ry*ry + rz*rz)
  kx = rx/theta
  ky = ry/theta
  kz = rz/theta
  cth =  cos(theta)
  sth =  sin(theta)
  vth = 1- cos(theta)
  
  r11 = kx*kx*vth + cth
  r12 = kx*ky*vth - (kz*sth)
  r13 = kx*kz*vth + (ky*sth)
  r21 = kx*ky*vth + (kz*sth)
  r22 = ky*ky*vth + cth
  r23 = ky*kz*vth - (kx*sth)
  r31 = kx*kz*vth - (ky*sth)
  r32 = ky*kz*vth + (kx*sth)
  r33 = kz*kz*vth + cth
  
  beta = atan2(-r31, sqrt(r11*r11+r21*r21))
  
  if beta > radians(89.99):
      beta = radians(89.99)
      alpha = 0
      gamma = atan2(r12,r22)
  elif beta < -radians(89.99):
      beta = -radians(89.99)
      alpha = 0
      gamma = -atan2(r12,r22)
  else:
      cb = cos(beta)
      alpha = atan2(r21/cb,r11/cb)
      gamma = atan2(r32/cb,r33/cb)
      
  gamma = degrees(gamma)   
  beta = degrees(beta) 
  alpha = degrees(alpha) 
      
  rpy.append(gamma) 
  rpy.append(beta)
  rpy.append(alpha)
  
  # rpy[0]= gamma
  # rpy[1]= beta
  # rpy[2]= alpha
  
  return rpy

def arm_close():
    global rob
    rob.close()
    
def restart():
    global rob
    rob = urx.Robot("192.168.1.101")
    rob.set_tcp((0, 0, 0.1, 0, 0, 0))
    rob.set_payload(2, (0, 0, 0.1))
    time.sleep(0.2)

