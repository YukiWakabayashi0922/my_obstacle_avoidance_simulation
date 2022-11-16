# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import cvxpy as cp
import math
import numpy as np
import sys
sys.path.append("../CubicSpline/")
import cubic_spline_planner
import random
from operator import add

# road = False
road = True
save_simulation = False
path_planning = True

Nx = 4 #number of states
Nu = 2 #number of control inputs
dt = 0.2
DT = 0.2
W_track = 1.5 #wheel track in metre
W_base = 3.5 #wheel base in metre

N_search = 10 # search for closest point in the next N_search points on the path
H = 5 # Horizon length
simulation_time_limit = 100 #seconds
accept_dist = 0.2 #acceptable destination distance
accept_stop_v = 0.05 #acceptable stopping velocity

desired_speed = 5           # m/s
max_speed = 15              # m/s
max_reverse_speed = 5       # m/s
max_steer_angle = np.pi / 4     #max steering angle
max_steer_rate = np.pi / 6      #max steering speed
max_acc = 2                     #maximum acceleration m/s^2

W1 = np.array([0.01, 0.01])  # input weightage
W2 = np.array([2.0, 2.0, 0.5, 0.5])  # state error weightage
W3 = np.array([0.01, 0.1])  # rate of input change weightage
W4 = W2  # state error weightage

#potential field
start_x1 = 0.0  # start x position [m]
start_y1 = -5.0  # start y positon [m]
goal_x1 = 50.0  # goal x position [m]
goal_y1 = 30.0  # goal y position [m]

start_x2 = 10.0  # start x position [m]
start_y2 = 5.0  # start y positon [m]
goal_x2 = 60.0  # goal x position [m]
goal_y2 = 40.0  # goal y position [m]

grid_size = 0.2  # potential grid size [m]
robot_radius = 5.0  # robot radius [m]
NUM_OF_OBSTACLES = 12

################################################################################################### Class state
class State:
    def __init__(self,x_state,y_state,yaw,velocity):
        self.x = x_state
        self.y = y_state
        self.yaw = yaw
        self.v = velocity

    #Update state variable for given acceleration and steering angle
    def update_state(self, acc, steer_angle):
        self.x = self.x + self.v * np.cos(self.yaw) * dt
        self.y = self.y + self.v * np.sin(self.yaw) * dt
        self.yaw = self.yaw + (self.v / W_base) * np.tan (steer_angle) * dt
        self.v = self.v + acc * dt

    def state_to_vec(self):
        state_vec = np.array([self.x, self.y, self.yaw, self.v])
        return state_vec
    #end

################################################################################################### ########################### plot_car
# Vehicle parameters
L = 3.0  # LENGTH[m]
W = 3.0  # WIDTH[m]
D = 1.0  # BACKTOWHEEL[m]
WD = 1.0  # WHEEL_DIA[m]
WW = 0.25  # WHEEL_WIDTH[m]

def plot_car(x, y, yaw, steer=0.0, cabcolor="-y", truckcolor="-k"):  # pragma: no cover

    rotate = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    rotate_steer = np.array([[np.cos(steer),-np.sin(steer)],[np.sin(steer),np.cos(steer)]])

    car = np.array([[-D,-D,(L-D),(L-D),-D],[W/2,-W/2,-W/2,W/2,W/2]])
    car = np.matmul(rotate,car)
    car[0,:] += x
    car[1,:] += y
    plt.plot(car[0,:],car[1,:],cabcolor)

    wheel1 = np.array([[-WD/2,-WD/2,WD/2,WD/2,-WD/2],[WW/2,-WW/2,-WW/2,WW/2,WW/2]])
    wheel2 = np.array([[-WD/2,-WD/2,WD/2,WD/2,-WD/2],[WW/2,-WW/2,-WW/2,WW/2,WW/2]])

    wheel1[0,:] += 0.5
    wheel1[1,:] += 1.75
    wheel1 = np.matmul(rotate,wheel1)
    wheel1[0,:] += x
    wheel1[1,:] += y
    plt.plot(wheel1[0,:],wheel1[1,:],truckcolor)

    wheel2[0,:] += 0.5
    wheel2[1,:] -= 1.75
    wheel2 = np.matmul(rotate,wheel2)
    wheel2[0,:] += x
    wheel2[1,:] += y
    plt.plot(wheel2[0,:],wheel2[1,:],truckcolor)

################################################################################################### dynamic_model
def dynamic_model(velocity, yaw, steer):
    A = np.array([[1.0 , 0.0 , - dt * velocity * math.sin(yaw), dt * math.cos(yaw) ],\
                  [0.0 , 1.0 , dt * velocity * math.cos(yaw),  dt * math.sin(yaw)],\
                  [0.0 , 0.0 , 1.0                , dt * math.tan(steer) / W_base],\
                  [0.0 , 0.0 , 0.0 , 1.0]])

    B = np.array([[0.0 , 0.0 ],\
                  [0.0 , 0.0 ],\
                  [0.0 , dt * velocity / (W_base * math.cos(steer) ** 2) ],\
                  [dt  , 0.0]])


    C = np.array([dt * velocity * math.sin(yaw) * yaw,\
                   - dt * velocity * math.cos(yaw) * yaw ,\
                  - dt * velocity * steer / (W_base * math.cos(steer) ** 2) ,\
                  0.0])
    return A, B, C

################################################################################################### calc_predicted_trajectory
def calc_predicted_trajectory(acc,steer,cur_state_vec):
    traj_pred = np.zeros((Nx,H+1))  #Nx = 4, H = 5
    traj_pred[:,0] = cur_state_vec.T
    pred_state = State(cur_state_vec[0], cur_state_vec[1], cur_state_vec[2], cur_state_vec[3])
    for i in range(H):
        pred_state.update_state(acc[i], steer[i])
        temp_state = pred_state.state_to_vec()
        traj_pred[:,i+1] = temp_state.T
    return traj_pred

################################################################################################### run_MPC_solo
def run_MPC_solo(traj_des, cur_state_vec, mpc_acc, mpc_steer, steer_des, goal):

    for iter in range(3):
        traj_pred = calc_predicted_trajectory(mpc_acc,mpc_steer,cur_state_vec)
        x = cp.Variable([Nx, H+1])
        u = cp.Variable([Nu, H])

        cost = 0.0
        constraints = []
        for i in range(H):
            cost += cp.sum(W1 * cp.square(u[:, i]))                                   # input weightage
            cost += cp.sum(W2 * cp.square(traj_des[:, i] - x[:, i]))                  # state error weightage
            #cost += cp.sum(W2 * cp.square([goal[0],goal[1],0,0] - x[:, i]))                  # terminal cost
            if i < (H - 1):
                cost += cp.sum(W3 * cp.square(u[:, i+1] - u[:, i]))                    # rate of input change weightage
                constraints += [cp.abs(u[1, i+1] - u[1, i]) <= max_steer_rate * dt]

            A,B,C = dynamic_model(traj_pred[3,i], traj_pred[2,i], mpc_steer[i])
            constraints += [x[:, i+1] == A * x[:, i] + B * u[:, i] + C]


        cost += cp.sum(W4 * cp.square(traj_des[:, H] - x[:, H]))                      # final state error weightage
        #cost += cp.sum(10 * cp.square([goal[0],goal[1]] - x[:2, H]))                  # terminal cost

        constraints += [x[:, 0] == cur_state_vec]
        constraints += [x[3, :] <= max_speed]
        constraints += [x[3, :] >= -max_reverse_speed]
        constraints += [u[1, :] <= max_steer_angle]
        constraints += [u[1, :] >= -max_steer_angle]
        constraints += [u[0, :] <= max_acc]
        constraints += [u[0, :] >= -3*max_acc]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        mpc_x = x.value[0, :]
        mpc_y = x.value[1, :]
        mpc_yaw = x.value[2, :]
        mpc_v = x.value[3, :]
        mpc_acc = u.value[0, :]
        mpc_steer = u.value[1, :]

    return mpc_x, mpc_y, mpc_yaw, mpc_v, mpc_acc, mpc_steer

################################################################################################### run_MPC_duo
def run_MPC_duo(traj_des, cur_state_vec, mpc_acc, mpc_steer, steer_des, goal):

    for iter in range(3):
        traj_pred = calc_predicted_trajectory(mpc_acc,mpc_steer,cur_state_vec)
        x = cp.Variable([Nx, H+1])
        u = cp.Variable([Nu, H])

        cost = 0.0
        constraints = []
        for i in range(H):
            cost += cp.sum(W1 * cp.square(u[:, i]))                                   # input weightage
            cost += cp.sum(W2 * cp.square(traj_des[:, i] - x[:, i]))                  # state error weightage
            #cost += cp.sum(W2 * cp.square([goal[0],goal[1],0,0] - x[:, i]))                  # terminal cost
            if i < (H - 1):
                cost += cp.sum(W3 * cp.square(u[:, i+1] - u[:, i]))                    # rate of input change weightage
                constraints += [cp.abs(u[1, i+1] - u[1, i]) <= max_steer_rate * dt]

            A,B,C = dynamic_model(traj_pred[3,i], traj_pred[2,i], mpc_steer[i])
            constraints += [x[:, i+1] == A * x[:, i] + B * u[:, i] + C]


        cost += cp.sum(W4 * cp.square(traj_des[:, H] - x[:, H]))                      # final state error weightage
        #cost += cp.sum(10 * cp.square([goal[0],goal[1]] - x[:2, H]))                  # terminal cost

        constraints += [x[:, 0] == cur_state_vec]
        constraints += [x[3, :] <= max_speed]
        constraints += [x[3, :] >= -max_reverse_speed]
        constraints += [u[1, :] <= max_steer_angle]
        constraints += [u[1, :] >= -max_steer_angle]
        constraints += [u[0, :] <= max_acc]
        constraints += [u[0, :] >= -3*max_acc]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        mpc_x = x.value[0, :]
        mpc_y = x.value[1, :]
        mpc_yaw = x.value[2, :]
        mpc_v = x.value[3, :]
        mpc_acc = u.value[0, :]
        mpc_steer = u.value[1, :]

    return mpc_x, mpc_y, mpc_yaw, mpc_v, mpc_acc, mpc_steer
################################################################################################### cal_desired_trajectory
def cal_desired_trajectory(cur_state_vec, path_x, path_y, path_yaw, target_pt):
    traj_des = np.zeros((Nx,H+1))   #[4, 6]
    steer_des = np.zeros((1,H+1))   #[1, 6]
    distance = 0
    total_pts = len(path_x)

    target_pt = get_closest_point_on_path(path_x, path_y, cur_state_vec, target_pt)

    traj_des[0,0] = path_x[target_pt]
    traj_des[1,0] = path_y[target_pt]
    traj_des[2,0] = path_yaw[target_pt]
    traj_des[3,0] = desired_speed         #5 m/s

    for i in range(H):
        distance += abs(cur_state_vec[3]) * dt
        pts_travelled = int(round(distance))

        if (target_pt+pts_travelled)<total_pts:
            traj_des[0,i+1] = path_x[target_pt + pts_travelled]
            traj_des[1,i+1] = path_y[target_pt + pts_travelled]
            traj_des[2,i+1] = path_yaw[target_pt + pts_travelled]
            if (target_pt+pts_travelled) == total_pts - 1:
                traj_des[3,i+1] = 0.0
            else:
                traj_des[3,i+1] = desired_speed
        else:
            traj_des[0,i+1] = path_x[-1]
            traj_des[1,i+1] = path_y[-1]
            traj_des[2,i+1] = path_yaw[-1]
            traj_des[3,i+1] = 0.0
    if traj_des[3,1] == 0.0:
        traj_des[3,0] = 0.0
    return traj_des, target_pt, steer_des

################################################################################################### get_closest_point_on_path
def get_closest_point_on_path(path_x, path_y, cur_state_vec, point):
    diff_x = []
    diff_y = []
    dist_sq = []

    for i in range(len(path_x)):
        diff_x = np.append(diff_x, path_x[i] - cur_state_vec[0])
        diff_y = np.append(diff_y, path_y[i] - cur_state_vec[1])
        dist_sq = np.append(dist_sq, (diff_x)**2 + (diff_y)**2)

    min_d = min(dist_sq)
    temp = np.argwhere(dist_sq == min_d)
    target_pt = int(temp[0,0]) + point
    return target_pt

################################################################################################### destination_check
def destination_check(state, goal, target_pt, length_path):
    a = 0
    dist_to_dest = (state.x-goal[0])**2+(state.y-goal[1])**2
    if dist_to_dest < accept_dist:
        a += 1
    if state.v < abs(accept_stop_v):
        a += 1
    if abs(target_pt - length_path) < 5:
        a += 1
    if a == 3:
        return True
    return False

################################################################################################### run_controller
def run_controller(path_x1, path_x2, path_y1, path_y2, path_yaw1, path_yaw2, initial_state1, initial_state2, goal1, goal2, ox, oy, velX, velY, path_planning):
    current_state1 = initial_state1
    current_state2 = initial_state2
    imgct = 0

    #Initialize variables to store actual state values of car
    t = [0]

    x1 = [current_state1.x]
    y1 = [current_state1.y]
    yaw1 = [current_state1.yaw]
    vel1 = [current_state1.v]
    steer1 = [0]
    acc1 = [0]

    x2 = [current_state2.x]
    y2 = [current_state2.y]
    yaw2 = [current_state2.yaw]
    vel2 = [current_state2.v]
    steer2 = [0]
    acc2 = [0]

    mpc_acc1 = np.zeros(H)
    mpc_steer1 = np.zeros(H)
    mpc_acc2 = np.zeros(H)
    mpc_steer2 = np.zeros(H)

    cur_state_vec1 = current_state1.state_to_vec()
    target_pt1 = get_closest_point_on_path(path_x1, path_y1, cur_state_vec1, 0)
    cur_state_vec2 = current_state2.state_to_vec()
    target_pt2 = get_closest_point_on_path(path_x2, path_y2, cur_state_vec2, 0)

    while t[-1] <= simulation_time_limit:
        imgct += 1
        cur_state_vec1 = current_state1.state_to_vec()
        cur_state_vec2 = current_state2.state_to_vec()

        traj_des1, target_pt1, steer_des1 = cal_desired_trajectory(cur_state_vec1, path_x1, path_y1, path_yaw1, target_pt1)
        traj_des2, target_pt2, steer_des2 = cal_desired_trajectory(cur_state_vec2, path_x2, path_y2, path_yaw2, target_pt2)

        # mpc_x1, mpc_x2, mpc_y1, mpc_y2, mpc_yaw1, mpc_yaw2, mpc_v1, mpc_v2, mpc_acc1, mpc_acc2, mpc_steer1, mpc_steer2 \
                # = run_MPC_duo(traj_des1, traj_des2, cur_state_vec1, cur_state_vec2, mpc_acc1, mpc_acc2, mpc_steer1, mpc_steer2, steer_des1, steer_des2, goal1, goal2)

        mpc_x1, mpc_y1, mpc_yaw1, mpc_v1, mpc_acc1, mpc_steer1 = run_MPC_solo(traj_des1, cur_state_vec1, mpc_acc1, mpc_steer1, steer_des1, goal1)
        mpc_x2, mpc_y2, mpc_yaw2, mpc_v2, mpc_acc2, mpc_steer2 = run_MPC_solo(traj_des2, cur_state_vec2, mpc_acc2, mpc_steer2, steer_des2, goal2)

        current_state1.update_state(mpc_acc1[0], mpc_steer1[0])
        current_state2.update_state(mpc_acc2[0], mpc_steer2[0])

        time = t[-1] + dt
        t.append(time)

        x1.append(current_state1.x)
        y1.append(current_state1.y)
        yaw1.append(current_state1.yaw)
        vel1.append(current_state1.v)
        steer1.append(mpc_steer1[0])
        acc1.append(mpc_acc1[0])

        x2.append(current_state2.x)
        y2.append(current_state2.y)
        yaw2.append(current_state2.yaw)
        vel2.append(current_state2.v)
        steer2.append(mpc_steer2[0])
        acc2.append(mpc_acc2[0])

        if destination_check(current_state1, goal1, target_pt1, len(path_x1)) and destination_check(current_state2, goal2, target_pt2, len(path_x2)):
            print("Reached destination")
            break

        ox = np.add(ox,velX)
        oy = np.add(oy,velY)

        plt.cla()

        # plt.plot(mpc_x1, mpc_y1, "xr", label="MPC1")                             #赤色のバツ印
        plt.plot(path_x1, path_y1, "-r", label="course1")                        #赤色の直線
        plt.plot(ox, oy, "ok")                                                  #黒色の丸印
        # plt.plot(x1, y1, "ob", label="trajectory1")                              #青色の丸印
        plt.plot(goal1[0], goal1[1], "om")                                      #紫色の丸印
        # plt.plot(traj_des1[0, :], traj_des1[1, :], "xk", label="xref1")          #黒色のバツ印
        # plt.plot(path_x1[target_pt1], path_y1[target_pt1], "xg", label="target1")  #緑色のバツ印
        plot_car(current_state1.x, current_state1.y, current_state1.yaw, mpc_steer1[0])

        # plt.plot(mpc_x2, mpc_y2, "xr", label="MPC2")                             #赤色のバツ印
        plt.plot(path_x2, path_y2, "-r", label="course2")                        #赤色の直線
        # plt.plot(x2, y2, "ob", label="trajectory2")                              #青色の丸印
        plt.plot(goal2[0], goal2[1], "om")                                      #紫色の丸印
        # plt.plot(traj_des2[0, :], traj_des2[1, :], "xk", label="xref2")          #黒色のバツ印
        # plt.plot(path_x2[target_pt2], path_y2[target_pt2], "xg", label="target2")  #緑色のバツ印
        plot_car(current_state2.x, current_state2.y, current_state2.yaw, mpc_steer2[0])

        plt.axis("equal")
        plt.grid(True)
        plt.title("Time[s]:" + str(round(time, 2)))

        if save_simulation:
            plt.savefig('Q_'+str(imgct))
        plt.pause(0.0001)

        if path_planning:
            path_x1, path_x2, path_y1, path_y2, path_yaw1, path_yaw2, ox, oy = pot_field_path(current_state1.x, current_state2.x, current_state1.y, current_state2.y, ox, oy, velX, velY)
            if stop_planning(path_x1, path_y1, goal_x1, goal_y1) and stop_planning(path_x2, path_y2, goal_x2, goal_y2):
                path_planning = False

    return t, x1, x2, y1, y2, yaw1, yaw2, vel1, vel2, steer1, steer2, acc1, acc2

def stop_planning(path_x, path_y, goal_x, goal_y):
    dist_to_dest = (path_x[-1] - goal_x)**2 + (path_y[-1] - goal_y)**2
    if dist_to_dest < accept_dist:
        return True
    else:
        return False

###########################################################################################################################################
#                                                                                                                                         #
#                                                                                                                                         #
#                       PATH PLANNING                                                                                                     #
#                                                                                                                                         #
#                                                                                                                                         #
###########################################################################################################################################
# Parameters
KP = 15.0  # attractive potential gain
ETA = 500.0  # repulsive potential gain

show_animation=True

def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy) #+ 0.5*10*np.hypot(y-0,0)

def calc_repulsive_potential(x, y, ox, oy, obs):
    obs = len(ox)
    pot = 0
    for i in range(obs):
        pot += 0.5*ETA*np.exp(-np.hypot(x-ox[i],y-oy[i]))
    return  pot

def get_motion_model():
    motion = []
    num = 50
    for i in range(num*2):
        deg = 2*i*np.pi/num
        motion.append([np.cos(deg),np.sin(deg)])

    return motion

def potential_field_planning(sx, sy, gx, gy, ox, oy):
     mot = get_motion_model()
     predictX = []
     predictY = []
     for i in range(len(mot)):
         predictX.append(sx + mot[i][0] * grid_size)
         predictY.append(sy + mot[i][1] * grid_size)

     gnet = []
     min_gnet = 0
     min_gnet_pos = 0
     for i in range(len(mot)):
         ga = calc_attractive_potential(predictX[i], predictY[i], gx, gy)
         gr = calc_repulsive_potential(predictX[i], predictY[i], ox, oy, NUM_OF_OBSTACLES)
         gnet.append(ga + gr)

         if(i==0):
             min_gnet = gnet[i]
             min_gnet_pos = i
         else:
             if(min_gnet > gnet[i]):
                 min_gnet = gnet[i]
                 min_gnet_pos = i

     step_x = mot[min_gnet_pos][0]
     step_y = mot[min_gnet_pos][1]

     return [step_x,step_y]

def initialize_obstacles(NUM_OF_OBSTACLES):

    if road:
        ox = [10, 12, 15, 17, 20, 25, 27]
        oy = [30, 20, 10, 15, 25, 30, 17]
        velX = [0.05, 0.05, 0, 0, -0.05, -0.05, 0]
        velY = [0, 0, -0.05, 0.05, 0, 0.05, -0.05]
    else:
        coordinateX = []
        coordinateY = []
        velX = []
        velY = []

        for i in range(1,NUM_OF_OBSTACLES):
             coordinateX.append(random.randrange(10, gx, 1))
             coordinateY.append(random.randrange(10, gy, 1))
             velX.append((np.random.random()/40)*(-1)**i)
             velY.append((np.random.random()/40)*(-1)**i)

        ox = coordinateX
        oy = coordinateY
    return ox,oy,velX,velY

def get_spline_path(ax,ay,dl):
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, dl)
    return cx, cy, cyaw

def pot_field_path(sx1, sx2, sy1, sy2, ox, oy, velX, velY):
    iter = 0
    path_x1 = np.array([])
    path_y1 = np.array([])
    path_x2 = np.array([])
    path_y2 = np.array([])
    # pathYaw=np.array([])

    while iter <= H+5:
         [step_x1, step_y1] = potential_field_planning(sx1, sy1, goal_x1, goal_y1, ox, oy)
         [step_x2, step_y2] = potential_field_planning(sx2, sy2, goal_x2, goal_y2, ox, oy)

         sx1 = sx1 + step_x1
         sy1 = sy1 + step_y1

         sx2 = sx2 + step_x2
         sy2 = sy2 + step_y2

         ox = np.add(ox, velX)
         oy = np.add(oy, velY)

         path_x1 = np.append(path_x1, sx1)
         path_y1 = np.append(path_y1, sy1)

         path_x2 = np.append(path_x2, sx2)
         path_y2 = np.append(path_y2, sy2)
         # pathYaw=np.append(pathYaw, math.atan2(cy, -cx))
         iter += 1

    path_x1, path_y1, path_yaw1 = get_spline_path(path_x1, path_y1, 1)
    path_x2, path_y2, path_yaw2 = get_spline_path(path_x2, path_y2, 1)

    return path_x1, path_x2, path_y1, path_y2, path_yaw1, path_yaw2, ox, oy

################################################################################################### main
def main():

    ox,oy,velX,velY = initialize_obstacles(NUM_OF_OBSTACLES)

    path_x1, path_x2, path_y1, path_y2, path_yaw1, path_yaw2, ox, oy = pot_field_path(start_x1, start_x2, start_y1, start_y2, ox, oy, velX, velY)

    initial_state1 = State(path_x1[0], path_y1[0], path_yaw1[0], 0.0)
    initial_state2 = State(path_x2[0], path_y2[0], path_yaw2[0], 0.0)
    goal1 = np.array([goal_x1, goal_y1])
    # goal2 = np.array([goal_x2, goal_y2])
    goal2 = np.array([path_x1[0] - 10, path_y1[0] - 10])

    t, x1, x2, y1, y2, yaw1, yaw2, vel1, vel2, steer1, steer2, acc1, acc2 \
            = run_controller(path_x1, path_x2, path_y1, path_y2, path_yaw1, path_yaw2, initial_state1, initial_state2, goal1, goal2, ox, oy, velX, velY, path_planning)

    # path_x1, path_y1, path_yaw1, ox, oy = pot_field_path(start_x1, start_y1, ox, oy, velX, velY)
    # initial_state1 = State(path_x1[0], path_y1[0], path_yaw1[0], 0.0)
    # goal1 = np.array([goal_x1, goal_y1])
    # t1, x1, y1, yaw1, vel1, steer1, acc1 = run_controller(path_x1, path_y1, path_yaw1, initial_state1, goal1, ox, oy, velX, velY, path_planning)

    # path_x2, path_y2, path_yaw2, ox, oy = pot_field_path(start_x2, start_y2, ox, oy, velX, velY, dist_step)
    # initial_state2 = State(path_x2[0], path_y2[0], path_yaw2[0], 0.0)
    # goal2 = np.array([goal_x2, goal_y2])
    # t2, x2, y2, yaw2, vel2, steer2, acc2, lyap2, lu2, lx2, ldu2 = run_controller(path_x2, path_y2, path_yaw2, dist_step, initial_state2, goal2, ox, oy, velX, velY, path_planning)

if __name__ == '__main__':
    main()
