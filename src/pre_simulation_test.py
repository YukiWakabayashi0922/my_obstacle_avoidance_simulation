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

# get_obstacle = True
get_obstacle = False
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
accept_stop_v = 0.08 #acceptable stopping velocity

desired_speed = 3           # m/s
max_speed = 10              # m/s
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
start_y1 = 0.0  # start y position [m]
goal_x1 = 30.0  # goal x position [m]
goal_y1 = 30.0  # goal y position [m]

start_x2 = 5.0  # start x position [m]
start_y2 = 5.0  # start y position [m]
goal_x2 = 40.0  # goal x position [m]
goal_y2 = 40.0  # goal y position [m]

grid_size = 0.2  # potential grid size [m]
robot_radius = 5.0  # robot radius [m]
NUM_OF_OBSTACLES = 12

#initial_param : Set start position and goal position of robots
class initial_param:
    def __init__(self, start_x, start_y, goal_x, goal_y):
        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y

    def start(self):
        start_robot = np.array([self.start_x, self.start_y])
        return start_robot

    def goal(self):
        goal_robot = np.array([self.goal_x, self.goal_y])
        return goal_robot
#end

#State : Set x, y, yaw, velocity of robots
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

#Path : robot path
class Path:
    def __init__(self, current, goal, obstacle_state):
        self.current_x = current[0]
        self.current_y = current[1]
        self.goal_x = goal[0]
        self.goal_y = goal[1]
        self.obstacle_x = obstacle_state[0]
        self.obstacle_y = obstacle_state[1]
        self.obstacle_velocity_x = obstacle_state[2]
        self.obstacle_velocity_y = obstacle_state[3]

        self.KP = 15
        self.ETA = 500

    def calc_attractive_potential(self, predictX, predictY):
        return 0.5 * self.KP * np.hypot(predictX - self.goal_x, predictY - self.goal_y)

    def calc_repulsive_potential(self, predictX, predictY):
        obs = len(self.obstacle_x)
        pot = 0

        for i in range(obs):
            pot += 0.5 * self.ETA * np.exp(-np.hypot(predictX - self.obstacle_x[i], predictY - self.obstacle_y[i]))

        return pot

    def get_motion_model(self):
        motion = []
        num = 50
        for i in range(num*2):
            deg = 2*i*np.pi/num
            motion.append([np.cos(deg),np.sin(deg)])

        return motion

    def potential_field_planning(self):
        mot = Path.get_motion_model(self)
        predictX = []
        predictY = []

        for i in range(len(mot)):
            predictX.append(self.current_x + mot[i][0] * grid_size)
            predictY.append(self.current_y + mot[i][1] * grid_size)

        gnet = []
        min_gnet = 0
        min_gnet_pos = 0

        for i in range(len(mot)):
            ga = Path.calc_attractive_potential(self, predictX[i], predictY[i])
            gr = Path.calc_repulsive_potential(self, predictX[i], predictY[i])
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

    def get_spline_path(self, path_x, path_y):
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(path_x, path_y, 1)
        return cx, cy, cyaw

    def potential_field_path(self):
        iter = 0
        path_x = np.array([])
        path_y = np.array([])

        while iter <= H + 5:
            [step_x, step_y] = Path.potential_field_planning(self)

            self.current_x = self.current_x + step_x
            self.current_y = self.current_y + step_y

            self.obstacle_x = np.add(self.obstacle_x, self.obstacle_velocity_x)
            self.obstacle_y = np.add(self.obstacle_y, self.obstacle_velocity_y)

            path_x = np.append(path_x, self.current_x)
            path_y = np.append(path_y, self.current_y)

            iter += 1

        path_x, path_y, path_yaw = Path.get_spline_path(self, path_x, path_y)

        path = np.array([path_x, path_y, path_yaw])
        obstacle_state = np.array([self.obstacle_x, self.obstacle_y, self.obstacle_velocity_x, self.obstacle_velocity_y])

        return path, obstacle_state
#end

class Controller:
    def __init__(self, path, goal, obstacle_state):
        self.path_x = path[0]
        self.path_y = path[1]
        self.path_yaw = path[2]
        self.goal_x = goal[0]
        self.goal_y = goal[1]
        self.obstacle_x = obstacle_state[0]
        self.obstacle_y = obstacle_state[1]
        self.obstacle_velocity_x = obstacle_state[2]
        self.obstacle_velocity_y = obstacle_state[3]

    def get_closest_point_on_path(self, cur_state_vec):
        diff_x = []
        diff_y = []
        dist_sq = []

        for i in range(len(self.path_x)):
            diff_x.append(self.path_x[i] - cur_state_vec[0])
            diff_y.append(self.path_y[i] - cur_state_vec[1])
            dist_sq.append((diff_x[i])**2+(diff_y[i])**2)

        min_d = min(dist_sq)
        temp = np.argwhere(dist_sq == min_d)
        target_pt = int(temp[0])
        return target_pt

    def cal_desired_trajectory(self, cur_state_vec):
        traj_des = np.zeros((Nx,H+1))   #[4, 6]
        distance = 0
        total_pts = len(self.path_x)

        target_pt = get_closest_point_on_path(self, cur_state_vec)

        traj_des[0,0] = self.path_x[target_pt]
        traj_des[1,0] = self.path_y[target_pt]
        traj_des[2,0] = self.path_yaw[target_pt]
        traj_des[3,0] = desired_speed         #5 m/s

        for i in range(H):
            distance += abs(cur_state_vec[3]) * dt
            pts_travelled = int(round(distance))

            if (target_pt + pts_travelled) < total_pts:
                traj_des[0,i+1] = self.path_x[target_pt + pts_travelled]
                traj_des[1,i+1] = self.path_y[target_pt + pts_travelled]
                traj_des[2,i+1] = self.path_yaw[target_pt + pts_travelled]
                if (target_pt + pts_travelled) == total_pts - 1:
                    traj_des[3,i+1] = 0.0
                else:
                    traj_des[3,i+1] = desired_speed
            else:
                traj_des[0,i+1] = self.path_x[-1]
                traj_des[1,i+1] = self.path_y[-1]
                traj_des[2,i+1] = self.path_yaw[-1]
                traj_des[3,i+1] = 0.0

        if traj_des[3,1] == 0.0:
            traj_des[3,0] = 0.0

        return traj_des, target_pt

    def calc_predicted_trajectory(self, cur_state_vec, mpc_acc, mpc_steer):
        traj_pred = np.zeros((Nx,H+1))  #Nx = 4, H = 5
        traj_pred[:,0] = cur_state_vec.T
        pred_state = State(cur_state_vec[0], cur_state_vec[1], cur_state_vec[2], cur_state_vec[3])

        for i in range(H):
            pred_state.update_state(self.mpc_acc[i], self.mpc_steer[i])
            temp_state = pred_state.state_to_vec()
            traj_pred[:,i+1] = temp_state.T

        return traj_pred

    def dynamic_model(self, velocity, yaw, steer):
        A = np.array([[1.0 , 0.0 , - dt * velocity * math.sin(yaw), dt * math.cos(yaw)],\
                      [0.0 , 1.0 , dt * velocity * math.cos(yaw),  dt * math.sin(yaw)],\
                      [0.0 , 0.0 , 1.0                , dt * math.tan(steer) / W_base],\
                      [0.0 , 0.0 , 0.0 , 1.0]])

        B = np.array([[0.0 , 0.0],\
                      [0.0 , 0.0],\
                      [0.0 , dt * velocity / (W_base * math.cos(steer) ** 2)],\
                      [dt  , 0.0]])


        C = np.array([dt * velocity * math.sin(yaw) * yaw,\
                       - dt * velocity * math.cos(yaw) * yaw ,\
                      - dt * velocity * steer / (W_base * math.cos(steer) ** 2) ,\
                      0.0])
        return A, B, C

    def run_MPC(self, cur_state_vec, traj_des, mpc_acc, mpc_steer):
        for iter in range(3):
            traj_pred = calc_predicted_trajectory(self, cur_state_vec, mpc_acc, mpc_steer)
            x = cp.Variable([Nx, H+1]) #(4,6)
            u = cp.Variable([Nu, H])   #(2,5)

            cost = 0.0
            constraints = []
            for i in range(H):
                cost += cp.sum(W1 * cp.square(u[:, i]))                                   # input weightage
                cost += cp.sum(W2 * cp.square(traj_des[:, i] - x[:, i]))                  # state error weightage
                if i < (H - 1):
                    cost += cp.sum(W3 * cp.square(u[:, i+1] - u[:, i]))                    # rate of input change weightage
                    constraints += [cp.abs(u[1, i+1] - u[1, i]) <= max_steer_rate * dt]

                A,B,C = dynamic_model(self, traj_pred[3,i], traj_pred[2,i], mpc_steer[i])
                constraints += [x[:, i+1] == A * x[:, i] + B * u[:, i] + C]

            cost += cp.sum(W4 * cp.square(traj_des[:, H] - x[:, H]))                      # final state error weightage

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
            mpc_acc = u.value[0, :]
            mpc_steer = u.value[1, :]

        mpc = np.array([mpc_x, mpc_y, mpc_acc, mpc_steer])
        return mpc

    def run_controller(self):
        current_state = State(self.path_x, self.path_y, self.path_yaw, 0.0)
        cur_state_vec = current_state.state_to_vec()

        mpc_acc = np.zeros(H)
        mpc_steer = np.zeros(H)

        traj_des, target_pt = cal_desired_trajectory(self, cur_state_vec)
        mpc = run_MPC(self, cur_state_vec, traj_des, mpc_acc, mpc_steer)
        current_state.update_state(mpc_acc[0], mpc_steer[0])

        return target_pt, mpc, current_state

class Check:
    def __init__(self, current_state, goal, target_pt):
        self.path_x = current_state[0]
        self.path_y = current_state[1]
        self.velocity = current_state[3]
        self.goal_x = goal[0]
        self.goal_y = goal[1]
        self.target_pt = target_pt

    def destination_check(self):
        a = 0
        dist_to_dest = (self.path_x - self.goal_x)**2 + (self.path_y - self.goal_y)**2
        if dist_to_dest < accept_dist:
            a += 1
        if self.velocity < abs(accept_stop_v):
            a += 1
        if abs(self.target_pt - len(self.path_x)) < 5:
            a += 1
        if a == 3:
            return True

        return False

    def stop_planning(self):
        dist_to_dest = (self.path_x - self.goal_x)**2 + (self.path_y - self.goal_y)**2
        if dist_to_dest < accept_dist:
            return True
        else:
            return False

#################################################################################################### ########################### plot_car
# Vehicle parameters
L = 2.0  # LENGTH[m]
W = 2.0  # WIDTH[m]
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

    # wheel1[0,:] += 1.0
    wheel1[1,:] += 1.3
    wheel1 = np.matmul(rotate,wheel1)
    wheel1[0,:] += x
    wheel1[1,:] += y
    plt.plot(wheel1[0,:],wheel1[1,:],truckcolor)

    # wheel2[0,:] += 1.0
    wheel2[1,:] -= 1.3
    wheel2 = np.matmul(rotate,wheel2)
    wheel2[0,:] += x
    wheel2[1,:] += y
    plt.plot(wheel2[0,:],wheel2[1,:],truckcolor)

################################################################################################### dynamic_model
def dynamic_model(velocity, yaw, steer):
    A = np.array([[1.0 , 0.0 , - dt * velocity * math.sin(yaw), dt * math.cos(yaw)],\
                  [0.0 , 1.0 , dt * velocity * math.cos(yaw),  dt * math.sin(yaw)],\
                  [0.0 , 0.0 , 1.0                , dt * math.tan(steer) / W_base],\
                  [0.0 , 0.0 , 0.0 , 1.0]])

    B = np.array([[0.0 , 0.0],\
                  [0.0 , 0.0],\
                  [0.0 , dt * velocity / (W_base * math.cos(steer) ** 2)],\
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

################################################################################################### run_MPC
def run_MPC(traj_des, cur_state_vec, mpc_acc, mpc_steer, goal):

    for iter in range(3):
        traj_pred = calc_predicted_trajectory(mpc_acc, mpc_steer, cur_state_vec)
        x = cp.Variable([Nx, H+1]) #(4,6)
        u = cp.Variable([Nu, H])   #(2,5)

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
        mpc_acc = u.value[0, :]
        mpc_steer = u.value[1, :]

    return mpc_x, mpc_y, mpc_acc, mpc_steer

################################################################################################### cal_desired_trajectory
def cal_desired_trajectory(cur_state_vec, path_x, path_y, path_yaw):
    traj_des = np.zeros((Nx,H+1))   #[4, 6]
    distance = 0
    total_pts = len(path_x)

    target_pt = get_closest_point_on_path(path_x, path_y, cur_state_vec)

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
    return traj_des, target_pt

################################################################################################### get_closest_point_on_path
def get_closest_point_on_path(path_x, path_y, cur_state_vec):
    diff_x = []
    diff_y = []
    dist_sq = []

    for i in range(len(path_x)):
        diff_x.append(path_x[i] - cur_state_vec[0])
        diff_y.append(path_y[i] - cur_state_vec[1])
        dist_sq.append((diff_x[i])**2+(diff_y[i])**2)

    min_d = min(dist_sq)
    temp = np.argwhere(dist_sq == min_d)
    target_pt = int(temp[0])
    return target_pt

################################################################################################### destination_check
def destination_check(state, goal, target_pt, length_path):
    a = 0
    dist_to_dest = (state.x - goal[0])**2 + (state.y - goal[1])**2
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
def run_controller(path_robot1, path_robot2, goal1, goal2, obstacle_state, path_planning):
    current_state1 = State(path_robot1[0][0], path_robot1[1][0], path_robot1[2][0], 0.0)
    current_state2 = State(path_robot2[0][0], path_robot2[1][0], path_robot2[2][0], 0.0)
    imgct = 0

    #Initialize variables to store actual state values of car
    t = [0]

    #plotにしか用いてない
    x1 = [current_state1.x]
    y1 = [current_state1.y]
    x2 = [current_state2.x]
    y2 = [current_state2.y]

    mpc_acc1 = np.zeros(H)
    mpc_steer1 = np.zeros(H)
    mpc_acc2 = np.zeros(H)
    mpc_steer2 = np.zeros(H)

    cur_state_vec1 = current_state1.state_to_vec()
    cur_state_vec2 = current_state2.state_to_vec()

    while t[-1] <= simulation_time_limit:
        imgct += 1
        cur_state_vec1 = current_state1.state_to_vec()
        cur_state_vec2 = current_state2.state_to_vec()

        traj_des1, target_pt1 = cal_desired_trajectory(cur_state_vec1, path_robot1[0], path_robot1[1], path_robot1[2])
        traj_des2, target_pt2 = cal_desired_trajectory(cur_state_vec2, path_robot2[0], path_robot2[1], path_robot2[2])

        mpc_x1, mpc_y1, mpc_acc1, mpc_steer1 = run_MPC(traj_des1, cur_state_vec1, mpc_acc1, mpc_steer1, goal1)
        mpc_x2, mpc_y2, mpc_acc2, mpc_steer2 = run_MPC(traj_des2, cur_state_vec2, mpc_acc2, mpc_steer2, goal2)

        current_state1.update_state(mpc_acc1[0], mpc_steer1[0])
        current_state2.update_state(mpc_acc2[0], mpc_steer2[0])

        time = t[-1] + dt
        t.append(time)

        #plotにしか用いてない
        x1.append(current_state1.x)
        y1.append(current_state1.y)
        x2.append(current_state2.x)
        y2.append(current_state2.y)

        if destination_check(current_state1, goal1, target_pt1, len(path_robot1[0])) and destination_check(current_state2, goal2, target_pt2, len(path_robot2[1])):
            print("Reached destination")
            break

        plt.cla()

        plt.plot(mpc_x1, mpc_y1, "xr", label="MPC1")                             #赤色のバツ印
        plt.plot(path_robot1[0], path_robot1[1], "-r", label="course1")                        #赤色の直線
        plt.plot(obstacle_state[0], obstacle_state[1], "ok")                                                  #黒色の丸印
        # plt.plot(x1, y1, "ob", label="trajectory1")                              #青色の丸印
        plt.plot(goal1[0], goal1[1], "om")                                      #紫色の丸印
        # plt.plot(traj_des1[0, :], traj_des1[1, :], "xk", label="xref1")          #黒色のバツ印
        # plt.plot(path_robot1[target_pt1], path_robot1[target_pt1], "xg", label="target1")  #緑色のバツ印
        plot_car(current_state1.x, current_state1.y, current_state1.yaw, mpc_steer1[0])

        plt.plot(mpc_x2, mpc_y2, "xr", label="MPC2")                             #赤色のバツ印
        plt.plot(path_robot2[0], path_robot2[1], "-r", label="course2")                        #赤色の直線
        # plt.plot(x2, y2, "ob", label="trajectory2")                              #青色の丸印
        plt.plot(goal2[0], goal2[1], "og")                                      #紫色の丸印
        # plt.plot(traj_des2[0, :], traj_des2[1, :], "xk", label="xref2")          #黒色のバツ印
        # plt.plot(path_robot2[target_pt2], path_robot2[target_pt2], "xg", label="target2")  #緑色のバツ印
        plot_car(current_state2.x, current_state2.y, current_state2.yaw, mpc_steer2[0])

        plt.axis("equal")
        plt.grid(True)
        plt.title("Time[s]:" + str(round(time, 2)))

        if save_simulation:
            plt.savefig('Q_'+str(imgct))
        plt.pause(0.0001)

        test_state1 = np.array([current_state1.x, current_state1.y])
        test_state2 = np.array([current_state2.x, current_state2.y])

        # goal2 = np.array([path_robot1[0][target_pt1] - 5, path_robot1[1][target_pt1] - 5])

        if path_planning:
            test1 = Path(test_state1, goal1, obstacle_state)
            test2 = Path(test_state2, goal2, obstacle_state)
            path_robot1, obstacle_state = test1.potential_field_path()
            path_robot2, obstacle_state = test2.potential_field_path()
            if stop_planning(path_robot1[0], path_robot1[1], goal1[0], goal1[1]) and stop_planning(path_robot2[0], path_robot2[1], goal2[0], goal2[1]):
                path_planning = False

def stop_planning(path_x, path_y, goal_x, goal_y):
    dist_to_dest = (path_x[-1] - goal_x)**2 + (path_y[-1] - goal_y)**2
    if dist_to_dest < accept_dist:
        return True
    else:
        return False

def initialize_obstacles(NUM_OF_OBSTACLES):
    if get_obstacle:
        ox = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        oy = [0, 10, 20, 25, 30, 35, 40, 45, 50]
        velX = [-0.02, -0.02, -0.02, -0.02, -0.02]
        velY = [0, 0, 0, 0, 0]
        # ox = [0]
        # oy = [0]
        # velX = [0.05]
        # velY = [0.05]
    else:
        ox = []
        oy = []
        velX = []
        velY = []

        for i in range(1,NUM_OF_OBSTACLES):
             ox.append(random.randrange(5, 30, 1))
             oy.append(random.randrange(5, 30, 1))
             velX.append((np.random.random()/70)*(-1)**i)
             velY.append((np.random.random()/70)*(-1)**i)

    obstacle_state = np.array([ox, oy, velX, velY])
    return obstacle_state

################################################################################################### main
def main():
    initial_obstacle_state = initialize_obstacles(NUM_OF_OBSTACLES)  #dimention = 2

    robot1 = initial_param(start_x1, start_y1, goal_x1, goal_y1)
    robot2 = initial_param(start_x2, start_y2, goal_x2, goal_y2)
    # robot2 = initial_param(start_x2, start_y2, start_x1 - 5, start_y1  - 5)

    start_robot1 = robot1.start()  #dimention = 1
    start_robot2 = robot2.start()  #dimention = 1
    goal_robot1 = robot1.goal()     #dimention = 1
    goal_robot2 = robot2.goal()     #dimention = 1

    test1 = Path(start_robot1, goal_robot1, initial_obstacle_state)
    test2 = Path(start_robot2, goal_robot2, initial_obstacle_state)
    path_robot1, obstacle_state = test1.potential_field_path()
    path_robot2, obstacle_state = test2.potential_field_path()

    run_controller(path_robot1, path_robot2, goal_robot1, goal_robot2, obstacle_state, path_planning)

if __name__ == '__main__':
    main()