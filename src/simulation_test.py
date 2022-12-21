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

get_obstacle = True
# get_obstacle = False
save_simulation = False

robot_number = 3

Nx = 4 #number of states
Nu = 2 #number of control inputs
dt = 0.5
DT = 0.2
W_track = 1.5 #wheel track in metre
W_base = 3.5 #wheel base in metre

N_search = 10 # search for closest point in the next N_search points on the path
H = 5 # Horizon length
simulation_time_limit = 100 #seconds
accept_dist = 0.5 #acceptable destination distance
accept_stop_v = 0.08 #acceptable stopping velocity
accept_robot_distance = 8.0

desired_speed = 0.1           # m/s
max_speed = 1              # m/s
max_reverse_speed = 0.5       # m/s
max_steer_angle = np.pi / 4     #max steering angle
max_steer_rate = np.pi / 6      #max steering speed
max_acc = 0.2                   #maximum acceleration m/s^2

W1 = np.array([0.01, 0.01])  # input weightage
W2 = np.array([2.0, 2.0, 0.5, 0.5])  # state error weightage
W3 = np.array([0.01, 0.1])  # rate of input change weightage
W4 = W2  # state error weightage

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
        start_robot = []
        for i in range(robot_number):
            start_robot.append([self.start_x[i], self.start_y[i]])
        return start_robot

    def goal(self):
        goal_robot = []
        for i in range(robot_number):
            goal_robot.append([self.goal_x[i], self.goal_y[i]])
        return goal_robot

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

#Path : robot path
class Path:
    def __init__(self, all_current_state, all_goal, obstacle_state, path_planning):
        self.all_current_state = all_current_state    #[[x, y], [x, y], [x, y]]
        self.all_goal = all_goal                      #[[x, y], [x, y], [x, y]]
        self.obstacle_x = obstacle_state[0]           #[x]
        self.obstacle_y = obstacle_state[1]           #[y]
        self.obstacle_velocity_x = obstacle_state[2]  #[Vx]
        self.obstacle_velocity_y = obstacle_state[3]  #[Vy]
        self.path_planning = path_planning

        self.KP = 30
        self.ETA = 100
        self.FP = 100

    def individual_current_state(self, current_number):
        current_state = []
        current_state = self.all_current_state[current_number]
        return current_state

    def individual_goal(self, current_number):
        goal = []
        goal = self.all_goal[current_number]
        return goal

    def calc_goal_potential(self, predictX, predictY, goal):
        return 0.5 * self.KP * np.hypot(predictX - goal[0], predictY - goal[1])

    def calc_obstacle_avoidance_potential(self, predictX, predictY):
        obs = len(self.obstacle_x)
        pot = 0

        for i in range(obs):
            pot += 0.5 * self.ETA * np.exp(-np.hypot(predictX - self.obstacle_x[i], predictY - self.obstacle_y[i]))

        return pot

    # def calc_formation_potential(self, predictX, predictY, count):
        # return 0.5 * self.FP * np.exp(-np.hypot(predictX - self.current[count + 1][0], predictY - self.current[count + 1][1]))

    def get_motion_model(self):
        motion = []
        num = 360
        for i in range(num):
            deg = i*np.pi/num
            motion.append([np.cos(deg),np.sin(deg)])

        return motion

    def get_spline_path(self, path_x, path_y):
        x, y, yaw, k, s = cubic_spline_planner.calc_spline_course(path_x, path_y, 1)
        return x, y, yaw

    def leader_planning(self, current_state, goal):
        motion = Path.get_motion_model(self)
        predictX = []
        predictY = []

        for i in range(len(motion)):
            predictX.append(current_state[0] + motion[i][0] * grid_size)
            predictY.append(current_state[1] + motion[i][1] * grid_size)

        gnet = []
        min_gnet = 0
        min_gnet_pos = 0

        for i in range(len(motion)):
            gg = Path.calc_goal_potential(self, predictX[i], predictY[i], goal)
            go = Path.calc_obstacle_avoidance_potential(self, predictX[i], predictY[i])
            gnet.append(gg + go)

            if(i==0):
                min_gnet = gnet[i]
                min_gnet_pos = i
            else:
                if(min_gnet > gnet[i]):
                    min_gnet = gnet[i]
                    min_gnet_pos = i

        step_x = motion[min_gnet_pos][0] / 1.5
        step_y = motion[min_gnet_pos][1] / 1.5

        step = [step_x, step_y]
        return step

    def middle_follower_planning(self, current_state, goal):
        motion = Path.get_motion_model(self)
        predictX = []
        predictY = []

        for i in range(len(motion)):
            predictX.append(current_state[0] + motion[i][0] * grid_size)
            predictY.append(current_state[1] + motion[i][1] * grid_size)

        gnet = []
        min_gnet = 0
        min_gnet_pos = 0

        for i in range(len(motion)):
            gg = Path.calc_goal_potential(self, predictX[i], predictY[i], goal)
            go = Path.calc_obstacle_avoidance_potential(self, predictX[i], predictY[i])
            gnet.append(gg + go)

            if(i==0):
                min_gnet = gnet[i]
                min_gnet_pos = i
            else:
                if(min_gnet > gnet[i]):
                    min_gnet = gnet[i]
                    min_gnet_pos = i

        step_x = motion[min_gnet_pos][0] / 1.5
        step_y = motion[min_gnet_pos][1] / 1.5

        step = [step_x, step_y]
        return step

    def last_follower_planning(self, current_state, goal):
        motion = Path.get_motion_model(self)
        predictX = []
        predictY = []

        for i in range(len(motion)):
            predictX.append(current_state[0] + motion[i][0] * grid_size)
            predictY.append(current_state[1] + motion[i][1] * grid_size)

        gnet = []
        min_gnet = 0
        min_gnet_pos = 0

        for i in range(len(motion)):
            gg = Path.calc_goal_potential(self, predictX[i], predictY[i], goal)
            go = Path.calc_obstacle_avoidance_potential(self, predictX[i], predictY[i])
            gnet.append(gg + go)

            if(i==0):
                min_gnet = gnet[i]
                min_gnet_pos = i
            else:
                if(min_gnet > gnet[i]):
                    min_gnet = gnet[i]
                    min_gnet_pos = i

        step_x = motion[min_gnet_pos][0] / 1.5
        step_y = motion[min_gnet_pos][1] / 1.5

        step = [step_x, step_y]
        return step

    def leader_path(self, current_state, goal, current_number):
        leader_path = []
        path_x = []
        path_y = []
        iter = 0

        if self.path_planning[current_number] == True:
            while iter <= H:
                step = Path.leader_planning(self, current_state, goal)

                current_state[0] = current_state[0] + step[0]
                current_state[1] = current_state[1] + step[1]

                path_x.append(current_state[0])
                path_y.append(current_state[1])

                iter += 1

            path_x, path_y, path_yaw = Path.get_spline_path(self, path_x, path_y)
            leader_path = [path_x, path_y, path_yaw]

        else:
            path_yaw = []
            while iter <= H:
                path_x.append(current_state[0])
                path_y.append(current_state[1])
                path_yaw.append(0)

                iter += 1

            leader_path = [path_x, path_y, path_yaw]

        return leader_path

    def middle_follower_path(self, current_state, goal, current_number):
        middle_follower_path = []
        path_x = []
        path_y = []
        iter = 0

        if self.path_planning[current_number] == True:
            while iter <= H:
                step = Path.middle_follower_planning(self, current_state, goal)

                current_state[0] = current_state[0] + step[0]
                current_state[1] = current_state[1] + step[1]

                path_x.append(current_state[0])
                path_y.append(current_state[1])

                iter += 1

            path_x, path_y, path_yaw = Path.get_spline_path(self, path_x, path_y)
            middle_follower_path = [path_x, path_y, path_yaw]

        else:
            path_yaw = []
            while iter <= H:
                path_x.append(current_state[0])
                path_y.append(current_state[1])
                path_yaw.append(0)

                iter += 1

            middle_follower_path = [path_x, path_y, path_yaw]

        return middle_follower_path

    def last_follower_path(self, current_state, goal, current_number):
        last_follower_path = []
        path_x = []
        path_y = []
        iter = 0

        if self.path_planning[current_number] == True:
            while iter <= H:
                step = Path.last_follower_planning(self, current_state, goal)

                current_state[0] = current_state[0] + step[0]
                current_state[1] = current_state[1] + step[1]

                path_x.append(current_state[0])
                path_y.append(current_state[1])

                iter += 1

            path_x, path_y, path_yaw = Path.get_spline_path(self, path_x, path_y)
            last_follower_path = [path_x, path_y, path_yaw]

        else:
            path_yaw = []
            while iter <= H:
                path_x.append(current_state[0])
                path_y.append(current_state[1])
                path_yaw.append(0)

                iter += 1

            last_follower_path = [path_x, path_y, path_yaw]

        return last_follower_path

    def obstacle_path(self):
        iter = 0

        while iter <= H:
            self.obstacle_x = np.add(self.obstacle_x, self.obstacle_velocity_x)
            self.obstacle_y = np.add(self.obstacle_y, self.obstacle_velocity_y)

            iter += 1

        obstacle_path = np.array([self.obstacle_x, self.obstacle_y, self.obstacle_velocity_x, self.obstacle_velocity_y])
        return obstacle_path

    def path_do(self):
        current_state = []
        goal = []
        path = []
        leader_path = []
        middle_follower_path = []
        last_follower_path = []

        for i in range(robot_number):
            current_state = Path.individual_current_state(self, i)
            goal = Path.individual_goal(self, i)

            if i == 0:
                leader_path = Path.leader_path(self, current_state, goal, i)
                path.append(leader_path)

            elif i == robot_number - 1:
                last_follower_path = Path.last_follower_path(self, current_state, goal, i)
                path.append(last_follower_path)

            else:
                middle_follower_path = Path.middle_follower_path(self, current_state, goal, i)
                path.append(middle_follower_path)

            current_state = []
            goal = []

        obstacle_path = Path.obstacle_path(self)

        return path, obstacle_path

class Controller:
    def __init__(self, all_path, all_goal, all_mpc_acc, all_mpc_steer, obstacle_state):
        self.all_path = all_path                      #[[x, y], [x, y], [x, y]]
        self.all_goal = all_goal                      #[[x, y], [x, y], [x, y]]
        self.all_mpc_acc = all_mpc_acc                #[[acc], [acc], [acc]]
        self.all_mpc_steer = all_mpc_steer            #[[steer], [steer], [steer]]
        self.obstacle_x = obstacle_state[0]           #[x]
        self.obstacle_y = obstacle_state[1]           #[y]
        self.obstacle_velocity_x = obstacle_state[2]  #[Vx]
        self.obstacle_velocity_y = obstacle_state[3]  #[Vy]

    def individual_path(self, current_number):
        path = []
        path = self.all_path[current_number]
        return path

    def individual_goal(self, current_number):
        goal = []
        goal = self.all_goal[current_number]
        return goal

    def individual_mpc_acc(self, current_number):
        mpc_acc = []
        mpc_acc = self.all_mpc_acc[current_number]
        return mpc_acc

    def individual_mpc_steer(self, current_number):
        mpc_steer = []
        mpc_steer = self.all_mpc_steer[current_number]
        return mpc_steer

    def get_closest_point_on_path(self, cur_state_vec, path):
        diff_x = []
        diff_y = []
        dist_sq = []

        for i in range(len(path[0])):
            diff_x.append(path[0][i] - cur_state_vec[0])
            diff_y.append(path[1][i] - cur_state_vec[1])
            dist_sq.append((diff_x[i])**2 + (diff_y[i])**2)

        min_d = min(dist_sq)
        temp = np.argwhere(dist_sq == min_d)
        target_pt = int(temp[0])
        return target_pt

    def cal_desired_trajectory(self, cur_state_vec, path):
        traj_des = np.zeros((Nx,H+1))   #[4, 6]
        distance = 0
        total_pts = len(path[0])

        target_pt = Controller.get_closest_point_on_path(self, cur_state_vec, path)

        traj_des[0,0] = path[0][target_pt]
        traj_des[1,0] = path[1][target_pt]
        traj_des[2,0] = path[2][target_pt]
        traj_des[3,0] = desired_speed         #5 m/s

        for i in range(H):
            distance += abs(cur_state_vec[3]) * dt
            pts_travelled = int(round(distance))

            if (target_pt + pts_travelled) < total_pts:
                traj_des[0,i+1] = path[0][target_pt + pts_travelled]
                traj_des[1,i+1] = path[1][target_pt + pts_travelled]
                traj_des[2,i+1] = path[2][target_pt + pts_travelled]
                if (target_pt + pts_travelled) == total_pts - 1:
                    traj_des[3,i+1] = 0.0
                else:
                    traj_des[3,i+1] = desired_speed
            else:
                traj_des[0,i+1] = path[0][-1]
                traj_des[1,i+1] = path[1][-1]
                traj_des[2,i+1] = path[2][-1]
                traj_des[3,i+1] = 0.0

        if traj_des[3,1] == 0.0:
            traj_des[3,0] = 0.0

        return traj_des, target_pt

    def calc_predicted_trajectory(self, cur_state_vec, mpc_acc, mpc_steer):
        traj_pred = np.zeros((Nx,H+1))  #Nx = 4, H = 5
        traj_pred[:,0] = cur_state_vec.T
        pred_state = State(cur_state_vec[0], cur_state_vec[1], cur_state_vec[2], cur_state_vec[3])

        for i in range(H):
            pred_state.update_state(mpc_acc[i], mpc_steer[i])
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
            traj_pred = Controller.calc_predicted_trajectory(self, cur_state_vec, mpc_acc, mpc_steer)
            x = cp.Variable([Nx, H+1]) #(4,6)
            u = cp.Variable([Nu, H])   #(2,5)

            cost = 0.0
            constraints = []
            for i in range(H):
                cost += cp.sum(W1 @ cp.square(u[:, i]))                                   # input weightage
                cost += cp.sum(W2 @ cp.square(traj_des[:, i] - x[:, i]))                  # state error weightage
                if i < (H - 1):
                    cost += cp.sum(W3 @ cp.square(u[:, i+1] - u[:, i]))                    # rate of input change weightage
                    constraints += [cp.abs(u[1, i+1] - u[1, i]) <= max_steer_rate * dt]

                A,B,C = Controller.dynamic_model(self, traj_pred[3,i], traj_pred[2,i], mpc_steer[i])
                constraints += [x[:, i+1] == A @ x[:, i] + B @ u[:, i] + C]

            cost += cp.sum(W4 @ cp.square(traj_des[:, H] - x[:, H]))                      # final state error weightage

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

    def controller_do(self):
        all_target_pt = []
        all_mpc_state = []
        all_mpc_acc = []
        all_mpc_steer = []
        all_current_state = []
        path = []
        goal = []
        mpc_acc = []
        mpc_steer = []

        for i in range(robot_number):
            path = Controller.individual_path(self, i)
            goal = Controller.individual_goal(self, i)
            mpc_acc = Controller.individual_mpc_acc(self, i)
            mpc_steer = Controller.individual_mpc_steer(self, i)

            current_state = State(path[0][0], path[1][0], path[2][0], 0.0)
            cur_state_vec = current_state.state_to_vec()

            traj_des, target_pt = Controller.cal_desired_trajectory(self, cur_state_vec, path)
            mpc_x, mpc_y, mpc_acc, mpc_steer = Controller.run_MPC(self, cur_state_vec, traj_des, mpc_acc, mpc_steer)
            mpc_state = [mpc_x, mpc_y]

            current_state.update_state(mpc_acc[0], mpc_steer[0])
            cur_state_vec = current_state.state_to_vec()

            all_target_pt.append(target_pt)
            all_mpc_state.append(mpc_state)
            all_mpc_acc.append(mpc_acc)
            all_mpc_steer.append(mpc_steer)
            all_current_state.append(cur_state_vec)

            path = []
            goal = []
            mpc_acc = []
            mpc_steer = []

        return all_target_pt, all_mpc_state, all_mpc_acc, all_mpc_steer, all_current_state

class Check:
    def __init__(self, current_state, goal, target_pt, path):
        self.path_x = current_state[0]
        self.path_y = current_state[1]
        self.velocity = current_state[3]
        self.goal_x = goal[0]
        self.goal_y = goal[1]
        self.target_pt = target_pt
        self.length = len(path[0])

    def destination_check(self):
        a = 0
        dist_to_dest = (self.path_x - self.goal_x)**2 + (self.path_y - self.goal_y)**2
        if dist_to_dest < accept_dist:
            a += 1
        if self.velocity < abs(accept_stop_v):
            a += 1
        if abs(self.target_pt - self.length) < 5:
            a += 1
        if a == 3:
            return True

        return False

    # def robot_distance_check(self):


    def stop_planning(self):
        dist_to_dest = (self.path_x - self.goal_x)**2 + (self.path_y - self.goal_y)**2
        if dist_to_dest < accept_dist:
            return True
        else:
            return False

class Plot:
    def __init__(self, mpc_state, now_state, goal_robot, obstacle_state):
        self.mpc_state = mpc_state
        self.now_state = now_state
        self.goal_robot = goal_robot
        self.obstacle_x = obstacle_state[0]
        self.obstacle_y = obstacle_state[1]

    def plot_leader_robot(self):
        plt.plot(self.mpc_state[0][0], self.mpc_state[0][1], "or")
        plt.plot(self.now_state[0][0], self.now_state[0][1], "-r")
        plt.plot(self.goal_robot[0][0], self.goal_robot[0][1], "xm")

    # def plot_follower_robot(self):
        # plt.plot(self.mpc_x, self.mpc_y, "og")
        # plt.plot(self.x, self.y, "-g")
        # plt.plot(self.goal_x, self.goal_y, "om")

    def plot_obstacle(self):
        plt.plot(self.obstacle_x, self.obstacle_y, "ok")

    def plot_do(self):
        plt.cla()

        for i in range(robot_number):
            if i == 0:
                Plot.plot_leader_robot(self)
            else:
                plt.plot(self.mpc_state[i][0], self.mpc_state[i][1], "og")
                plt.plot(self.now_state[i][0], self.now_state[i][1], "-g")
                plt.plot(self.goal_robot[i][0], self.goal_robot[i][1], "xm")
                # plot_follower_robot(self)

        Plot.plot_obstacle(self)

        # plt.axis("equal")
        # plt.grid(True)
        # plt.title("Time[s]:" + str(round(time, 2)))

        # if save_simulation:
            # plt.savefig('Q_' + str(imgct))
        # plt.pause(0.0001)


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

def initialize_obstacles(NUM_OF_OBSTACLES):
    if get_obstacle:
        ox = [8, 9, 10, 11, 12, 13, 14, 15]
                # , 16, 17, 18, 19, 20, 21, 22, 23]
        oy = [7, 6, 5, 4, 3, 2, 1, 0]
                # , -2, -3, -4, -5, -6, -7, -8, -9]
        velX = [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05]
                # , -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02]
        velY = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
                # , 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
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
             ox.append(random.randrange(0, 20, 1))
             oy.append(random.randrange(0, 20, 1))
             velX.append((np.random.random()/10)*(-1)**i)
             velY.append((np.random.random()/10)*(-1)**i)

    obstacle_state = np.array([ox, oy, velX, velY])
    return obstacle_state

################################################################################################### main
def main():
    imgct = 0
    path_planning = [True, True, True]

    initial_obstacle_state = initialize_obstacles(NUM_OF_OBSTACLES)  #dimention = 2

    start_x = [0.0, -5.0, -10.0]
    start_y = [9.0, 6.0, 3.0]
    goal_x = [30.0, 40.0, 50.0]
    goal_y = [10.0, 40.0, 50.0]

    robot = initial_param(start_x, start_y, goal_x, goal_y)
    start_robot = robot.start()  #dimention = 1
    goal_robot = robot.goal()     #dimention = 1

    test = Path(start_robot, goal_robot, initial_obstacle_state, path_planning)
    path_robot, obstacle_state = test.path_do()

    t = [0]
    mpc_acc = np.zeros((robot_number, H))
    mpc_steer = np.zeros((robot_number, H))

    while t[-1] <= simulation_time_limit:
        imgct += 1

        control_robot = Controller(path_robot, goal_robot, mpc_acc, mpc_steer, obstacle_state)
        target_pt, mpc_state, mpc_acc, mpc_steer, current_state = control_robot.controller_do()

        time = t[-1] + dt
        t.append(time)

        check1 = Check(current_state[0], goal_robot[0], target_pt[0], path_robot[0])
        check2 = Check(current_state[1], goal_robot[1], target_pt[1], path_robot[1])
        check3 = Check(current_state[2], goal_robot[2], target_pt[2], path_robot[2])
        if check1.destination_check() and check2.destination_check() and check3.destination_check():
            print("Goal!!!")
            # break

        plot = Plot(mpc_state, current_state, goal_robot, obstacle_state)
        plot.plot_do()

        plt.axis("equal")
        plt.grid(True)
        plt.title("Time[s]:" + str(round(time, 2)))

        if save_simulation:
            plt.savefig('Q_' + str(imgct))
        plt.pause(0.0001)

        new_state = [[current_state[0][0], current_state[0][1]], [current_state[1][0], current_state[1][1]], [current_state[2][0], current_state[2][1]]]

        new = Path(new_state, goal_robot, obstacle_state, path_planning)
        path_robot, obstacle_state = new.path_do()

        # if path_planning1:
            # path_robot1 = new1.potential_field_path()

        # if path_planning2:
            # path_robot2 = new2.potential_field_path()

        # if path_planning3:
            # path_robot3 = new3.potential_field_path()

        # robot_distance12 = np.hypot(current_state1[0] - current_state2[0], current_state1[1] - current_state2[1])
        # robot_distance13 = np.hypot(current_state1[0] - current_state3[0], current_state1[1] - current_state3[1])
        # robot_distance23 = np.hypot(current_state2[0] - current_state3[0], current_state2[1] - current_state3[1])

        # if robot_distance12 >= accept_robot_distance:
            # path_planning = [False, True, False]
            # path_planning1 = False
            # path_planning3 = False
        # elif robot_distance13 >= accept_robot_distance:
            # path_planning = [False, False, True]
            # path_planning1 = False
            # path_planning2 = False
        # elif robot_distance23 >= accept_robot_distance:
            # path_planning = [False, False, True]
            # path_planning1 = False
            # path_planning2 = False
        # else:
            # path_planning = [True, True, True]
            # path_planning1 = True
            # path_planning2 = True
            # path_planning3 = True

        # if check1.stop_planning():
            # path_planning = [False, True, True]
            # path_planning1 = False
            # stop1 = True

        # if check2.stop_planning():
            # path_planning = [False, False, True]
            # path_planning2 = False
            # stop2 = True

        # if check3.stop_planning():
            # path_planning = [False, False, False]
            # path_planning3 = False
            # stop3 = True

        # if stop1 and stop2 and stop3:
            # print('Finish!')
            # break

        # goal_robot[1] = np.array([path_robot1[0][0] - 3, path_robot1[1][0] - 3])
        goal_robot[1] = np.array([path_robot[0][0][0] - 3, path_robot[0][1][0]])
        # goal_robot[2] = np.array([path_robot1[0][0] - 3, path_robot1[1][0] + 3])
        goal_robot[2] = np.array([path_robot[1][0][0] - 3, path_robot[1][1][0]])


if __name__ == '__main__':
    main()
