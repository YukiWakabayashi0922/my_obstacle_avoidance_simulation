"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random

import sys
sys.path.append("../../")

show_animation = True   # show trajectory
# show_animation = False  # don't show trajectory

robot_number = 2        # simulation robot number

# robot parameter
class Config():
    def __init__(self):
        self.max_speed    =  1.0  # [m/s]
        self.min_speed    = -1.0  # [m/s]
        self.max_yawrate  =  40.0 * math.pi / 180.0  # [rad/s]
        self.min_yawrate  = -40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel    = 0.2  # [m/ss]
        self.max_dyawrate = 20.0 * math.pi / 180.0  # [rad/ss]
        self.vel_reso     = 0.01  # [m/s]
        self.yawrate_reso = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt           = 0.1  # [s]
        self.predict_time = 2.0  # [s]
        self.to_goal_cost_gain = 0.7
        self.speed_cost_gain   = 1.0
        self.robot_radius = 0.5  # [m]
        self.select_obstacle_radius = 4.0  # [m]
        # self.obstacle_update_x = 0.1  # [m/ss]
        # self.obstacle_update_y = 0.1  # [m/ss]


# ロボットの予測軌跡を算出するクラス
class Predict_path():
    def __init__(self, state):
        self.state = state      # [x, y, yaw, velocity, yawrate]
        self.config = Config()

    def calc_dynamic_window(self):
        Vs = [self.config.min_speed, self.config.max_speed, self.config.min_yawrate, self.config.max_yawrate]
        Vd = [self.state[3] - self.config.max_accel * self.config.dt,
              self.state[3] + self.config.max_accel * self.config.dt,
              self.state[4] - self.config.max_dyawrate * self.config.dt,
              self.state[4] + self.config.max_dyawrate * self.config.dt]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def calc_path(self, current_state, vel, yawrate):
        path_x = []
        path_y = []
        path_yaw = []
        time = 0

        while time <= self.config.predict_time:
            temp_x = vel * math.cos(current_state[2]) * self.config.dt + current_state[0]
            temp_y = vel * math.sin(current_state[2]) * self.config.dt + current_state[1]
            temp_yaw = yawrate * self.config.dt + current_state[2]

            path_x.append(temp_x)
            path_y.append(temp_y)
            path_yaw.append(temp_yaw)

            current_state[0] = temp_x
            current_state[1] = temp_y
            current_state[2] = temp_yaw

            time += self.config.dt

        path = [path_x, path_y, path_yaw, vel, yawrate]

        return path

    def get_predict_path(self):
        dw = Predict_path().calc_dynamic_window(self)
        publish_path = []

        for vel in np.arange(dw[0], dw[1], self.config.vel_reso):
            for yawrate in np.arange(dw[2], dw[3], self.config.yawrate_reso):
                current_state = [self.state[0], self.state[1], self.state[2]]
                path = Predict_path().calc_path(self, current_state, vel, yawrate)

                publish_path.append(path)

        return publish_path

# 評価関数の計算
class Dynamic_Windou_Approach():
    def __init__(self, all_goal, all_path, all_robot_state, obstacle_state, flag_path):
        self.all_goal = all_goal
        self.all_path = all_path
        self.all_robot_state = all_robot_state
        self.obstacle_state = obstacle_state
        self.flag_path = flag_path
        self.config = Config()

    def individual_goal(self, current_number):
        goal = []
        goal = self.all_goal[current_number]
        return goal

    def individual_path(self, current_number):
        path = []
        path = self.all_path[current_number]
        return path

    def individual_robot_state(self, current_number):
        robot_state = []
        robot_state = self.all_robot_state[current_number]
        return robot_state

    def calc_heading(self, goal, current_path):
        last_x = current_path[0][-1]
        last_y = current_path[1][-1]
        last_yaw = current_path[2][-1]

        angle_to_goal = math.atan2(goal[0] - last_x, goal[1] - last_y)
        score_angle = angle_to_goal - last_yaw

        if score_angle > math.pi:
            while score_angle > math.pi:
                score_angle -= 2 * math.pi
        elif score_angle < -math.pi:
            while score_angle < -math.pi:
                score_angle += 2 * math.pi

        cost = abs(score_angle) / math.pi
        return cost

    def select_obstacle(self, robot_state):
        select_obstacle_state = []

        for i in range(len(self.obstacle_state)):
            temp_dist_to_obstacle = math.sqrt(math.hypot(robot_state[0] - self.obstacle_state[i][0], robot_state[1] - self.obstacle_state[i][1]))

            if temp_dist_to_obstacle < self.config.select_obstacle_radius:
                select_obstacle_state.append(self.obstacle_state[i])

        return select_obstacle_state

    def calc_obstacle(self, current_path, robot_state):
        select_obstacle_state = Dynamic_Windou_Approach().select_obstacle(self, robot_state)
        min_dist = float("inf")
        max_dist = -float("inf")

        for i in range(len(current_path[0])):
            for j in range(len(select_obstacle_state)):
                temp_dist_to_obstacle = math.sqrt(math.hypot(current_path[0][i] - select_obstacle_state[j][0], current_path[1][i] - select_obstacle_state[j][1]))

                if temp_dist_to_obstacle < self.config.robot_radius:
                    return float("inf")

                if temp_dist_to_obstacle <= min_dist:
                    min_dist = temp_dist_to_obstacle

                if temp_dist_to_obstacle >= max_dist:
                    max_dist = temp_dist_to_obstacle

        cost = min_dist / max_dist
        return cost

    # def calc_velocity(self):


    def calc_velocity_last(self, velocity):
        cost = (self.config.max_speed - velocity) / self.config.max_speed
        return cost

    def new_robot_state(self, current_robot_state, vel, yawrate):
        current_robot_state[2] += yawrate * self.config.dt
        current_robot_state[0] += vel * math.cos(current_robot_state[2]) * self.config.dt
        current_robot_state[1] += vel * math.sin(current_robot_state[2]) * self.config.dt
        current_robot_state[3] = vel
        current_robot_state[4] = yawrate

        return current_robot_state

    def dwa_control(self):
        publish_best_path = []
        publish_robot_state = []

        for i in range(robot_number):
            current_goal = Dynamic_Windou_Approach().individual_goal(self, i)
            current_path = Dynamic_Windou_Approach().individual_path(self, i)
            current_robot_state = Dynamic_Windou_Approach().individual_robot_state(self, i)

            if self.flag_path[i] == False:
                publish_best_path.append(current_path)
                publish_robot_state.append(current_robot_state)
            else:
                if i == robot_number - 1:  # 最後尾のロボット
                    for j in range(current_path):
                        heading = Dynamic_Windou_Approach().calc_heading(self, current_goal, current_path[j])
                        obstacle = Dynamic_Windou_Approach().calc_obstacle(self, current_path[j], current_robot_state)
                        velocity = Dynamic_Windou_Approach().calc_velocity_last(self, current_path[j][3])

                        dwa_cost = heading + obstacle + velocity

                        if min_dwa_cost >= dwa_cost:
                            min_dwa_cost = dwa_cost
                            best_path = current_path[j]

                    new_state = Dynamic_Windou_Approach().new_robot_state(self, current_robot_state, best_path[3], best_path[4])
                    publish_best_path.append(best_path)
                    publish_robot_state.append(new_state)

                else:
                    for j in range(current_path):
                        heading = Dynamic_Windou_Approach().calc_heading(self, current_goal, current_path[j])
                        obstacle = Dynamic_Windou_Approach().calc_obstacle(self, current_path, current_robot_state)
                        velocity = Dynamic_Windou_Approach().calc_velocity_last(self, current_path[j][3])
                        # velocity = Dynamic_Windou_Approach().calc_velocity(self, )

                        dwa_cost = heading + obstacle + velocity

                        if min_dwa_cost >= dwa_cost:
                            min_dwa_cost = dwa_cost
                            best_path = current_path[j]

                    new_state = Dynamic_Windou_Approach().new_robot_state(self, current_robot_state, best_path[3], best_path[4])
                    publish_best_path.append(best_path)
                    publish_robot_state.append(new_state)

        return publish_best_path, publish_robot_state

class Plot():
    def __init__(self, all_goal, all_path, all_robot_state, obstacle_state, flag_path):
        self.all_goal = all_goal
        self.all_path = all_path
        self.all_robot_state = all_robot_state
        self.obstacle_state = obstacle_state
        self.flag_path = flag_path

    def individual_goal(self, current_number):
        goal = []
        goal = self.all_goal[current_number]
        return goal

    def individual_path(self, current_number):
        path = []
        path = self.all_path[current_number]
        return path

    def individual_robot_state(self, current_number):
        robot_state = []
        robot_state = self.all_robot_state[current_number]
        return robot_state

    def plot_do(self):
        plt.cla()

        for i in range(len(self.obstacle_state)):
            plt.plot(self.obstacle_state[i][0], self.obstacle_state[i][1], "ok")

        for i in range(robot_number):
            current_goal = Dynamic_Windou_Approach().individual_goal(self, i)
            current_path = Dynamic_Windou_Approach().individual_path(self, i)
            current_robot_state = Dynamic_Windou_Approach().individual_robot_state(self, i)

            plt.plot(current_robot_state[0], current_robot_state[1], "or")
            plt.plot(goal[0], goal[1], "xb")

            if self.flag_path[i] == True:
                plt.plot(current_path[0], current_path[1], "-r")

        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

def main():
    print(__file__ + " start!!")

    robot_state = [[0.0, 0.0, 0.0, 0.0, 0.0],
                   [-5.0, 0.0, 0.0, 0.0, 0.0]]

    obstacle_state = [[5, 5], [5, 7], [5, 9]]

    goal = [[10, 10], [10, 10]]

    traj1 = np.array(initial_robot_state[0])
    traj2 = np.array(initial_robot_state[1])

    flag_path = [True, True]

    config = Config()

    for i in range(1000):
        robot1_check = math.sqrt(math.hypot(robot_state[0][0] - goal[0][0], robot_state[0][1] - goal[0][1]))
        robot2_check = math.sqrt(math.hypot(robot_state[1][0] - goal[1][0], robot_state[1][1] - goal[1][1]))

        if robot1_check <= config.robot_radius:
            flag_path[0] = False
        else:
            flag_path[0] = True

        if robot2_check <= config.robot_radius:
            flag_path[1] = False
        else:
            flag_path[1] = True

        pp = Predict_path(robot_state)
        path = pp.get_predict_path()

        dwa = Dynamic_Windou_Approach(goal, path, robot_state, obstacle_state, flag_path)
        best_path, robot_state = dwa.dwa_control()

        plt = Plot(goal, best_path, robot_state, obstacle_state, flag_path)
        plt.plot_do()

        traj1 = np.vstack((traj1, robot_state[0]))
        traj2 = np.vstack((traj2, robot_state[1]))

        if flag_path[0] == False and flag_path[1] == False:
            print("Goal!!")
            break

        goal = [[10, 10], [robot_state[0], robot_state[1]]]

    plt.plot(traj1[:, 0], traj1[:, 1], "-r")
    plt.plot(traj2[:, 0], traj2[:, 1], "-r")
    plt.pause(0.0001)

    plt.show()

if __name__ == "__main__":
    main()




class DWA:
    def __init__(self, all_goal, all_robot_state, obstacle_state, flag_path):
        self.all_goal = all_goal                # goal
        self.all_robot_state = all_robot_state  # robot state
        self.obstacle_state = obstacle_state    # obstacle state
        self.flag_path = flag_path              # flag path do

    def individual_goal(self, current_number):
        goal = []
        goal = self.all_goal[current_number]
        return goal

    def individual_robot_state(self, current_number):
        robot_state = []
        robot_state = self.all_robot_state[current_number]
        return robot_state

    def publish_state(self, robot_state, u, dt):
        robot_state[2] += u[1] * dt
        robot_state[0] += u[0] * math.cos(robot_state[2]) * dt
        robot_state[1] += u[0] * math.sin(robot_state[2]) * dt
        robot_state[3] = u[0]
        robot_state[4] = u[1]

        return robot_state

    def motion(self, x, u, dt):
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]

        return x

    def calc_dynamic_window(self, robot_state, config):
        Vs = [config.min_speed, config.max_speed, config.min_yawrate, config.max_yawrate]
        Vd = [robot_state[3] - config.max_accel * config.dt,
              robot_state[3] + config.max_accel * config.dt,
              robot_state[4] - config.max_dyawrate * config.dt,
              robot_state[4] + config.max_dyawrate * config.dt]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def calc_trajectory(self, xinit, v, y, config):
        x = np.array(xinit)
        traj = np.array(x)
        time = 0
        while time <= config.predict_time:
            x = DWA.motion(self, x, [v, y], config.dt)
            traj = np.vstack((traj, x))
            time += config.dt

        return traj

    def calc_final_input(self, goal, robot_state, u, dw, config):
        xinit = robot_state[:]
        min_cost = 10000.0
        min_u = u
        min_u[0] = 0.0
        best_traj = np.array([xinit])

        for v in np.arange(dw[0], dw[1], config.v_reso):
            for y in np.arange(dw[2], dw[3], config.yawrate_reso):
                traj = DWA.calc_trajectory(self, xinit, v, y, config)

                to_goal_cost = DWA.calc_to_goal_cost(self, goal, traj, config)
                speed_cost = config.speed_cost_gain * (config.max_speed - traj[-1, 3])
                ob_cost = DWA.calc_obstacle_cost(self, traj, config)
                # print("gcost", to_goal_cost, "scost", speed_cost, "obcost", ob_cost)

                final_cost = 0.5*to_goal_cost + speed_cost + ob_cost

                if min_cost >= final_cost:
                    min_cost = final_cost
                    min_u = [v, y]
                    best_traj = traj

        return min_u, best_traj

    def calc_obstacle_cost(self, traj, config):
    # calc obstacle cost inf: collistion, 0:free

        skip_n = 1
        minr = float("inf")

        for ii in range(0, len(traj[:, 1]), skip_n):
            for i in range(len(self.obstacle_state)):
                ox = self.obstacle_state[i][0]
                oy = self.obstacle_state[i][1]
                dx = traj[ii, 0] - ox
                dy = traj[ii, 1] - oy

                r = math.sqrt(dx**2 + dy**2)
                if r <= config.robot_radius:
                    return float("Inf")  # collision

                if minr >= r:
                    minr = r

        return 1.0 / minr  # OK ぶつかりそうなほど大きい値


    def calc_to_goal_cost(self, goal, traj, config):
        # calc to goal cost. It is 2D norm.

        goal_magnitude = math.sqrt(goal[0]**2 + goal[1]**2)
        traj_magnitude = math.sqrt(traj[-1, 0]**2 + traj[-1, 1]**2)
        dot_product = (goal[0] * traj[-1, 0]) + (goal[1] * traj[-1, 1])
        error = dot_product / (goal_magnitude * traj_magnitude)
        error_angle = math.acos(error)
        cost = config.to_goal_cost_gain * error_angle

        return cost

    # def ovservation_cost(self, ):
        # self.

    def obstacle_update(self):
        publish_obstacle_state = []
        obstacle_update_x = 0.01
        obstacle_update_y = -0.03
        # obstacle_update_x = np.random.random()/10
        # obstacle_update_y = np.random.random()/10

        for i in range(len(self.obstacle_state)):
            self.obstacle_state[i][0] += obstacle_update_x
            self.obstacle_state[i][1] += obstacle_update_y

            publish_obstacle_state.append([self.obstacle_state[i][0], self.obstacle_state[i][1]])

        return publish_obstacle_state

    def dwa_control(self, u, config):
        publish_new_u = []
        publish_traj = []
        publish_state = []

        for i in range(robot_number):
            goal = DWA.individual_goal(self, i)
            robot_state = DWA.individual_robot_state(self, i)

            if self.flag_path[i] == True:
                dw = DWA.calc_dynamic_window(self, robot_state, config)
                new_u, traj = DWA.calc_final_input(self, goal, robot_state, u[i], dw, config)
                new_state = DWA.publish_state(self, robot_state, new_u, config.dt)

                publish_new_u.append(new_u)
                publish_traj.append(traj)
                publish_state.append(new_state)

            else:
                dw = DWA.calc_dynamic_window(self, robot_state, config)
                new_u, traj = DWA.calc_final_input(self, goal, robot_state, u[i], dw, config)

                publish_new_u.append(u[i])
                publish_traj.append(traj)
                publish_state.append(robot_state)

        return publish_new_u, publish_traj, publish_state

def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), head_length=width, head_width=width)
    plt.plot(x, y)

def main():
    print(__file__ + " start!!")

    robot_state = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [-5.0, 0.0, 0.0, 0.0, 0.0]])

    obstacle_state =[[4.0, 13.0],
                     [4.0, 12.0],
                     [4.0, 11.0],
                     [4.0, 10.0]]

    u = np.array([[0.0, 0.0],
                  [0.0, 0.0]])

    config = Config()

    traj1 = np.array(robot_state[0])
    traj2 = np.array(robot_state[1])

    flag_path = []
    for i in range(robot_number):
        flag_path.append(True)

    for i in range(1000):
        goal = np.array([[10, 10], [robot_state[0][0] - 0.3, robot_state[0][1]]])

        dwa = DWA(goal, robot_state, obstacle_state, flag_path)
        u, ltraj, robot_state = dwa.dwa_control(u, config)
        obstacle_state = dwa.obstacle_update()
        print(ltraj)
        # print("robot1 v:", ltraj[0][3], " yaw:", ltraj[0][4])
        # print("robot2 v:", ltraj[1][3], " yaw:", ltraj[1][4])

        traj1 = np.vstack((traj1, robot_state[0]))
        traj2 = np.vstack((traj2, robot_state[1]))

        robot1_check = math.sqrt((robot_state[0][0] - goal[0][0])**2 + (robot_state[0][1] - goal[0][1])**2)
        robot2_check = math.sqrt((robot_state[1][0] - goal[1][0])**2 + (robot_state[1][1] - goal[1][1])**2)

        if robot1_check <= config.robot_radius:
            flag_path[0] = False
        else:
            flag_path[0] = True

        if robot2_check <= config.robot_radius:
            flag_path[1] = False
        else:
            flag_path[1] = True

        if show_animation:
            plt.cla()
            plt.plot(robot_state[0][0], robot_state[0][1], "xr")
            plt.plot(goal[0][0], goal[0][1], "xb")

            plt.plot(robot_state[1][0], robot_state[1][1], "xg")
            plt.plot(goal[1][0], goal[1][1], "xb")

            for i in range(len(obstacle_state)):
                plt.plot(obstacle_state[i][0], obstacle_state[i][1], "ok")

            if flag_path[0] == True:
                plt.plot(ltraj[0][:, 0], ltraj[0][:, 1], "-r")

            if flag_path[1] == True:
                plt.plot(ltraj[1][:, 0], ltraj[1][:, 1], "-g")

            # plot_arrow(robot_state1[0], robot_state1[1], robot_state1[2])
            # plot_arrow(x2[0], x2[1], x2[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        if flag_path[0] == False and flag_path[1] == False:
            print("Goal!!")
            break

        # goal[1][0] = 0.0
        # goal[1][1] = 0.0

        # print(robot_state[0][0])

        # current_leader_x = robot_state[0][0]
        # current_leader_y = robot_state[0][1]

        # goal[1][0] = current_leader_x - 0.3
        # goal[1][1] = current_leader_y - 0.3

        # current_leader_x = 0.0
        # current_leader_y = 0.0

    print("Done")
    if show_animation:
        plt.plot(traj1[:, 0], traj1[:, 1], "-r")
        plt.plot(traj2[:, 0], traj2[:, 1], "-g")
        plt.pause(0.0001)

    plt.show()


if __name__ == '__main__':
    main()
