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
        self.select_obstacle_radius = 5.0  # [m]
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

    def get_predict_path(self, dw):
        # dw = Predict_path().calc_dynamic_window(self)
        publish_path = []

        for vel in np.arange(dw[0], dw[1], self.config.vel_reso):
            for yawrate in np.arange(dw[2], dw[3], self.config.yawrate_reso):
                current_state = [self.state[0], self.state[1], self.state[2]]
                path = Predict_path.calc_path(self, current_state, vel, yawrate)

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
        select_obstacle_state = [[2, 2], [3, 3]]

        for i in range(len(self.obstacle_state)):
            temp_dist_to_obstacle = math.sqrt(math.hypot(robot_state[0] - self.obstacle_state[i][0], robot_state[1] - self.obstacle_state[i][1]))

            if temp_dist_to_obstacle < self.config.select_obstacle_radius:
                select_obstacle_state.append(self.obstacle_state[i])

        return select_obstacle_state

    def calc_obstacle(self, current_path, robot_state):
        select_obstacle_state = Dynamic_Windou_Approach.select_obstacle(self, robot_state)
        min_dist = float("inf")
        max_dist = -float("inf")

        for i in range(len(current_path[0][0])-1):
            # print(len(current_path[0][0]))
            for j in range(len(select_obstacle_state)):
                # print(current_path[0][i][0])
                temp_dist_to_obstacle = math.sqrt(math.hypot(current_path[0][i][0] - select_obstacle_state[j][0], current_path[1][i][0] - select_obstacle_state[j][1]))

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
            current_goal = Dynamic_Windou_Approach.individual_goal(self, i)
            current_path = Dynamic_Windou_Approach.individual_path(self, i)
            print(len(current_path))
            current_robot_state = Dynamic_Windou_Approach.individual_robot_state(self, i)

            if self.flag_path[i] == False:
                publish_best_path.append(current_path)
                publish_robot_state.append(current_robot_state)
            else:
                if i == robot_number - 1:  # 最後尾のロボット
                    for j in range(len(current_path)):
                        heading = Dynamic_Windou_Approach.calc_heading(self, current_goal, current_path[j])
                        obstacle = Dynamic_Windou_Approach.calc_obstacle(self, current_path[j], current_robot_state)
                        velocity = Dynamic_Windou_Approach.calc_velocity_last(self, current_path[j][3])

                        dwa_cost = heading + obstacle + velocity

                        if min_dwa_cost >= dwa_cost:
                            min_dwa_cost = dwa_cost
                            best_path = current_path[j]

                    new_state = Dynamic_Windou_Approach.new_robot_state(self, current_robot_state, best_path[3], best_path[4])
                    publish_best_path.append(best_path)
                    publish_robot_state.append(new_state)

                else:
                    for j in range(len(current_path)):
                        heading = Dynamic_Windou_Approach.calc_heading(self, current_goal, current_path[j])
                        obstacle = Dynamic_Windou_Approach.calc_obstacle(self, current_path, current_robot_state)
                        velocity = Dynamic_Windou_Approach.calc_velocity_last(self, current_path[j][3])
                        # velocity = Dynamic_Windou_Approach().calc_velocity(self, )

                        dwa_cost = heading + obstacle + velocity

                        if min_dwa_cost >= dwa_cost:
                            min_dwa_cost = dwa_cost
                            best_path = current_path[j]

                    new_state = Dynamic_Windou_Approach.new_robot_state(self, current_robot_state, best_path[3], best_path[4])
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
            current_goal = Dynamic_Windou_Approach.individual_goal(self, i)
            current_path = Dynamic_Windou_Approach.individual_path(self, i)
            current_robot_state = Dynamic_Windou_Approach.individual_robot_state(self, i)

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

    traj1 = np.array(robot_state[0])
    traj2 = np.array(robot_state[1])

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

        pp1 = Predict_path(robot_state[0])
        dw1 = pp1.calc_dynamic_window()
        path1 = pp1.get_predict_path(dw1)

        pp2 = Predict_path(robot_state[1])
        dw2 = pp2.calc_dynamic_window()
        path2 = pp2.get_predict_path(dw2)

        path = [path1, path2]

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
