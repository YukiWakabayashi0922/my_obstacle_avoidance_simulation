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

# robot_number = 2        # simulation robot number
robot_number = 3        # simulation robot number
obstacle_number = 20

# robot parameter
class Config():
    def __init__(self):
        self.max_speed    = 0.5  # [m/s]
        self.min_speed    = 0.0  # [m/s]
        self.max_yawrate  =  60.0 * math.pi / 180.0  # [rad/s]
        self.min_yawrate  = -60.0 * math.pi / 180.0  # [rad/s]
        self.max_accel    = 0.5  # [m/ss]
        self.max_dyawrate = 180.0 * math.pi / 180.0  # [rad/ss]
        self.vel_reso     = 0.02  # [m/s]
        self.yawrate_reso = 2*math.pi / 180.0  # [rad/s]
        self.dt           = 0.1  # [s]
        self.predict_time = 2.0  # [s]
        self.heading_weight = 0.5
        self.obstacle_weight = 0.2
        self.velocity_weight = 0.4
        self.robot_distance_cost = 2.5
        self.robot_radius = 0.3  # [m]

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
        dw = Predict_path.calc_dynamic_window(self)
        # print(dw)
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
        angle = math.atan2(goal[1] - current_path[1][-1], goal[0] - current_path[0][-1])
        cost_angle = angle - current_path[2][-1]

        if cost_angle > math.pi:
            while cost_angle > math.pi:
                cost_angle -= 2 * math.pi
        elif cost_angle < -math.pi:
            while cost_angle < -math.pi:
                cost_angle += 2 * math.pi

        cost = abs(cost_angle)
        return cost

        # goal_magnitude = math.hypot(goal[0], goal[1])
        # path_magnitude = math.hypot(current_path[0][-1], current_path[1][-1])
        # dot_product = (goal[0] * current_path[0][-1]) + (goal[1] * current_path[1][-1])
        # error = dot_product / (goal_magnitude * path_magnitude)
        # error_angle = math.acos(error)
        # return error_angle

    def calc_obstacle(self, current_path, robot_state):
        min_dist = float("inf")

        for i in range(len(current_path[0])):
            for j in range(len(self.obstacle_state)):
                temp_dist_to_obstacle = math.hypot(current_path[0][i] - self.obstacle_state[j][0], current_path[1][i] - self.obstacle_state[j][1])

                if temp_dist_to_obstacle < self.config.robot_radius:
                    return float("inf")

                if temp_dist_to_obstacle <= min_dist:
                    min_dist = temp_dist_to_obstacle

        return 1.0 / min_dist

    def calc_velocity(self, velocity, current_number):
        distance_robot = math.hypot(self.all_robot_state[current_number][0] - self.all_robot_state[current_number + 1][0], self.all_robot_state[current_number][1] - self.all_robot_state[current_number + 1][1])
        distance_cost = self.config.robot_distance_cost - distance_robot # 離れると-

        speed_cost = self.config.max_speed - velocity # 速いと小さくなる

        if distance_cost <= 0.0:
            cost = -speed_cost * abs(distance_cost)
        else:
            cost = speed_cost * abs(distance_cost)

        return cost

    def calc_velocity_last(self, velocity):
        cost = self.config.max_speed - velocity
        return cost

    def new_robot_state(self, current_robot_state, vel, yawrate):
        current_robot_state[2] += yawrate * self.config.dt
        current_robot_state[0] += vel * math.cos(current_robot_state[2]) * self.config.dt
        current_robot_state[1] += vel * math.sin(current_robot_state[2]) * self.config.dt
        current_robot_state[3] = vel
        current_robot_state[4] = yawrate

        return current_robot_state

    def min_max_normalize(self, data):
        data = np.array(data)

        max_data = max(data)
        min_data = min(data)

        if max_data - min_data == 0.0:
            data = [0.0 for i in range(len(data))]
        else:
            data = (data - min_data) / (max_data - min_data)

        return data

    def dwa_control(self):
        publish_best_path = []
        publish_robot_state = []
        min_dwa_cost = float("inf")

        for i in range(robot_number):
            current_goal = Dynamic_Windou_Approach.individual_goal(self, i)
            current_path = Dynamic_Windou_Approach.individual_path(self, i)
            current_robot_state = Dynamic_Windou_Approach.individual_robot_state(self, i)

            if self.flag_path[i] == False:
                publish_best_path.append(current_path)
                publish_robot_state.append(current_robot_state)

            else:
                heading = []
                obstacle = []
                velocity = []

                if i == robot_number - 1:  # 最後尾のロボット
                    for j in range(len(current_path)):
                        heading.append(Dynamic_Windou_Approach.calc_heading(self, current_goal, current_path[j]))
                        obstacle.append(Dynamic_Windou_Approach.calc_obstacle(self, current_path[j], current_robot_state))
                        velocity.append(Dynamic_Windou_Approach.calc_velocity_last(self, current_path[j][3]))


                    heading = Dynamic_Windou_Approach.min_max_normalize(self, heading)
                    # obstacle = Dynamic_Windou_Approach.min_max_normalize(self, obstacle)
                    velocity = Dynamic_Windou_Approach.min_max_normalize(self, velocity)
                    # print(velocity)

                    min_dwa_cost = float("inf")

                    for j in range(len(current_path)):
                        dwa_cost = self.config.heading_weight * heading[j] + self.config.obstacle_weight * obstacle[j] + self.config.velocity_weight * velocity[j]

                        if min_dwa_cost >= dwa_cost:
                            min_dwa_cost = dwa_cost
                            best_path = current_path[j]

                    new_state = Dynamic_Windou_Approach.new_robot_state(self, current_robot_state, best_path[3], best_path[4])
                    publish_best_path.append(best_path)
                    publish_robot_state.append(new_state)

                else:
                    for j in range(len(current_path)):
                        heading.append(Dynamic_Windou_Approach.calc_heading(self, current_goal, current_path[j]))
                        obstacle.append(Dynamic_Windou_Approach.calc_obstacle(self, current_path[j], current_robot_state))
                        # velocity.append(Dynamic_Windou_Approach.calc_velocity(self, current_path[j][3], i))
                        velocity.append(Dynamic_Windou_Approach.calc_velocity_last(self, current_path[j][3]))

                    # print(velocity)
                    heading = Dynamic_Windou_Approach.min_max_normalize(self, heading)
                    # obstacle = Dynamic_Windou_Approach.min_max_normalize(self, obstacle)
                    velocity = Dynamic_Windou_Approach.min_max_normalize(self, velocity)
                    # if i == 0:
                        # print(len(heading))
                        # print(len(obstacle))
                        # print(heading)
                        # print(obstacle)
                        # print(velocity)

                    min_dwa_cost = float("inf")

                    for j in range(len(current_path)):
                        dwa_cost = self.config.heading_weight * heading[j] + self.config.obstacle_weight * obstacle[j] + self.config.velocity_weight * velocity[j]
                        # print(current_path[j][3])

                        if min_dwa_cost >= dwa_cost:
                            min_dwa_cost = dwa_cost
                            best_path = current_path[j]

                    # print(best_path[3])
                        # print("yes_norm: ", heading[j], obstacle[j], velocity[j])

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

    def plot_do(self):
        plt.cla()

        for i in range(len(self.obstacle_state)):
            plt.plot(self.obstacle_state[i][0], self.obstacle_state[i][1], "ok")

        for i in range(robot_number):
            current_goal = Plot.individual_goal(self, i)
            current_path = Plot.individual_path(self, i)
            current_robot_state = Plot.individual_robot_state(self, i)

            plt.plot(current_robot_state[0], current_robot_state[1], "og")
            plt.plot(current_goal[0], current_goal[1], "xb")

            if self.flag_path[i] == True:
                plt.plot(current_path[0], current_path[1], "-m")

        distance12 = math.hypot(self.all_robot_state[0][0] - self.all_robot_state[1][0], self.all_robot_state[0][1] - self.all_robot_state[1][1])
        distance23 = math.hypot(self.all_robot_state[1][0] - self.all_robot_state[2][0], self.all_robot_state[1][1] - self.all_robot_state[2][1])
        # print(distance)

        if distance12 <= self.config.robot_distance_cost:
            plt.plot([self.all_robot_state[0][0], self.all_robot_state[1][0]], [self.all_robot_state[0][1], self.all_robot_state[1][1]], "g")
        else:
            plt.plot([self.all_robot_state[0][0], self.all_robot_state[1][0]], [self.all_robot_state[0][1], self.all_robot_state[1][1]], "r")
            # print("dis12!")

        if distance23 <= self.config.robot_distance_cost:
            plt.plot([self.all_robot_state[1][0], self.all_robot_state[2][0]], [self.all_robot_state[1][1], self.all_robot_state[2][1]], "g")
        else:
            plt.plot([self.all_robot_state[1][0], self.all_robot_state[2][0]], [self.all_robot_state[1][1], self.all_robot_state[2][1]], "r")

        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

class Obstacle_update():
    def __init__(self, all_obstacle_state, all_obstacle_velocity):
        self.all_obstacle_state = all_obstacle_state
        self.all_obstacle_velocity = all_obstacle_velocity

    def individual_obstacle_state(self, current_number):
        obstacle_state = []
        obstacle_state = self.all_obstacle_state[current_number]
        return obstacle_state

    def individual_obstacle_velocity(self, current_number):
        obstacle_velocity = []
        obstacle_velocity = self.all_obstacle_velocity[current_number]
        return obstacle_velocity

    def calc_new_state(self):
        publish_obstacle_state = []

        for i in range(len(self.all_obstacle_state)):
            current_obstacle_state = Obstacle_update.individual_obstacle_state(self, i)
            current_obstacle_velocity = Obstacle_update.individual_obstacle_velocity(self, i)

            new_state = [current_obstacle_state[0] + current_obstacle_velocity[0], current_obstacle_state[1] + current_obstacle_velocity[1]]
            publish_obstacle_state.append(new_state)

        return publish_obstacle_state

def get_obstacle_state():
    publish_obstacle_state = []

    for i in range(obstacle_number):
        obstacle_state = [random.randrange(3, 15, 1), random.randrange(8, 12, 1)]
        publish_obstacle_state.append(obstacle_state)

    return publish_obstacle_state

def get_obstacle_velocity():
    publish_obstacle_velocity = []

    for i in range(obstacle_number):
        obstacle_velocity = [random.randrange(-2, 2, 1) / 100, random.randrange(-2, 2, 1) / 100]
        publish_obstacle_velocity.append(obstacle_velocity)

    return publish_obstacle_velocity

def main():
    print(__file__ + " start!!")

    robot_state = [[0.0, 10.0, 0.0, 0.0, 0.0],
                   [-2.5, 10.0, 0.0, 0.0, 0.0],
                   [-5.0, 10.0, 0.0, 0.0, 0.0]]

    obstacle_state = [[7, 8], [8, 9], [9, 11], [9, 9], [10, 12], [11, 8], [12, 9], [12, 13]]
    obstacle_velocity = [[-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.0], [-0.01, 0.0], [-0.01, -0.01], [-0.01, 0.01], [-0.01, 0.0], [-0.01, -0.01]]
    # obstacle_state = get_obstacle_state()
    # obstacle_velocity = get_obstacle_velocity()

    goal = [[20, 12], [20, 12], [20, 12]]

    traj1 = np.array(robot_state[0])
    traj2 = np.array(robot_state[1])
    traj3 = np.array(robot_state[2])

    dist_12 = np.array([])
    dist_23 = np.array([])
    dist_13 = np.array([])
    t = np.array([])

    flag_path = [True, True, True]

    config = Config()

    for i in range(1000):
        distance_12 = math.hypot(robot_state[0][0] - robot_state[1][0], robot_state[0][1] - robot_state[1][1])
        distance_23 = math.hypot(robot_state[1][0] - robot_state[2][0], robot_state[1][1] - robot_state[2][1])
        distance_13 = math.hypot(robot_state[0][0] - robot_state[2][0], robot_state[0][1] - robot_state[2][1])

        robot1_check = math.hypot(robot_state[0][0] - goal[0][0], robot_state[0][1] - goal[0][1])
        robot2_check = math.hypot(robot_state[1][0] - goal[1][0], robot_state[1][1] - goal[1][1])
        robot3_check = math.hypot(robot_state[2][0] - goal[2][0], robot_state[2][1] - goal[2][1])

        if robot1_check <= config.robot_radius:
            flag_path[0] = False
        else:
            flag_path[0] = True

        if robot2_check <= config.robot_radius:
            flag_path[1] = False
        else:
            flag_path[1] = True

        if robot3_check <= config.robot_radius:
            flag_path[2] = False
        else:
            flag_path[2] = True

        pp1 = Predict_path(robot_state[0])
        path1 = pp1.get_predict_path()

        pp2 = Predict_path(robot_state[1])
        path2 = pp2.get_predict_path()

        pp3 = Predict_path(robot_state[2])
        path3 = pp3.get_predict_path()

        path = [path1, path2, path3]

        dwa = Dynamic_Windou_Approach(goal, path, robot_state, obstacle_state, flag_path)
        best_path, robot_state = dwa.dwa_control()
        print("1:", robot_state[0][3], " 2:", robot_state[1][3], " 3:", robot_state[2][3])

        plot = Plot(goal, best_path, robot_state, obstacle_state, flag_path)
        plot.plot_do()

        traj1 = np.vstack((traj1, robot_state[0]))
        traj2 = np.vstack((traj2, robot_state[1]))
        traj3 = np.vstack((traj3, robot_state[2]))

        dist_12 = np.append(dist_12, distance_12)
        dist_23 = np.append(dist_23, distance_23)
        dist_13 = np.append(dist_13, distance_13)
        t = np.append(t, i)

        # if flag_path[0] == False and flag_path[1] == False:
        if flag_path[0] == False and flag_path[1] == False and flag_path[2] == False:
            print("Goal!!")
            break

        goal = [[20, 10], [robot_state[0][0] - 0.5, robot_state[0][1]], [robot_state[1][0] - 0.5, robot_state[1][1]]]
        ob_update = Obstacle_update(obstacle_state, obstacle_velocity)
        obstacle_state = ob_update.calc_new_state()

    plt.plot(traj1[:, 0], traj1[:, 1], "-r")
    plt.plot(traj2[:, 0], traj2[:, 1], "-r")
    plt.plot(traj3[:, 0], traj3[:, 1], "-r")
    plt.pause(0.0001)

    plt.subplots()
    for i in range(len(t)):
        plt.plot(t[i:i+2], dist_12[i:i+2], color="red" if dist_12[i] >= config.robot_distance_cost else "blue")
    plt.hlines(config.robot_distance_cost, 0, t[-1], "k")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")

    plt.subplots()
    for i in range(len(t)):
        plt.plot(t[i:i+2], dist_23[i:i+2], color="red" if dist_23[i] >= config.robot_distance_cost else "blue")
    plt.hlines(config.robot_distance_cost, 0, t[-1], "k")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")

    plt.subplots()
    for i in range(len(t)):
        plt.plot(t[i:i+2], dist_13[i:i+2], color="red" if dist_13[i] >= 2*config.robot_distance_cost else "blue")
    plt.hlines(2*config.robot_distance_cost, 0, t[-1], "k")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.show()


if __name__ == "__main__":
    main()
