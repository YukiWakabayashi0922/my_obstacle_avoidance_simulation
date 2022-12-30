"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../../")

show_animation = True   # show trajectory
# show_animation = False  # don't show trajectory

robot_number = 2        # simulation robot number

# robot parameter
class Config():
    def __init__(self):
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yawrate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_dyawrate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_reso = 0.01  # [m/s]
        self.yawrate_reso = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s]
        self.predict_time = 2.0  # [s]
        self.to_goal_cost_gain = 1.0
        self.speed_cost_gain = 1.0
        self.robot_radius = 1.0  # [m]

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
        Vs = [config.min_speed, config.max_speed, -config.max_yawrate, config.max_yawrate]
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

                final_cost = to_goal_cost + speed_cost + ob_cost

                if min_cost >= final_cost:
                    min_cost = final_cost
                    min_u = [v, y]
                    best_traj = traj

        return min_u, best_traj

    def calc_obstacle_cost(self, traj, config):
    # calc obstacle cost inf: collistion, 0:free

        skip_n = 2
        minr = float("inf")

        for ii in range(0, len(traj[:, 1]), skip_n):
            for i in range(len(self.obstacle_state[:, 0])):
                ox = self.obstacle_state[i, 0]
                oy = self.obstacle_state[i, 1]
                dx = traj[ii, 0] - ox
                dy = traj[ii, 1] - oy

                r = math.sqrt(dx**2 + dy**2)
                if r <= config.robot_radius:
                    return float("Inf")  # collision

                if minr >= r:
                    minr = r

        return 1.0 / minr  # OK


    def calc_to_goal_cost(self, goal, traj, config):
        # calc to goal cost. It is 2D norm.

        goal_magnitude = math.sqrt(goal[0]**2 + goal[1]**2)
        traj_magnitude = math.sqrt(traj[-1, 0]**2 + traj[-1, 1]**2)
        dot_product = (goal[0] * traj[-1, 0]) + (goal[1] * traj[-1, 1])
        error = dot_product / (goal_magnitude * traj_magnitude)
        error_angle = math.acos(error)
        cost = config.to_goal_cost_gain * error_angle

        return cost

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
    robot_state = np.array([[0.0, 0.0, math.pi / 8.0, 0.0, 0.0],
                            [0.0, 2.0, math.pi / 8.0, 0.0, 0.0]])
    goal = np.array([[10, 10],
                     [12, 12]])
    obstacle_state = np.array([[-1, -1],
                   # [0, 2],
                   [4.0, 2.0],
                   [5.0, 4.0],
                   [5.0, 5.0],
                   [5.0, 6.0],
                   [5.0, 9.0],
                   [8.0, 9.0],
                   [7.0, 9.0],
                   # [12.0, 12.0]
                   ])

    u = np.array([[0.0, 0.0],
                  [0.0, 0.0]])
    config = Config()
    traj1 = np.array(robot_state[0])
    traj2 = np.array(robot_state[1])

    flag_path = np.array([True,
                          True])

    for i in range(1000):
        dwa = DWA(goal, robot_state, obstacle_state, flag_path)
        u, ltraj, robot_state = dwa.dwa_control(u, config)
        traj1 = np.vstack((traj1, robot_state[0]))
        traj2 = np.vstack((traj2, robot_state[1]))

        robot1_check = math.sqrt((robot_state[0][0] - goal[0][0])**2 + (robot_state[0][1] - goal[0][1])**2)
        robot2_check = math.sqrt((robot_state[1][0] - goal[1][0])**2 + (robot_state[1][1] - goal[1][1])**2)

        if robot1_check <= config.robot_radius:
            flag_path[0] = False

        if robot2_check <= config.robot_radius:
            flag_path[1] = False

        if show_animation:
            plt.cla()
            plt.plot(robot_state[0][0], robot_state[0][1], "xr")
            plt.plot(goal[0][0], goal[0][1], "xb")

            plt.plot(robot_state[1][0], robot_state[1][1], "xr")
            plt.plot(goal[1][0], goal[1][1], "xb")

            plt.plot(obstacle_state[:, 0], obstacle_state[:, 1], "ok")

            if flag_path[0] == True:
                plt.plot(ltraj[0][:, 0], ltraj[0][:, 1], "-r")

            if flag_path[1] == True:
                plt.plot(ltraj[1][:, 0], ltraj[1][:, 1], "-r")

            # plot_arrow(robot_state1[0], robot_state1[1], robot_state1[2])
            # plot_arrow(x2[0], x2[1], x2[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        if flag_path[0] == False and flag_path[1] == False:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(traj1[:, 0], traj1[:, 1], "-r")
        plt.plot(traj2[:, 0], traj2[:, 1], "-g")
        plt.pause(0.0001)

    plt.show()


if __name__ == '__main__':
    main()
