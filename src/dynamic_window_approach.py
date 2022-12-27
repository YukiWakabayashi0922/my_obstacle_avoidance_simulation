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
    # def __init__(self, leader_goal, follower_goal, robot_state, obstacle_state):
    def __init__(self, leader_goal, robot_state, obstacle_state):
        self.leader_goal = leader_goal        # leader goal
        # self.follower_goal = follower_goal    # follower goal
        self.robot_state = robot_state        # robot state
        self.obstacle_state = obstacle_state  # obstacle state

    def publish_state(self, u, dt):
        self.robot_state[2] += u[1] * dt
        self.robot_state[0] += u[0] * math.cos(self.robot_state[2]) * dt
        self.robot_state[1] += u[0] * math.sin(self.robot_state[2]) * dt
        self.robot_state[3] = u[0]
        self.robot_state[4] = u[1]

        return self.robot_state

    def motion(self, x, u, dt):
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]

        return x

    def calc_dynamic_window(self, config):
        Vs = [config.min_speed, config.max_speed, -config.max_yawrate, config.max_yawrate]
        Vd = [self.robot_state[3] - config.max_accel * config.dt,
              self.robot_state[3] + config.max_accel * config.dt,
              self.robot_state[4] - config.max_dyawrate * config.dt,
              self.robot_state[4] + config.max_dyawrate * config.dt]
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

    def calc_final_input(self, u, dw, config):
        xinit = self.robot_state[:]
        min_cost = 10000.0
        min_u = u
        min_u[0] = 0.0
        best_traj = np.array([xinit])

        for v in np.arange(dw[0], dw[1], config.v_reso):
            for y in np.arange(dw[2], dw[3], config.yawrate_reso):
                traj = DWA.calc_trajectory(self, xinit, v, y, config)

                to_goal_cost = DWA.calc_to_goal_cost(self, traj, config)
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
            for i in range(len(self.ob[:, 0])):
                ox = self.ob[i, 0]
                oy = self.ob[i, 1]
                dx = traj[ii, 0] - ox
                dy = traj[ii, 1] - oy

                r = math.sqrt(dx**2 + dy**2)
                if r <= config.robot_radius:
                    return float("Inf")  # collision

                if minr >= r:
                    minr = r

        return 1.0 / minr  # OK


    def calc_to_goal_cost(self, traj, config):
        # calc to goal cost. It is 2D norm.

        goal_magnitude = math.sqrt(self.goal[0]**2 + self.goal[1]**2)
        traj_magnitude = math.sqrt(traj[-1, 0]**2 + traj[-1, 1]**2)
        dot_product = (self.goal[0] * traj[-1, 0]) + (self.goal[1] * traj[-1, 1])
        error = dot_product / (goal_magnitude * traj_magnitude)
        error_angle = math.acos(error)
        cost = config.to_goal_cost_gain * error_angle

        return cost

    def dwa_control(self, u, config):
        dw = DWA.calc_dynamic_window(self, config)
        u, traj = DWA.calc_final_input(self, u, dw, config)
        publish_state = DWA.publish_state(self, u, config.dt)

        return u, traj, publish_state


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), head_length=width, head_width=width)
    plt.plot(x, y)

# def main(gx=10, gy=10):
def main():
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    robot_state1 = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # x2 = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # goal position [x(m), y(m)]
    # goal = np.array([gx, gy])
    leader_goal = np.array([10, 10])
    # goal2 = np.array([11, 10])
    # obstacles [x(m) y(m), ....]
    obstacle_state = np.array([[-1, -1],
                   # [0, 2],
                   [4.0, 2.0],
                   [5.0, 4.0],
                   [5.0, 5.0],
                   [5.0, 6.0],
                   [5.0, 9.0],
                   [8.0, 9.0],
                   [7.0, 9.0],
                   [12.0, 12.0]
                   ])

    u1 = np.array([0.0, 0.0])
    # u2 = np.array([0.0, 1.0])
    config = Config()
    traj1 = np.array(robot_state1)
    # traj2 = np.array(x2)

    robot1_path = True
    # robot2_path = True

    for i in range(1000):
        if robot1_path == True:
            dwa1 = DWA(leader_goal, robot_state1, obstacle_state)
            u1, ltraj1, robot_state1 = dwa1.dwa_control(u1, config)
            traj1 = np.vstack((traj1, robot_state1))  # store state history
            # print("ltraj1 : ", ltraj1)

        # if robot2_path == True:
        #     dwa2 = DWA(goal2, ob)
            # u2, ltraj2, x2 = dwa2.dwa_control(x2, u2, config)
            # traj2 = np.vstack((traj2, x2))  # store state history

        # print("u1 : ", u1, "u2 : ", u2)

        if show_animation:
            plt.cla()
            plt.plot(ltraj1[:, 0], ltraj1[:, 1], "-r")
            plt.plot(robot_state1[0], robot_state1[1], "xr")
            plt.plot(leader_goal[0], leader_goal[1], "xb")
            # plt.plot(ltraj2[:, 0], ltraj2[:, 1], "-g")
            # plt.plot(x2[0], x2[1], "xg")
            # plt.plot(goal2[0], goal2[1], "xb")
            plt.plot(obstacle_state[:, 0], obstacle_state[:, 1], "ok")
            plot_arrow(robot_state1[0], robot_state1[1], robot_state1[2])
            # plot_arrow(x2[0], x2[1], x2[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check goal
        robot1_check = math.sqrt((robot_state1[0] - leader_goal[0])**2 + (robot_state1[1] - leader_goal[1])**2)
        # robot2_check = math.sqrt((x2[0] - goal2[0])**2 + (x2[1] - goal2[1])**2)
        if robot1_check <= config.robot_radius:
            robot1_path = False

        # if robot2_check <= config.robot_radius:
            # robot2_path = False

        # if robot1_path == False and robot2_path == False:
        if robot1_path == False:
        # if robot2_path == False:
            # if math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2) <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(traj1[:, 0], traj1[:, 1], "-r")
        # plt.plot(traj2[:, 0], traj2[:, 1], "-g")
        plt.pause(0.0001)

    plt.show()


if __name__ == '__main__':
    main()
