from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt

class Quadrotor():
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, size=0.25, show_animation=True):
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T

        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.show_animation = show_animation

        if self.show_animation:
            plt.ion()
            fig = plt.figure()
            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            self.ax = fig.add_subplot(111, projection='3d')

        self.update_pose(x, y, z, roll, pitch, yaw)

    def update_pose(self, x, y, z, roll, pitch, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)

        if self.show_animation:
            self.plot()

    def transformation_matrix(self):
        x, y, z = self.x, self.y, self.z
        roll, pitch, yaw = self.roll, self.pitch, self.yaw

        # Rotation matrix
        R_yaw = np.array([
            [cos(yaw), -sin(yaw), 0],
            [sin(yaw), cos(yaw), 0],
            [0, 0, 1]
        ])

        R_pitch = np.array([
            [cos(pitch), 0, sin(pitch)],
            [0, 1, 0],
            [-sin(pitch), 0, cos(pitch)]
        ])

        R_roll = np.array([
            [1, 0, 0],
            [0, cos(roll), -sin(roll)],
            [0, sin(roll), cos(roll)]
        ])

        # Combined rotation matrix
        R = np.dot(np.dot(R_yaw, R_pitch), R_roll)

        # Translation vector
        T = np.array([[x, y, z]]).T

        # Homogeneous transformation matrix
        transformation_matrix = np.vstack([np.hstack([R, T]), [0, 0, 0, 1]])

        return transformation_matrix

    def plot(self):  # pragma: no cover
        T = self.transformation_matrix()

        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        plt.cla()

        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.')

        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'r-')
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'r-')

        self.ax.plot(self.x_data, self.y_data, self.z_data, 'b:')

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        self.ax.set_zlim(0, 10)

        plt.pause(0.001)

# Example usage:
if __name__ == "__main__":
    quadrotor = Quadrotor()
    for i in range(100):
        quadrotor.update_pose(x=i * 0.1, y=i * 0.1, z=i * 0.1, roll=i * 0.01, pitch=i * 0.01, yaw=i * 0.01)
