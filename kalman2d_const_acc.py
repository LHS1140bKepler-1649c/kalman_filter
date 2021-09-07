import numpy as np
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
import pandas as pd

def log(_str='', value=None, logging=False):
    if logging:
        print(f'{_str}: {value}')

logging_flag = False

class Kalman2D:
    def __init__(self):
        self.x = 0                  # m
        self.y = 0                  # m
        self.vx = 0                 # m/s
        self.vy = 0                 # m/s
        self.ax_const = 60          # m/s^2
        self.ay_const = 30          # m/s^2
        self.ax = self.ax_const     # m/s^2
        self.ay = self.ay_const     # m/s^2
        self.dt = 0.066             # s
        self.vec = np.array([[0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0]])
        self.ax_sigma = 0.5
        self.ay_sigma = 0.5
        self.VelCov = np.array([[self.ax_sigma**2, 0, self.ax_sigma**2, 0, self.ax_sigma**2, 0],
                                [0, self.ay_sigma**2, 0, self.ay_sigma**2, 0, self.ay_sigma**2],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]])
        self.P = np.array([[1000, 0, 0, 0, 0, 0],
                            [0, 1000, 0, 0, 0, 0],
                            [0, 0, 1000, 0, 0, 0],
                            [0, 0, 0, 1000, 0, 0],
                            [0, 0, 0, 0, 1000, 0],
                            [0, 0, 0, 0, 0, 1000]])
        self.A = np.array([[1, 0, self.dt, 0, (self.dt**2)/2, 0],
                            [0, 1, 0, self.dt, 0, (self.dt**2)/2],
                            [0, 0, 1, 0, self.dt, 0],
                            [0, 0, 0, 1, 0, self.dt],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0]])
        self.HT = self.H.T
        self.R = np.array([[10, 0],
                            [0, 10]])
        self.Q_red_discrete = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0, (self.dt**2)/2, 0],
                                [0, (self.dt**4)/4, 0, (self.dt**3)/2, 0, (self.dt**2)/2],
                                [(self.dt**3)/2, 0, self.dt**2, 0, self.dt, 0],
                                [0, (self.dt**3)/2, 0, self.dt**2, 0, self.dt],
                                [(self.dt**2)/2, 0, self.dt, 0, 1, 0],
                                [0, (self.dt**2)/2, 0, self.dt, 0, 1]])
        self.Q_red_continous = np.array([[(self.dt**5)/20, 0, (self.dt**4)/8, 0, (self.dt**3)/6, 0],
                                [0, (self.dt**5)/20, 0, (self.dt**4)/8, 0, (self.dt**3)/6],
                                [(self.dt**4)/8, 0, (self.dt**3)/3, 0, (self.dt**2)/2, 0],
                                [0, (self.dt**4)/8, 0, (self.dt**3)/3, 0, (self.dt**2)/2],
                                [(self.dt**3)/6, 0, (self.dt**2)/2, 0, self.dt, 0],
                                [0, (self.dt**3)/6, 0, (self.dt**2)/2, 0, self.dt]])
        self.Q = self.Q_red_continous.dot(self.VelCov)

    def getMeasurement(self):
        sigma_x = self.ax_sigma * np.random.randn(1)
        sigma_y = self.ay_sigma * np.random.randn(1)
        z_x = self.x + self.vx * self.dt + sigma_x
        z_y = self.y + self.vy * self.dt + sigma_y
        self.x = z_x - sigma_x
        self.y = z_y - sigma_y
        return [z_x, z_y, self.x, self.y, sigma_x, sigma_y]

    def writeMeasurement(self, num_of_data=10000, name='measurement_2d_data_linear_const_acc.csv'):
        t = np.linspace(0, 10, num=num_of_data + 1)
        num_of_measurements = len(t)
        time = 0
        meas_time = list()
        meas_x = list()
        meas_y = list()
        actual_x = list()
        actual_y = list()
        sigma_x = list()
        sigma_y = list()
        actual_vx = list()
        actual_vy = list()
        time_list = list()
        for k in range(1, num_of_measurements):
            z = self.getMeasurement()
            time += self.dt
            time_list.append(time)
            meas_time.append(k)
            meas_x.append(z[0][0])
            meas_y.append(z[1][0])
            actual_x.append(z[2][0])
            actual_y.append(z[3][0])
            sigma_x.append(z[4][0])
            sigma_y.append(z[5][0])
            actual_vx.append(self.vx)
            actual_vy.append(self.vy)
        meas_dict = {'time_steps': meas_time, 'x_m': meas_x, 'y_m': meas_y, 'x_a': actual_x, 'y_a': actual_y, \
            'sigma_x_m': sigma_x, 'sigma_y_m': sigma_y, 'vx_a': actual_vx, 'vy_a': actual_vy, 'time': time_list}
        df = pd.DataFrame.from_dict(meas_dict)
        df.to_csv(name, index=False)

    def filterFromMeasurements(self, measurement_data='measurement_2d_data_linear_const_acc.csv',
                                out_data_name='output_2d_data_linear_const_acc.csv'):
        df = pd.read_csv(measurement_data)
        x_m = list(df['x_m'])
        y_m = list(df['y_m'])
        est_x = list()
        est_y = list()
        est_vx = list()
        est_vy = list()

        for i in range(len(x_m)):
            z = [[x_m[i]], [y_m[i]]]
            log('Pre-filter z shape', np.shape(z), logging_flag)
            # Call Filter and return new State
            f = self.filter(z)
            est_x.append(f[0][0])
            est_y.append(f[1][0])
            est_vx.append(f[2][0])
            est_vy.append(f[3][0])

        df['x_e'] = est_x
        df['y_e'] = est_y
        df['vx_e'] = est_vx
        df['vy_e'] = est_vy
        df.to_csv(out_data_name, index=False)

    def filter(self, z):
        # Predict State Forward
        log('State vector shape', self.vec.shape, logging_flag)
        x_p = self.A.dot(self.vec)
        log('x_p shape', x_p.shape, logging_flag)
        # Predict Covariance Forward
        P_p = self.A.dot(self.P).dot(self.A.T) + self.Q
        log('P_p shape', P_p.shape, logging_flag)
        # Compute Innovation
        S = self.H.dot(P_p).dot(self.HT) + self.R
        log('S shape', S.shape, logging_flag)
        # Compute Kalman Gain
        K = P_p.dot(self.HT).dot(inv(S))
        log('K shape', K.shape, logging_flag)
        # Estimate State
        log('H shape', self.H.shape, logging_flag)
        transformed = self.H.dot(x_p)
        log('Transformed shape', transformed.shape, logging_flag)
        residual = z - self.H.dot(x_p)
        log('residual shape', residual.shape, logging_flag)
        self.vec = x_p + K.dot(residual)
        # Estimate Covariance
        self.P = P_p - K.dot(self.H).dot(P_p)

        return [self.vec[0], self.vec[1], self.vec[2], self.vec[3], self.P]

    def calculateErrors(self, data_name='output_2d_data_linear_const_acc.csv'):
        df = pd.read_csv(data_name)
        x_est_diff_squared_sum = 0
        y_est_diff_squared_sum = 0
        x_meas_diff_squared_sum = 0
        y_meas_diff_squared_sum = 0
        for i in range(df.shape[0]):
            x_est_diff_squared_sum += (df['x_a'][i] - df['x_e'][i])**2
            y_est_diff_squared_sum += (df['y_a'][i] - df['y_e'][i])**2
            x_meas_diff_squared_sum += (df['x_a'][i] - df['x_m'][i])**2
            y_meas_diff_squared_sum += (df['y_a'][i] - df['y_m'][i])**2
        x_est_rmse = np.sqrt(x_est_diff_squared_sum / df.shape[0])
        y_est_rmse = np.sqrt(y_est_diff_squared_sum / df.shape[0])
        x_meas_rmse = np.sqrt(x_meas_diff_squared_sum / df.shape[0])
        y_meas_rmse = np.sqrt(y_meas_diff_squared_sum / df.shape[0])
        print(f'RMSE of estimated position x: {x_est_rmse}')
        print(f'RMSE of estimated position y: {y_est_rmse}')
        print(f'RMSE of measured position x: {x_meas_rmse}')
        print(f'RMSE of measured position y: {y_meas_rmse}')
        print('\n')
        print(f'Ratio of estimated and measured x: {x_meas_rmse / x_est_rmse}')
        print(f'Ratio of estimated and measured y: {y_meas_rmse / y_est_rmse}')




if __name__ == '__main__':

    logging_flag = False
    K = Kalman2D()
    K.writeMeasurement()
    K.filterFromMeasurements()
    K.calculateErrors()