#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import casadi as ca
from casadi import Function
from nav_msgs.msg import Odometry
import time
import matplotlib.pyplot as plt
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
from acados_template import AcadosModel

class PayloadControlNode(Node):
    def __init__(self):
        super().__init__('PayloadDynamics')

        # Time Definition
        self.ts = 0.03
        self.final = 20
        self.t =np.arange(0, self.final + self.ts, self.ts, dtype=np.double)

        # Internal parameters defintion
        self.robot_num = 3
        self.mass = 0.5
        self.M_load = self.mass *  np.eye((3))
        self.inertia = np.array([[0.013344, 0.0, 0.0], [0.0, 0.012810, 0.0], [0.0, 0.0, 0.03064]], dtype=np.double)
        self.gravity = 9.81

        # Load shape parameters triangle
        self.p1 = np.array([0.20, 0.0, 0.0], dtype=np.double)
        self.p2 = np.array([-0.20, 0.3, 0.0], dtype=np.double)
        self.p3 = np.array([-0.20, -0.3, 0.0], dtype=np.double)
        self.p = np.vstack((self.p1, self.p2, self.p3)).T
        self.e3 = np.array([0.0, 0.0, 1.0], dtype=np.double)

        # Payload odometry
        self.odom_payload_msg = Odometry()
        self.publisher_payload_odom_ = self.create_publisher(Odometry, "odom", 10)

        self.odom_payload_desired_msg = Odometry()
        self.publisher_payload_desired_odom_ = self.create_publisher(Odometry, "desired", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Position of the system
        pos_0 = np.array([0.0, 0.0, 1.0], dtype=np.double)
        # Linear velocity of the sytem respect to the inertial frame
        vel_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Angular velocity respect to the Body frame
        omega_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Initial Orientation expressed as quaternionn
        quat_0 = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Initial Wrench
        Wrench0 = np.array([0, 0, self.mass*self.gravity, 0, 0, 0])

        # Auxiliary vector [x, v, q, w], which is used to update the odometry and the states of the system
        self.init = np.hstack((pos_0, vel_0, quat_0, omega_0))
        tension_matrix, P, tension_vector = self.jacobian_forces(Wrench0, self.init)
        print(np.linalg.norm(tension_matrix, axis=0))
        print(tension_matrix)
        print(Wrench0)

        self.x_0 = np.hstack((pos_0, vel_0, quat_0, omega_0))

        # MPC Parameters
        self.N = 10

        self.payloadModel()

        #self.timer = self.create_timer(self.ts, self.run)  # 0.01 seconds = 100 Hz
        #self.start_time = time.time()

    def quatTorot_c(self, quat):
        # Normalized quaternion
        q = quat

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        Q = ca.vertcat(
            ca.horzcat(q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)),
            ca.horzcat(2*(q1*q2+q0*q3), q0**2+q2**2-q1**2-q3**2, 2*(q2*q3-q0*q1)),
            ca.horzcat(2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2+q3**2-q1**2-q2**2))
        return Q

    def quatdot_c(self, quat, omega):
        # Quaternion evolution guaranteeing norm 1 (Improve this section)
        # INPUT
        # quat                                                   - actual quaternion
        # omega                                                  - angular velocities
        # OUTPUT
        # qdot                                                   - rate of change of the quaternion
        # Split values quaternion
        qw = quat[0, 0]
        qx = quat[1, 0]
        qy = quat[2, 0]
        qz = quat[3, 0]


        # Auxiliary variable in order to avoid numerical issues
        K_quat = 10
        quat_error = 1 - (qw**2 + qx**2 + qy**2 + qz**2)

        # Create skew matrix
        H_r_plus = ca.vertcat(ca.horzcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0]),
                                    ca.horzcat(quat[1, 0], quat[0, 0], -quat[3, 0], quat[2, 0]),
                                    ca.horzcat(quat[2, 0], quat[3, 0], quat[0, 0], -quat[1, 0]),
                                    ca.horzcat(quat[3, 0], -quat[2, 0], quat[1, 0], quat[0, 0]))

        omega_quat = ca.vertcat(0.0, omega[0, 0], omega[1, 0], omega[2, 0])


        q_dot = (1/2)*(H_r_plus@omega_quat) + K_quat*quat_error*quat
        return q_dot

    def payloadModel(self)->AcadosModel:
        # Model Name
        model_name = "payload"

        # Matrix for the inertia
        sparsity_I = ca.Sparsity.diag(3)
        I_sym = ca.MX(sparsity_I)
        # Assign the values from I_load
        I_sym[0, 0] = self.inertia[0, 0]
        I_sym[1, 1] = self.inertia[1, 1]
        I_sym[2, 2] = self.inertia[2, 2]

        #position 
        p_x = ca.MX.sym('p_x')
        p_y = ca.MX.sym('p_y')
        p_z = ca.MX.sym('p_z')
        x_p = ca.vertcat(p_x, p_y, p_z)
    
        #linear vel
        vx_p = ca.MX.sym("vx_p")
        vy_p = ca.MX.sym("vy_p")
        vz_p = ca.MX.sym("vz_p")   
        v_p = ca.vertcat(vx_p, vy_p, vz_p)

        #angles quaternion 
        qw = ca.MX.sym('qw')
        qx = ca.MX.sym('qx')
        qy = ca.MX.sym('qy')
        qz = ca.MX.sym('qz')        
        quat = ca.vertcat(qw, qx, qy, qz)
        
        #angular velocity
        wx = ca.MX.sym('wx')
        wy = ca.MX.sym('wy')
        wz = ca.MX.sym('wz')
        omega = ca.vertcat(wx, wy, wz) 

        # Cable kinematics
        nx_1 = ca.MX.sym('nx_1')
        ny_1 = ca.MX.sym('ny_1')
        nz_1 = ca.MX.sym('nz_1')
        n1 = ca.vertcat(nx_1, ny_1, nz_1)

        nx_2 = ca.MX.sym('nx_2')
        ny_2 = ca.MX.sym('ny_2')
        nz_2 = ca.MX.sym('nz_2')
        n2 = ca.vertcat(nx_2, ny_2, nz_2)

        nx_3 = ca.MX.sym('nx_3')
        ny_3 = ca.MX.sym('ny_3')
        nz_3 = ca.MX.sym('nz_3')
        n3 = ca.vertcat(nx_3, ny_3, nz_3)
        
        # Full states of the system
        x = ca.vertcat(x_p, v_p, quat, omega, n1, n2, n3)
        
        # Control actions of the system
        t_1_cmd = ca.MX.sym("t_1_cmd")
        rx_1_cmd = ca.MX.sym("rx_1_cmd")
        ry_1_cmd = ca.MX.sym("ry_1_cmd")
        rz_1_cmd = ca.MX.sym("rz_1_cmd")

        r1 = ca.vertcat(rx_1_cmd, ry_1_cmd, rz_1_cmd) 

        t_2_cmd = ca.MX.sym("t_2_cmd")
        rx_2_cmd = ca.MX.sym("rx_2_cmd")
        ry_2_cmd = ca.MX.sym("ry_2_cmd")
        rz_2_cmd = ca.MX.sym("rz_2_cmd")

        r2 = ca.vertcat(rx_2_cmd, ry_2_cmd, rz_2_cmd) 

        t_3_cmd = ca.MX.sym("t_3_cmd")
        rx_3_cmd = ca.MX.sym("rx_3_cmd")
        ry_3_cmd = ca.MX.sym("ry_3_cmd")
        rz_3_cmd = ca.MX.sym("rz_3_cmd")

        r3 = ca.vertcat(rx_3_cmd, ry_3_cmd, rz_3_cmd) 

        # Vector of control actions
        u = ca.vertcat(t_1_cmd, rx_1_cmd, ry_1_cmd, rz_1_cmd, t_2_cmd, rx_2_cmd, ry_2_cmd, rz_2_cmd, t_3_cmd, rx_3_cmd, ry_3_cmd, rz_3_cmd)

        # Rotation matrix
        Rot = self.quatTorot_c(quat)

        # Linear Dynamics
        linear_velocity = v_p
        linear_acceleration = -(1/self.mass)*t_1_cmd*n1 -(1/self.mass)*t_2_cmd*n2 -(1/self.mass)*t_3_cmd*n3 - self.gravity*self.e3

        # Angular dynamics
        quat_dt = self.quatdot_c(quat, omega)
        rho = ca.DM(self.p)
        cc_forces = ca.cross(omega, I_sym @ omega)
        tauB = t_1_cmd*ca.cross(Rot.T @ n1, rho[:,0]) \
                + t_2_cmd*ca.cross(Rot.T @ n2, rho[:,1]) \
                + t_3_cmd*ca.cross(Rot.T @ n3, rho[:,2])
        omega_dot = ca.solve(I_sym, -cc_forces + tauB)
        # Cable Kinematics
        n1_dot = ca.cross(r1, n1)
        n2_dot = ca.cross(r2, n2)
        n3_dot = ca.cross(r3, n3)

        # Explicit Dynamics
        f_expl = ca.vertcat(linear_velocity, linear_acceleration, quat_dt, omega_dot, n1_dot, n2_dot, n3_dot)

        #position desired
        p_x_d = ca.MX.sym('p_x_d')
        p_y_d = ca.MX.sym('p_y_d')
        p_z_d = ca.MX.sym('p_z_d')
        x_p_d = ca.vertcat(p_x_d, p_y_d, p_z_d)
    
        #linear vel
        vx_d = ca.MX.sym("vx_d")
        vy_d = ca.MX.sym("vy_d")
        vz_d = ca.MX.sym("vz_d")   
        v_d = ca.vertcat(vx_d, vy_d, vz_d)

        #angles quaternion 
        qw_d = ca.MX.sym('qw_d')
        qx_d = ca.MX.sym('qx_d')
        qy_d = ca.MX.sym('qy_d')
        qz_d = ca.MX.sym('qz_d')        
        quat_d = ca.vertcat(qw_d, qx_d, qy_d, qz_d)
        
        #angular velocity
        wx_d = ca.MX.sym('wx_d')
        wy_d = ca.MX.sym('wy_d')
        wz_d = ca.MX.sym('wz_d')
        omega_d = ca.vertcat(wx_d, wy_d, wz_d) 

        # Cable kinematics
        nx_1_d = ca.MX.sym('nx_1_d')
        ny_1_d = ca.MX.sym('ny_1_d')
        nz_1_d = ca.MX.sym('nz_1_d')
        n1_d = ca.vertcat(nx_1_d, ny_1_d, nz_1_d)

        nx_2_d = ca.MX.sym('nx_2_d')
        ny_2_d = ca.MX.sym('ny_2_d')
        nz_2_d = ca.MX.sym('nz_2_d')
        n2_d = ca.vertcat(nx_2_d, ny_2_d, nz_2_d)

        nx_3_d = ca.MX.sym('nx_3_d')
        ny_3_d = ca.MX.sym('ny_3_d')
        nz_3_d = ca.MX.sym('nz_3_d')
        n3_d = ca.vertcat(nx_3_d, ny_3_d, nz_3_d)
        
        # Full states desired
        x_d = ca.vertcat(x_p_d, v_d, quat_d, omega_d, n1_d, n2_d, n3_d)
        p = x_d

        # Dynamics
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.p = p
        model.name = model_name
        return model
        
    def hat(self, v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def jacobian_forces(self, wrench, payload):
        I = np.eye(3)
        top_block = np.hstack([I, I, I])  # shape: (3, 9)

        # Block 2: three rotation matrices
        q = np.array([payload[7], payload[8], payload[9], payload[6]])
        R_object = R.from_quat(q)
        R_ql = R_object.as_matrix()

        p1_hat = self.hat(self.p1)
        p2_hat = self.hat(self.p2)
        p3_hat = self.hat(self.p3)

        bottom_block = np.hstack([p1_hat@R_ql.T, p2_hat@R_ql.T, p3_hat@R_ql.T])  # shape: (3, 9)

        # Final 6x9 matrix
        P = np.vstack([top_block, bottom_block])

        tension = np.linalg.pinv(P)@wrench
        tensions_vectors = tension.reshape(-1, 3).T
        return tensions_vectors, P, tension

def main(arg = None):
    rclpy.init(args=arg)
    payload_node = PayloadControlNode()
    try:
        rclpy.spin(payload_node)  # Will run until manually interrupted
    except KeyboardInterrupt:
        payload_node.get_logger().info('Simulation stopped manually.')
        payload_node.destroy_node()
        rclpy.shutdown()
    finally:
        payload_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()