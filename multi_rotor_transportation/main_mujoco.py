#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import casadi as ca
from casadi import Function
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from scipy.spatial.transform import Rotation as R
import time
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class PayloadControlMujocoNode(Node):
    def __init__(self):
        super().__init__('PayloadControl')

        # Time Definition
        self.ts = 0.03
        self.final = 15
        self.t =np.arange(0, self.final + self.ts, self.ts, dtype=np.double)

        # Prediction Node of the NMPC formulation
        self.t_N = 0.5
        self.N = np.arange(0, self.t_N + self.ts, self.ts)
        self.N_prediction = self.N.shape[0]

        # Internal parameters defintion
        self.robot_num = 4
        self.mass = 1.5
        self.M_load = self.mass *  np.eye((3))
        self.inertia = np.array([[0.013344, 0.0, 0.0], [0.0, 0.012810, 0.0], [0.0, 0.0, 0.03064]], dtype=np.double)
        self.gravity = 9.81

        # Control gains
        c1 = 1
        kv_min = c1 + 1/4 + 0.1
        kp_min = (c1*(kv_min*kv_min) + 2*kv_min*c1 - c1*c1)/((self.mass)*(4*(kv_min - c1)-1))
        kp_min = 80
        self.kp_min = kp_min
        self.kv_min = 50
        self.c1 = c1
        print(self.kp_min)
        print(self.kv_min)
        print(c1)
        print("--------------------------")

        ## Compute minimiun values for the angular controller
        eigenvalues = np.linalg.eigvals(self.inertia)
        min_eigenvalue = np.min(eigenvalues)
        c2 = 0.2
        kw_min = (1/2)*c2 + (1/4) + 0.1
        kr_min = c2*(kw_min*kw_min)/(min_eigenvalue*(4*(kw_min - (1/2)*c2) - 1))

        self.kr_min = kr_min
        self.kw_min = 100
        self.c2 = c2
        print(self.kr_min)
        print(self.kw_min)
        print(c2)
        print("--------------------------")


        # Load shape parameters triangle
        self.p1 = np.array([-0.3, -0.2, 0.0], dtype=np.double)
        self.p2 = np.array([-0.3, 0.2, 0.0], dtype=np.double)
        self.p3 = np.array([0.3, -0.2, 0.0], dtype=np.double)
        self.p4 = np.array([0.3, 0.2, 0.0], dtype=np.double)

        self.p = np.vstack((self.p1, self.p2, self.p3, self.p4)).T
        self.length = 1.0 + 0.015
        self.e3 = ca.DM([0, 0, 1])

        # Position of the system we shoudl update this by the quadrotor initial positions
        pos_0 = np.array([0.6, 0.6, 0.87], dtype=np.double)
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

        # Init Tension of the cables so we can get initial cable direction
        self.tensions_init = np.linalg.norm(tension_matrix, axis=0)
        self.n_init = -tension_matrix/self.tensions_init
        self.n_init =  self.n_init.flatten(order='F')
        self.r_init = np.array([0.0, 0.0, 0.0]*self.robot_num, dtype=np.double)

        ## Init states
        self.x_0 = np.hstack((pos_0, vel_0, quat_0, omega_0, self.n_init))

        ## Init Control Actions or equilibirum
        self.r_dot_init = np.array([0.0, 0.0, 0.0]*self.robot_num, dtype=np.double)
        self.u_equilibrium = np.hstack((self.tensions_init, self.r_dot_init))

        ### Maximum and minimun control actions
        tension_min = 0.5*self.tensions_init
        tension_max = 4.5*self.tensions_init
        r_dot_max = np.array([2.0, 2.0, 2.0]*self.robot_num, dtype=np.double)
        r_dot_min = -r_dot_max
        self.u_min =  np.hstack((tension_min, r_dot_min))
        self.u_max =  np.hstack((tension_max, r_dot_max))

        ### Define state dimension and control action
        self.n_x = self.x_0.shape[0]
        self.n_u = self.u_equilibrium.shape[0]
        print(self.n_x)
        print(self.n_u)

        
        # Define odometry subscriber
        self.subscriber_payload_ = self.create_subscription(Odometry, "/payload/odom", self.callback_get_odometry_payload, 10)
        self.subscriber_drone_0_ = self.create_subscription(Odometry, "/drone_0/odom", self.callback_get_odometry_drone_0, 10)
        self.subscriber_drone_1_ = self.create_subscription(Odometry, "/drone_1/odom", self.callback_get_odometry_drone_1, 10)
        self.subscriber_drone_2_ = self.create_subscription(Odometry, "/drone_2/odom", self.callback_get_odometry_drone_2, 10)
        self.subscriber_drone_3_ = self.create_subscription(Odometry, "/drone_3/odom", self.callback_get_odometry_drone_3, 10)

        # Define PositionCmd publisher for each droe
        self.publisher_ref_drone_0 = self.create_publisher(PositionCommand, "/drone_0/position_cmd", 10)
        self.publisher_ref_drone_1 = self.create_publisher(PositionCommand, "/drone_1/position_cmd", 10)
        self.publisher_ref_drone_2 = self.create_publisher(PositionCommand, "/drone_2/position_cmd", 10)
        self.publisher_ref_drone_3 = self.create_publisher(PositionCommand, "/drone_3/position_cmd", 10)

        # TF
        self.tf_broadcaster = TransformBroadcaster(self)

         ### Create the initial states for quadrotor
        pos_quad_0 = np.array([0.06, 0.07, 1.8], dtype=np.double)
        ### Linear velocity of the sytem respect to the inertial frame
        vel_quad_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Angular velocity respect to the Body frame
        omega_quad_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Initial Orientation expressed as quaternionn
        quat_quad_0 = np.array([1.0, 0.0, 0.0, 0.0])

        pos_quad_1 = np.array([0.06, 1.12, 1.8], dtype=np.double)
        ### Linear velocity of the sytem respect to the inertial frame
        vel_quad_1 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Angular velocity respect to the Body frame
        omega_quad_1 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Initial Orientation expressed as quaternionn
        quat_quad_1 = np.array([1.0, 0.0, 0.0, 0.0])

        pos_quad_2 = np.array([1.13, 0.07, 1.8], dtype=np.double)
        ### Linear velocity of the sytem respect to the inertial frame
        vel_quad_2 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Angular velocity respect to the Body frame
        omega_quad_2 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Initial Orientation expressed as quaternionn
        quat_quad_2 = np.array([1.0, 0.0, 0.0, 0.0])

        pos_quad_3 = np.array([1.13, 1.12, 1.8], dtype=np.double)
        ### Linear velocity of the sytem respect to the inertial frame
        vel_quad_3 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Angular velocity respect to the Body frame
        omega_quad_3 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Initial Orientation expressed as quaternionn
        quat_quad_3 = np.array([1.0, 0.0, 0.0, 0.0])

        self.xq0_0 = np.hstack((pos_quad_0, vel_quad_0, quat_quad_0, omega_quad_0))
        self.xq1_0 = np.hstack((pos_quad_1, vel_quad_1, quat_quad_1, omega_quad_1))
        self.xq2_0 = np.hstack((pos_quad_2, vel_quad_2, quat_quad_2, omega_quad_2))
        self.xq3_0 = np.hstack((pos_quad_3, vel_quad_3, quat_quad_3, omega_quad_3))

        self.unit_vector_from_measurements = self.quadrotor_payload_unit_vector_c()

        self.timer = self.create_timer(self.ts, self.run)  # 0.01 seconds = 100 Hz
        self.start_time = time.time()

        #self.timer = self.create_timer(self.ts, self.validation)  # 0.01 seconds = 100 Hz
        #self.start_time = time.time()
    def callback_get_odometry_payload(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((13, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        vz_b = msg.twist.twist.linear.z

        vb = np.array([[vx_b], [vy_b], [vz_b]])
        
        # Get angular velocity body frame
        x[10] = msg.twist.twist.angular.x
        x[11] = msg.twist.twist.angular.y
        x[12] = msg.twist.twist.angular.z
        
        # Get quaternions
        x[7] = msg.pose.pose.orientation.x
        x[8] = msg.pose.pose.orientation.y
        x[9] = msg.pose.pose.orientation.z
        x[6] = msg.pose.pose.orientation.w

        # Rotation inertial frame
        rotational = R.from_quat([x[7], x[8], x[9], x[6]])
        rotational_matrix = rotational.as_matrix()
        vx_i = rotational_matrix@vb
    
        # Put values in the vector
        x[3] = vx_i[0, 0]
        x[4] = vx_i[1, 0]
        x[5] = vx_i[2, 0]

        xquadrotors = np.hstack((self.xq0_0[0:3], self.xq1_0[0:3], self.xq2_0[0:3], self.xq3_0[0:3])) 
        unit = np.array(self.unit_vector_from_measurements(x, xquadrotors)).reshape((self.robot_num*3, ))
        payload_states = np.hstack((x, unit))
        self.x_0 = payload_states
        return None

    def callback_get_odometry_drone_0(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((13, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        vz_b = msg.twist.twist.linear.z

        vb = np.array([[vx_b], [vy_b], [vz_b]])
        
        # Get angular velocity body frame
        x[10] = msg.twist.twist.angular.x
        x[11] = msg.twist.twist.angular.y
        x[12] = msg.twist.twist.angular.z
        
        # Get quaternions
        x[7] = msg.pose.pose.orientation.x
        x[8] = msg.pose.pose.orientation.y
        x[9] = msg.pose.pose.orientation.z
        x[6] = msg.pose.pose.orientation.w

        # Rotation inertial frame
        rotational = R.from_quat([x[7], x[8], x[9], x[6]])
        rotational_matrix = rotational.as_matrix()
        vx_i = rotational_matrix@vb
    
        # Put values in the vector
        x[3] = vx_i[0, 0]
        x[4] = vx_i[1, 0]
        x[5] = vx_i[2, 0]
        self.xq0_0 = x
        return None
        
    def callback_get_odometry_drone_1(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((13, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        vz_b = msg.twist.twist.linear.z

        vb = np.array([[vx_b], [vy_b], [vz_b]])
        
        # Get angular velocity body frame
        x[10] = msg.twist.twist.angular.x
        x[11] = msg.twist.twist.angular.y
        x[12] = msg.twist.twist.angular.z
        
        # Get quaternions
        x[7] = msg.pose.pose.orientation.x
        x[8] = msg.pose.pose.orientation.y
        x[9] = msg.pose.pose.orientation.z
        x[6] = msg.pose.pose.orientation.w

        # Rotation inertial frame
        rotational = R.from_quat([x[7], x[8], x[9], x[6]])
        rotational_matrix = rotational.as_matrix()
        vx_i = rotational_matrix@vb
    
        # Put values in the vector
        x[3] = vx_i[0, 0]
        x[4] = vx_i[1, 0]
        x[5] = vx_i[2, 0]
        self.xq1_0 = x
        return None

    def callback_get_odometry_drone_2(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((13, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        vz_b = msg.twist.twist.linear.z

        vb = np.array([[vx_b], [vy_b], [vz_b]])
        
        # Get angular velocity body frame
        x[10] = msg.twist.twist.angular.x
        x[11] = msg.twist.twist.angular.y
        x[12] = msg.twist.twist.angular.z
        
        # Get quaternions
        x[7] = msg.pose.pose.orientation.x
        x[8] = msg.pose.pose.orientation.y
        x[9] = msg.pose.pose.orientation.z
        x[6] = msg.pose.pose.orientation.w

        # Rotation inertial frame
        rotational = R.from_quat([x[7], x[8], x[9], x[6]])
        rotational_matrix = rotational.as_matrix()
        vx_i = rotational_matrix@vb
    
        # Put values in the vector
        x[3] = vx_i[0, 0]
        x[4] = vx_i[1, 0]
        x[5] = vx_i[2, 0]
        self.xq2_0 = x
        return None

    def callback_get_odometry_drone_3(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((13, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        vz_b = msg.twist.twist.linear.z

        vb = np.array([[vx_b], [vy_b], [vz_b]])
        
        # Get angular velocity body frame
        x[10] = msg.twist.twist.angular.x
        x[11] = msg.twist.twist.angular.y
        x[12] = msg.twist.twist.angular.z
        
        # Get quaternions
        x[7] = msg.pose.pose.orientation.x
        x[8] = msg.pose.pose.orientation.y
        x[9] = msg.pose.pose.orientation.z
        x[6] = msg.pose.pose.orientation.w

        # Rotation inertial frame
        rotational = R.from_quat([x[7], x[8], x[9], x[6]])
        rotational_matrix = rotational.as_matrix()
        vx_i = rotational_matrix@vb
    
        # Put values in the vector
        x[3] = vx_i[0, 0]
        x[4] = vx_i[1, 0]
        x[5] = vx_i[2, 0]
        self.xq3_0 = x
        return None
    
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

    def quatTorot(self, quat):
        # Normalized quaternion
        q0, q1, q2, q3 = quat
        Q = np.array([
            [q0**2 + q1**2 - q2**2 - q3**2,   2*(q1*q2 - q0*q3),       2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3),               q0**2 + q2**2 - q1**2 - q3**2,  2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2),               2*(q2*q3 + q0*q1),       q0**2 + q3**2 - q1**2 - q2**2]
            ])
        return Q

    def vee(self, X):
        x = np.zeros((3, 1), dtype=np.double)
        x[0, 0] = X[2, 1]
        x[1, 0] = X[0, 2]
        x[2, 0] = X[1, 0]
        return x

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

    def cost_quaternion_c(self):
        # Desired quaternion
        qd = ca.MX.sym('qd', 4, 1)

        # Current quaternion
        q = ca.MX.sym('q', 4, 1)

        # Compute conjugate of the quatenrion
        qd_conjugate = ca.vertcat(qd[0], -qd[1], -qd[2], -qd[3])

        # Quaternion multiplication q_e = qd_conjugate * q
        H_r_plus = ca.vertcat(
            ca.horzcat(qd_conjugate[0], -qd_conjugate[1], -qd_conjugate[2], -qd_conjugate[3]),
            ca.horzcat(qd_conjugate[1], qd_conjugate[0], qd_conjugate[3], -qd_conjugate[2]),
            ca.horzcat(qd_conjugate[2], -qd_conjugate[3], qd_conjugate[0], qd_conjugate[1]),
            ca.horzcat(qd_conjugate[3], qd_conjugate[2], -qd_conjugate[1], qd_conjugate[0])
        )

        q_error = H_r_plus @ q

        # Compute the angle and the log map
        norm_vec = ca.norm_2(q_error[1:4] + ca.np.finfo(np.float64).eps)
        angle = 2 * ca.atan2(norm_vec, q_error[0])

        # Avoid division by zero
        ln_quaternion = ca.vertcat(0.0, (1/2)*angle*q_error[1]/norm_vec, (1/2)*angle*q_error[2]/norm_vec, (1/2)*angle*q_error[3]/norm_vec)
        ln_quaternion_f = Function('ln_quaternion_f', [qd, q], [ln_quaternion])

        return  ln_quaternion_f

    def rotation_matrix_error_c(self):
        # Desired Quaternion
        qd = ca.MX.sym('qd', 4, 1)

        # Current quaternion
        q = ca.MX.sym('q', 4, 1)

        Rd = self.quatTorot_c(qd)
        R = self.quatTorot_c(q)

        error_matrix = R.T@Rd
        error_matrix_f = Function('error_matrix_f', [qd, q], [error_matrix])

        return error_matrix_f

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

        nx_4 = ca.MX.sym('nx_4')
        ny_4 = ca.MX.sym('ny_4')
        nz_4 = ca.MX.sym('nz_4')
        n4 = ca.vertcat(nx_4, ny_4, nz_4)

        # Cable kinematics
        rx_1 = ca.MX.sym('rx_1')
        ry_1 = ca.MX.sym('ry_1')
        rz_1 = ca.MX.sym('rz_1')
        r1 = ca.vertcat(rx_1, ry_1, rz_1)

        rx_2 = ca.MX.sym('rx_2')
        ry_2 = ca.MX.sym('ry_2')
        rz_2 = ca.MX.sym('rz_2')
        r2 = ca.vertcat(rx_2, ry_2, rz_2)

        rx_3 = ca.MX.sym('rx_3')
        ry_3 = ca.MX.sym('ry_3')
        rz_3 = ca.MX.sym('rz_3')
        r3 = ca.vertcat(rx_3, ry_3, rz_3)

        rx_4 = ca.MX.sym('rx_4')
        ry_4 = ca.MX.sym('ry_4')
        rz_4 = ca.MX.sym('rz_4')
        r4 = ca.vertcat(rx_4, ry_4, rz_4)
        
        # Full states of the system
        x = ca.vertcat(x_p, v_p, quat, omega, n1, n2, n3, n4, r1, r2, r3, r4)
        
        # Control actions of the system
        t_1_cmd = ca.MX.sym("t_1_cmd")
        rx_1_cmd = ca.MX.sym("rx_1_cmd")
        ry_1_cmd = ca.MX.sym("ry_1_cmd")
        rz_1_cmd = ca.MX.sym("rz_1_cmd")

        r1_cmd = ca.vertcat(rx_1_cmd, ry_1_cmd, rz_1_cmd) 

        t_2_cmd = ca.MX.sym("t_2_cmd")
        rx_2_cmd = ca.MX.sym("rx_2_cmd")
        ry_2_cmd = ca.MX.sym("ry_2_cmd")
        rz_2_cmd = ca.MX.sym("rz_2_cmd")

        r2_cmd = ca.vertcat(rx_2_cmd, ry_2_cmd, rz_2_cmd) 

        t_3_cmd = ca.MX.sym("t_3_cmd")
        rx_3_cmd = ca.MX.sym("rx_3_cmd")
        ry_3_cmd = ca.MX.sym("ry_3_cmd")
        rz_3_cmd = ca.MX.sym("rz_3_cmd")

        r3_cmd = ca.vertcat(rx_3_cmd, ry_3_cmd, rz_3_cmd) 

        t_4_cmd = ca.MX.sym("t_4_cmd")
        rx_4_cmd = ca.MX.sym("rx_4_cmd")
        ry_4_cmd = ca.MX.sym("ry_4_cmd")
        rz_4_cmd = ca.MX.sym("rz_4_cmd")

        r4_cmd = ca.vertcat(rx_4_cmd, ry_4_cmd, rz_4_cmd) 

        # Vector of control actions
        u = ca.vertcat(t_1_cmd, t_2_cmd, t_3_cmd, t_4_cmd, rx_1_cmd, ry_1_cmd, rz_1_cmd, rx_2_cmd, ry_2_cmd, rz_2_cmd, rx_3_cmd, ry_3_cmd, rz_3_cmd, rx_4_cmd, ry_4_cmd, rz_4_cmd)

        # Rotation matrix
        Rot = self.quatTorot_c(quat)

        # Linear Dynamics
        linear_velocity = v_p
        linear_acceleration = -(1/self.mass)*t_1_cmd*n1 -(1/self.mass)*t_2_cmd*n2 -(1/self.mass)*t_3_cmd*n3 -(1/self.mass)*t_4_cmd*n4 - self.gravity*self.e3

        # Angular dynamics
        quat_dt = self.quatdot_c(quat, omega)
        rho = ca.DM(self.p)
        cc_forces = ca.cross(omega, I_sym @ omega)
        tauB =    t_1_cmd*ca.cross(Rot.T @ n1, rho[:,0]) \
                + t_2_cmd*ca.cross(Rot.T @ n2, rho[:,1]) \
                + t_3_cmd*ca.cross(Rot.T @ n3, rho[:,2]) \
                + t_4_cmd*ca.cross(Rot.T @ n4, rho[:,3])
        omega_dot = ca.solve(I_sym, -cc_forces + tauB)
        # Cable Kinematics
        n1_dot = ca.cross(r1, n1)
        n2_dot = ca.cross(r2, n2)
        n3_dot = ca.cross(r3, n3)
        n4_dot = ca.cross(r4, n4)

        r1_dot = (r1_cmd)
        r2_dot = (r2_cmd)
        r3_dot = (r3_cmd)
        r4_dot = (r4_cmd)

        # Explicit Dynamics
        f_expl = ca.vertcat(linear_velocity, linear_acceleration, quat_dt, omega_dot, n1_dot, n2_dot, n3_dot, n4_dot, r1_dot, r2_dot, r3_dot, r4_dot)
        p = ca.MX.sym('p', x.shape[0] + u.shape[0], 1)

        # Dynamics
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.p = p
        model.name = model_name
        return model

    def solver(self, x0):
        # get dynamical model
        model = self.payloadModel()
        
        # Optimal control problem
        ocp = AcadosOcp()
        ocp.model = model

        # Get size of the system
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu

        # Set Dimension of the problem
        ocp.p = model.p
        ocp.dims.N = self.N_prediction

        # Definition of the cost functions (EXTERNAL)
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL" 

        # some variables
        x = ocp.model.x
        u = ocp.model.u
        p = ocp.model.p

        # Split states of the system
        x_p = x[0:3]
        v_p = x[3:6]
        quat = x[6:10]
        omega = x[10:13]
        n1 = x[13:16]
        n2 = x[16:19]
        n3 = x[19:22]
        n4 = x[22:25]

        r1 = x[25:28]
        r2 = x[28:31]
        r3 = x[31:34]
        r4 = x[34:37]
        r = x[25:37]

        # Split control actions
        t_cmd = u[0:4]
        r_dot_cmd = u[4:16]

        # Get desired states of the system
        x_p_d = p[0:3]
        v_p_d = p[3:6]
        quat_d = p[6:10]
        omega_d = p[10:13]

        n1_d = p[13:16]
        n2_d = p[16:19]
        n3_d = p[19:22]
        n4_d = p[22:25]

        r1_d = p[25:28]
        r2_d = p[28:31]
        r3_d = p[31:34]
        r4_d = p[34:37]
        r_d = p[25:37]

        t_d = p[37:41]
        r_dot_d = p[41:53]

        # Cost Functions of the system
        cost_quaternion_f = self.cost_quaternion_c()
        cost_matrix_error_f = self.rotation_matrix_error_c()
        #quadrotor_velocity_f = self.quadrotor_velocity_c()

        angular_error = cost_quaternion_f(quat_d, quat)
        angular_velocity_error = omega - cost_matrix_error_f(quat_d, quat)@omega_d

        error_position_quad = x_p - x_p_d
        error_velocity_quad = v_p - v_p_d

        # Cost functions
        lyapunov_position = (1/2)*self.kp_min*error_position_quad.T@error_position_quad + self.kv_min*(1/2)*(self.mass)*error_velocity_quad.T@error_velocity_quad + self.c1*error_position_quad.T@error_velocity_quad
        lyapunov_orientation = self.kr_min*angular_error.T@angular_error + self.kw_min*(1/2)*angular_velocity_error.T@self.inertia@angular_velocity_error

        # Error cable direction
        error_n1 = ca.cross(n1_d, n1)
        error_n2 = ca.cross(n2_d, n2)
        error_n3 = ca.cross(n3_d, n3)
        error_n4 = ca.cross(n4_d, n4)

        # Cost Function control actions
        tension_error = t_d - t_cmd
        r_error = r_d - r

        ocp.model.cost_expr_ext_cost = lyapunov_position + lyapunov_orientation  + error_n1.T@error_n1 + error_n2.T@error_n2 + error_n3.T@error_n3 + error_n4.T@error_n4 + 0.1*(r_error.T@r_error) + 1*(tension_error.T@tension_error) + 0.01*(r_dot_cmd.T@r_dot_cmd)
        ocp.model.cost_expr_ext_cost_e = lyapunov_position + lyapunov_orientation + error_n1.T@error_n1 + error_n2.T@error_n2 + error_n3.T@error_n3 + error_n4.T@error_n4 + 0.1*(r_error.T@r_error) 

        ref_params = np.hstack((self.x_0, self.u_equilibrium))

        ocp.parameter_values = ref_params

        ocp.constraints.constr_type = 'BGH'

        # Set constraints
        ocp.constraints.lbu = self.u_min
        ocp.constraints.ubu = self.u_max
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        ocp.constraints.x0 = x0

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" 
        ocp.solver_options.qp_solver_cond_N = self.N_prediction // 4
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.Tsim = self.ts
        ocp.solver_options.tf = self.t_N

        return ocp

    def hat(self, v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def jacobian_forces(self, wrench, payload):
        I = np.eye(3)
        top_block = np.hstack([I, I, I, I])  # shape: (3, 9)

        # Block 2: three rotation matrices
        q = np.array([payload[7], payload[8], payload[9], payload[6]])
        R_object = R.from_quat(q)
        R_ql = R_object.as_matrix()

        p1_hat = self.hat(self.p1)
        p2_hat = self.hat(self.p2)
        p3_hat = self.hat(self.p3)
        p4_hat = self.hat(self.p4)

        bottom_block = np.hstack([p1_hat@R_ql.T, p2_hat@R_ql.T, p3_hat@R_ql.T, p4_hat@R_ql.T])  # shape: (3, 9)

        # Final 6x9 matrix
        P = np.vstack([top_block, bottom_block])

        tension = np.linalg.pinv(P)@wrench
        tensions_vectors = tension.reshape(-1, 3).T
        return tensions_vectors, P, tension

    def ro_w(self, payload):
        # Rotation payload
        q = np.array([payload[7], payload[8], payload[9], payload[6]])
        R_object = R.from_quat(q)
        R_ql = R_object.as_matrix()

        # Translation
        t = payload[0:3]

        ro = np.zeros((3, self.p.shape[1]))

        for k in range(0, self.p.shape[1]):
            ro[:, k] = t + R_ql@self.p[:, k]
        return ro
    
    def quadrotor_payload_unit_vector_c(self):
        x = ca.MX.sym('x', 13, 1)

        x_p   = x[0:3]          # 3x1
        quat  = x[6:10]         # 4x1

        P = ca.DM(self.p) if isinstance(self.p, np.ndarray) else self.p  # 3 x m
        m = P.shape[1]
        L = self.length

        xq = ca.MX.sym('xq', 3*self.robot_num, 1)
        xq_p = ca.reshape(xq, 3, m)

        # rotation from quaternion
        Rot = self.quatTorot_c(quat)  # 3 x 3

        # Vectorized expression:
        cols = []
        for k in range(m):
            term = x_p + Rot@P[:, k] - xq_p[:, k]
            n_k      = L*(term/ca.norm_2(term))
            cols.append(n_k)
        quad_payload_mat = ca.hcat(cols)             # 3 x m
        quad_payload_vec = ca.reshape(quad_payload_mat, 3*m, 1)  # (3m) x 1
        quadrotor_payload_vector_f = ca.Function('quadrotor_payload_vector_f', [x, xq], [quad_payload_vec])
        return quadrotor_payload_vector_f

    def quadrotor_position_c(self):
        x = ca.MX.sym('x', self.n_x, 1)

        x_p   = x[0:3]          # 3x1
        quat  = x[6:10]         # 4x1
        nflat = x[13:25]        # (3*m)x1   (assumes your state packs 3*m entries here)

        if isinstance(self.p, np.ndarray):
            P = ca.DM(self.p)   # 3 x m
        else:
            P = self.p          # already a CasADi object with shape 3 x m

        m = P.shape[1]

        # reshape n into 3 x m 
        n = ca.reshape(nflat, 3, m)  # 3 x m

        # rotation from quaternion
        Rot = self.quatTorot_c(quat)  # 3 x 3

        # Vectorized expression:
        # each column: x_p + Rot * p[:,k] - length * n[:,k]
        quadrotor = ca.repmat(x_p, 1, m) + Rot @ P - (self.length * n)  # 3 x m

        quadrotor_vec = ca.reshape(quadrotor, 3*m, 1)

        quad_position_f = ca.Function('quad_position_f', [x], [quadrotor_vec])
        return quad_position_f

    def quadrotor_velocity_c(self):
        # --- symbols ---
        P = ca.DM(self.p) if isinstance(self.p, np.ndarray) else self.p  # 3 x m
        m = P.shape[1]
        L = self.length

        # state & input
        x = ca.MX.sym('x', self.n_x, 1)

        # unpack state
        x_p = x[0:3]
        v_p = x[3:6]
        quat = x[6:10]
        omega = x[10:13]
        nflat = x[13:25]
        n     = ca.reshape(nflat, 3, m)  # 3 x m
        rflat = x[25:37]
        r = ca.reshape(rflat, 3, m) # 3 x m (per-link angular velocities of n)

        # rotation
        R = self.quatTorot_c(quat)  # 3 x 3

        # build velocities per anchor (safe column-wise construction)
        cols = []
        for k in range(m):
            term_rot = R @ ca.cross(omega, P[:, k])        # R (omega x p_k)
            term_n   = L * ca.cross(r[:, k], n[:, k])      # L (r_k x n_k) = L n_dot_k
            v_k      = v_p + term_rot - term_n             # v_p + R(ω×p_k) - L(r_k×n_k)
            cols.append(v_k)
        quad_vel_mat = ca.hcat(cols)             # 3 x m
        quad_vel_vec = ca.reshape(quad_vel_mat, 3*m, 1)  # (3m) x 1
        quad_velocity_f = ca.Function('quad_velocity_f', [x], [quad_vel_vec])
        return quad_velocity_f

    def quadrotor_acceleration_c(self):
        # --- symbols ---
        P = ca.DM(self.p) if isinstance(self.p, np.ndarray) else self.p  # 3 x m
        m = P.shape[1]
        L = self.length
        sparsity_I = ca.Sparsity.diag(3)
        I_sym = ca.MX(sparsity_I)
        # Assign the values from I_load
        I_sym[0, 0] = self.inertia[0, 0]
        I_sym[1, 1] = self.inertia[1, 1]
        I_sym[2, 2] = self.inertia[2, 2]

        # state & input
        x = ca.MX.sym('x', self.n_x, 1)
        u = ca.MX.sym('u', self.n_u, 1)  # general: 3 thrust comps + 3m 'r' comps

        # unpack state
        x_p = x[0:3]
        v_p = x[3:6]
        quat = x[6:10]
        omega = x[10:13]
        nflat = x[13:25]
        n     = ca.reshape(nflat, 3, m)  # 3 x m
        rflat = x[25:37]
        r = ca.reshape(rflat, 3, m) # 3 x m (per-link angular velocities of n)

        # unpack Control Action
        t_1_cmd = u[0]                  # optional thrust vector (unused here)
        t_2_cmd = u[1]                  # optional thrust vector (unused here)
        t_3_cmd = u[2]                  # optional thrust vector (unused here)
        t_4_cmd = u[3]                  # optional thrust vector (unused here)
        r_dot = ca.reshape(u[4:], 3, m) # 3 x m (per-link angular velocities of n)

        # rotation
        Rot = self.quatTorot_c(quat)  # 3 x 3

        # Linear Acceleration Payload
        linear_acceleration = -(1/self.mass)*t_1_cmd*n[:, 0] -(1/self.mass)*t_2_cmd*n[:, 1] -(1/self.mass)*t_3_cmd*n[:, 2] -(1/self.mass)*t_4_cmd*n[:, 3] - self.gravity*self.e3

        # Angular Acceleration payload
        cc_forces = ca.cross(omega, I_sym @ omega)
        tauB = t_1_cmd*ca.cross(Rot.T @ n[:, 0], P[:,0]) \
                + t_2_cmd*ca.cross(Rot.T @ n[:, 1], P[:,1]) \
                + t_3_cmd*ca.cross(Rot.T @ n[:, 2], P[:,2]) \
                + t_4_cmd*ca.cross(Rot.T @ n[:, 3], P[:,3])
        omega_dot = ca.solve(I_sym, -cc_forces + tauB)

        # build velocities per anchor (safe column-wise construction)
        cols = []
        for k in range(m):
            term_acceleration = linear_acceleration + Rot@(ca.cross(omega_dot, P[:, k]))
            aux_cross_payload_angular = ca.cross(omega, P[:, k])
            payload_angular = Rot@(ca.cross(omega, aux_cross_payload_angular))
            input_angular_acc_cable = - L*(ca.cross(r_dot[:, k], n[:, k]))
            angular_velocity_cable_aux = ca.cross(r[:, k], n[:, k])
            angular_velocity_cable = -L*ca.cross(r[:, k], angular_velocity_cable_aux)
            v_k      = term_acceleration + payload_angular + input_angular_acc_cable + angular_velocity_cable
            cols.append(v_k)
        quad_acc_mat = ca.hcat(cols)             # 3 x m
        quad_acc_vec = ca.reshape(quad_acc_mat, 3*m, 1)  # (3m) x 1
        quad_acc_f = ca.Function('quad_acc_f', [x, u], [quad_acc_vec])
        return quad_acc_f

    def cable_angular_velocity_c(self):
        # --- symbols ---
        P = ca.DM(self.p) if isinstance(self.p, np.ndarray) else self.p  # 3 x m
        m = P.shape[1]
        L = self.length

        # state & input
        x = ca.MX.sym('x', 22, 1)
        xQ = ca.MX.sym('xQ', 3*m, 1)  # general: 3 thrust comps + 3m 'r' comps

        # unpack state
        x_p   = x[0:3]      # 3x1
        v_p   = x[3:6]      # 3x1
        quat  = x[6:10]     # 4x1
        omega = x[10:13]    # 3x1
        nflat = x[13:13+3*m]
        n     = ca.reshape(nflat, 3, m)  # 3 x m

        # unpack Quadrotor velocity
        v_Q = ca.reshape(xQ, 3, m) # 3 x m (per-link angular velocities of n)


        # rotation
        R = self.quatTorot_c(quat)  # 3 x 3

        # build velocities per anchor (safe column-wise construction)
        cols = []
        for k in range(m):
            term_rot = R @ ca.cross(omega, P[:, k])        # R (omega x p_k)
            term_quad_velocity   = v_Q[:, k]
            n_dot_k      = (v_p + term_rot - term_quad_velocity)/L
            r_k = ca.cross(n[:, k], n_dot_k)
            cols.append(r_k)
        r_mat = ca.hcat(cols)             # 3 x m
        r_vec = ca.reshape(r_mat, 3*m, 1)  # (3m) x 1
        r_velocity_f = ca.Function('r_velocity_f', [x, xQ], [r_vec])
        return r_velocity_f

    def quadrotors_w(self, payload):
        # Rotation payload
        q = np.array([payload[7], payload[8], payload[9], payload[6]])
        R_object = R.from_quat(q)
        R_ql = R_object.as_matrix()

        # Translation
        t = payload[0:3]

        n = payload[13:25]

        n = n.reshape((3, self.p.shape[1]), order='F')

        quat = np.zeros((3, self.p.shape[1]))

        for k in range(0, self.p.shape[1]):
            quat[:, k] = t + R_ql@self.p[:, k] - 1*(n[:, k])*self.length
        return quat     

    def jacobian_forces(self, wrench, payload):
        I = np.eye(3)
        top_block = np.hstack([I, I, I, I])  # shape: (3, 9)

        # Block 2: three rotation matrices
        q = np.array([payload[7], payload[8], payload[9], payload[6]])
        R_object = R.from_quat(q)
        R_ql = R_object.as_matrix()

        p1_hat = self.hat(self.p1)
        p2_hat = self.hat(self.p2)
        p3_hat = self.hat(self.p3)
        p4_hat = self.hat(self.p4)

        bottom_block = np.hstack([p1_hat@R_ql.T, p2_hat@R_ql.T, p3_hat@R_ql.T, p4_hat@R_ql.T])  # shape: (3, 9)

        # Final 6x9 matrix
        P = np.vstack([top_block, bottom_block])

        tension = np.linalg.pinv(P)@wrench
        tensions_vectors = tension.reshape(-1, 3).T
        return tensions_vectors, P, tension

    def send_position_cmd(self, publisher, x, v, a):
        position_cmd_msg = PositionCommand()
        position_cmd_msg.position.x = x[0]
        position_cmd_msg.position.y = x[1]
        position_cmd_msg.position.z = x[2]

        position_cmd_msg.velocity.x = v[0]
        position_cmd_msg.velocity.y = v[1]
        position_cmd_msg.velocity.z = v[2]
        
        position_cmd_msg.acceleration.x = a[0]
        position_cmd_msg.acceleration.y = a[1]
        position_cmd_msg.acceleration.z = a[2]
        publisher.publish(position_cmd_msg)
        return None 

    def publish_transforms(self):
        # Payload
## -------------------------------------------------------------------------------------------------------------------
        tf_world_load = TransformStamped()
        tf_world_load.header.stamp = self.get_clock().now().to_msg()
        tf_world_load.header.frame_id = 'world'            # <-- world is the parent
        tf_world_load.child_frame_id = 'payload'          # <-- imu_link is rotated
        tf_world_load.transform.translation.x = self.x_0[0]
        tf_world_load.transform.translation.y = self.x_0[1]
        tf_world_load.transform.translation.z = self.x_0[2]
        tf_world_load.transform.rotation.x = self.x_0[7]
        tf_world_load.transform.rotation.y = self.x_0[8]
        tf_world_load.transform.rotation.z = self.x_0[9]
        tf_world_load.transform.rotation.w = self.x_0[6]

        # Quadrotor
        tf_world_quad1 = TransformStamped()
        tf_world_quad1.header.stamp = self.get_clock().now().to_msg()
        tf_world_quad1.header.frame_id = 'world'            # <-- world is the parent
        tf_world_quad1.child_frame_id = 'drone_0'          # <-- imu_link is rotated
        tf_world_quad1.transform.translation.x = self.xq0_0[0]
        tf_world_quad1.transform.translation.y = self.xq0_0[1]
        tf_world_quad1.transform.translation.z = self.xq0_0[2]
        tf_world_quad1.transform.rotation.x = self.xq0_0[7]
        tf_world_quad1.transform.rotation.y = self.xq0_0[8]
        tf_world_quad1.transform.rotation.z = self.xq0_0[9]
        tf_world_quad1.transform.rotation.w = self.xq0_0[6]

        tf_world_quad2 = TransformStamped()
        tf_world_quad2.header.stamp = self.get_clock().now().to_msg()
        tf_world_quad2.header.frame_id = 'world'            # <-- world is the parent
        tf_world_quad2.child_frame_id = 'drone_1'          # <-- imu_link is rotated
        tf_world_quad2.transform.translation.x = self.xq1_0[0]
        tf_world_quad2.transform.translation.y = self.xq1_0[1]
        tf_world_quad2.transform.translation.z = self.xq1_0[2]
        tf_world_quad2.transform.rotation.x = self.xq1_0[7]
        tf_world_quad2.transform.rotation.y = self.xq1_0[8]
        tf_world_quad2.transform.rotation.z = self.xq1_0[9]
        tf_world_quad2.transform.rotation.w = self.xq1_0[6]

        tf_world_quad3 = TransformStamped()
        tf_world_quad3.header.stamp = self.get_clock().now().to_msg()
        tf_world_quad3.header.frame_id = 'world'            # <-- world is the parent
        tf_world_quad3.child_frame_id = 'drone_2'          # <-- imu_link is rotated
        tf_world_quad3.transform.translation.x = self.xq2_0[0]
        tf_world_quad3.transform.translation.y = self.xq2_0[1]
        tf_world_quad3.transform.translation.z = self.xq2_0[2]
        tf_world_quad3.transform.rotation.x = self.xq2_0[7]
        tf_world_quad3.transform.rotation.y = self.xq2_0[8]
        tf_world_quad3.transform.rotation.z = self.xq2_0[9]
        tf_world_quad3.transform.rotation.w = self.xq2_0[6]
        
        tf_world_quad4 = TransformStamped()
        tf_world_quad4.header.stamp = self.get_clock().now().to_msg()
        tf_world_quad4.header.frame_id = 'world'            # <-- world is the parent
        tf_world_quad4.child_frame_id = 'drone_3'          # <-- imu_link is rotated
        tf_world_quad4.transform.translation.x = self.xq3_0[0]
        tf_world_quad4.transform.translation.y = self.xq3_0[1]
        tf_world_quad4.transform.translation.z = self.xq3_0[2]
        tf_world_quad4.transform.rotation.x = self.xq3_0[7]
        tf_world_quad4.transform.rotation.y = self.xq3_0[8]
        tf_world_quad4.transform.rotation.z = self.xq3_0[9]
        tf_world_quad4.transform.rotation.w = self.xq3_0[6]

        
        ro = self.ro_w(self.x_0)
        tf_payload_p1 = TransformStamped()
        tf_payload_p1.header.stamp = self.get_clock().now().to_msg()
        tf_payload_p1.header.frame_id = 'payload'
        tf_payload_p1.child_frame_id = 'p_0'
        tf_payload_p1.transform.translation.x = self.p1[0]
        tf_payload_p1.transform.translation.y = self.p1[1]
        tf_payload_p1.transform.translation.z = self.p1[2]

        tf_payload_p2 = TransformStamped()
        tf_payload_p2.header.stamp = self.get_clock().now().to_msg()
        tf_payload_p2.header.frame_id = 'payload'
        tf_payload_p2.child_frame_id = 'p_1'
        tf_payload_p2.transform.translation.x = self.p2[0]
        tf_payload_p2.transform.translation.y = self.p2[1]
        tf_payload_p2.transform.translation.z = self.p2[2]

        tf_payload_p3 = TransformStamped()
        tf_payload_p3.header.stamp = self.get_clock().now().to_msg()
        tf_payload_p3.header.frame_id = 'payload'
        tf_payload_p3.child_frame_id = 'p_2'
        tf_payload_p3.transform.translation.x = self.p3[0]
        tf_payload_p3.transform.translation.y = self.p3[1]
        tf_payload_p3.transform.translation.z = self.p3[2]

        tf_payload_p4 = TransformStamped()
        tf_payload_p4.header.stamp = self.get_clock().now().to_msg()
        tf_payload_p4.header.frame_id = 'payload'
        tf_payload_p4.child_frame_id = 'p_3'
        tf_payload_p4.transform.translation.x = self.p4[0]
        tf_payload_p4.transform.translation.y = self.p4[1]
        tf_payload_p4.transform.translation.z = self.p4[2]

        ## ------------------------------------------------------------- AUX TF------------------------------- 
        tf_world_p1 = TransformStamped()
        tf_world_p1.header.stamp = self.get_clock().now().to_msg()
        tf_world_p1.header.frame_id = 'world'
        tf_world_p1.child_frame_id = 'p_0_aux'
        tf_world_p1.transform.translation.x = ro[0, 0]
        tf_world_p1.transform.translation.y = ro[1, 0]
        tf_world_p1.transform.translation.z = ro[2, 0]

        tf_world_p2 = TransformStamped()
        tf_world_p2.header.stamp = self.get_clock().now().to_msg()
        tf_world_p2.header.frame_id = 'world'
        tf_world_p2.child_frame_id = 'p_1_aux'
        tf_world_p2.transform.translation.x = ro[0, 1]
        tf_world_p2.transform.translation.y = ro[1, 1]
        tf_world_p2.transform.translation.z = ro[2, 1]

        tf_world_p3 = TransformStamped()
        tf_world_p3.header.stamp = self.get_clock().now().to_msg()
        tf_world_p3.header.frame_id = 'world'
        tf_world_p3.child_frame_id = 'p_2_aux'
        tf_world_p3.transform.translation.x = ro[0, 2]
        tf_world_p3.transform.translation.y = ro[1, 2]
        tf_world_p3.transform.translation.z = ro[2, 2]

        tf_world_p4 = TransformStamped()
        tf_world_p4.header.stamp = self.get_clock().now().to_msg()
        tf_world_p4.header.frame_id = 'world'
        tf_world_p4.child_frame_id = 'p_3_aux'
        tf_world_p4.transform.translation.x = ro[0, 3]
        tf_world_p4.transform.translation.y = ro[1, 3]
        tf_world_p4.transform.translation.z = ro[2, 3]

        
        # Extract Direction Only to check Values
        n = self.x_0[13:25]
        tension = n.reshape((3, self.p.shape[1]), order='F')

        tf_p1_q1 = TransformStamped()
        tf_p1_q1.header.stamp = self.get_clock().now().to_msg()
        tf_p1_q1.header.frame_id = 'p_0_aux'
        tf_p1_q1.child_frame_id = 'quadrotor_0'
        data_p1_q1 = -(tension[:, 0])*self.length
        tf_p1_q1.transform.translation.x = data_p1_q1[0]
        tf_p1_q1.transform.translation.y = data_p1_q1[1]
        tf_p1_q1.transform.translation.z = data_p1_q1[2]

        tf_p2_q2 = TransformStamped()
        tf_p2_q2.header.stamp = self.get_clock().now().to_msg()
        tf_p2_q2.header.frame_id = 'p_1_aux'
        tf_p2_q2.child_frame_id = 'quadrotor_1'
        data_p2_q2 = -(tension[:, 1])*self.length
        tf_p2_q2.transform.translation.x = data_p2_q2[0]
        tf_p2_q2.transform.translation.y = data_p2_q2[1]
        tf_p2_q2.transform.translation.z = data_p2_q2[2]

        tf_p3_q3 = TransformStamped()
        tf_p3_q3.header.stamp = self.get_clock().now().to_msg()
        tf_p3_q3.header.frame_id = 'p_2_aux'
        tf_p3_q3.child_frame_id = 'quadrotor_2'
        data_p3_q3 = -(tension[:, 2])*self.length
        tf_p3_q3.transform.translation.x = data_p3_q3[0]
        tf_p3_q3.transform.translation.y = data_p3_q3[1]
        tf_p3_q3.transform.translation.z = data_p3_q3[2]

        tf_p4_q4 = TransformStamped()
        tf_p4_q4.header.stamp = self.get_clock().now().to_msg()
        tf_p4_q4.header.frame_id = 'p_3_aux'
        tf_p4_q4.child_frame_id = 'quadrotor_3'
        data_p4_q4 = -(tension[:, 3])*self.length
        tf_p4_q4.transform.translation.x = data_p4_q4[0]
        tf_p4_q4.transform.translation.y = data_p4_q4[1]
        tf_p4_q4.transform.translation.z = data_p4_q4[2]

        self.tf_broadcaster.sendTransform([tf_world_load, tf_world_quad1, tf_world_quad2, tf_world_quad3, tf_world_quad4, tf_payload_p1, tf_payload_p2, tf_payload_p3, tf_payload_p4, tf_world_p1, tf_world_p2, tf_world_p3, tf_world_p4, tf_p1_q1, tf_p2_q2, tf_p3_q3, tf_p4_q4])
        return None
    def run(self):

        xd_q0 = np.array([0.0, -2.5, 2.0], dtype=np.double)
        vd_q0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ad_q0 = np.array([0.0, 0.0, 0.0], dtype=np.double)

        xd_q1 = np.array([0.0, -1.5, 2.0], dtype=np.double)
        vd_q1 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ad_q1 = np.array([0.0, 0.0, 0.0], dtype=np.double)

        xd_q2 = np.array([1.0, -2.5, 2.0], dtype=np.double)
        vd_q2 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ad_q2 = np.array([0.0, 0.0, 0.0], dtype=np.double)

        xd_q3 = np.array([1.13, -1.5, 2.0], dtype=np.double)
        vd_q3 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ad_q3 = np.array([0.0, 0.0, 0.0], dtype=np.double)

        self.send_position_cmd(self.publisher_ref_drone_0, xd_q0, vd_q0, ad_q0)
        self.send_position_cmd(self.publisher_ref_drone_1, xd_q1, vd_q1, ad_q1)
        self.send_position_cmd(self.publisher_ref_drone_2, xd_q2, vd_q2, ad_q2)
        self.send_position_cmd(self.publisher_ref_drone_3, xd_q3, vd_q3, ad_q3)
        self.publish_transforms()

def main(arg = None):
    rclpy.init(args=arg)
    payload_node = PayloadControlMujocoNode()
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