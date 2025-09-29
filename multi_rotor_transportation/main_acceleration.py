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
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from multi_rotor_transportation import plot_tensions, plot_angular_velocities, plot_quad_position, plot_quad_velocity, plot_quad_position_desired, plot_quad_velocity_desired
from multi_rotor_transportation import plot_angular_velocities_aux, plot_angular_cable_desired
class PayloadControlNode(Node):
    def __init__(self):
        super().__init__('PayloadDynamics')

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
        self.mass = 1.0
        self.M_load = self.mass *  np.eye((3))
        self.inertia = np.array([[0.013344, 0.0, 0.0], [0.0, 0.012810, 0.0], [0.0, 0.0, 0.03064]], dtype=np.double)
        self.gravity = 9.81

        # Quadrotor paramaters
        self.mass_quad = 1.0
        self.inertia_quad = np.array([[0.00305587, 0.0, 0.0], [0.0, 0.00159695, 0.0], [0.0, 0.0, 0.00159687]], dtype=np.double)

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
        self.p1 = np.array([0.35, 0.35, 0.0], dtype=np.double)
        self.p2 = np.array([-0.35, 0.35, 0.0], dtype=np.double)
        self.p3 = np.array([-0.35, -0.35, 0.0], dtype=np.double)
        self.p4 = np.array([0.35, -0.35, 0.0], dtype=np.double)

        self.p = np.vstack((self.p1, self.p2, self.p3, self.p4)).T
        self.length = 1.5
        self.e3 = ca.DM([0, 0, 1])

        # Payload odometry
        self.odom_payload_msg = Odometry()
        self.publisher_payload_odom_ = self.create_publisher(Odometry, "odom", 10)

        self.odom_payload_desired_msg = Odometry()
        self.publisher_payload_desired_odom_ = self.create_publisher(Odometry, "desired", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        # Position of the system
        pos_0 = np.array([0.0, 0.0, 0.5], dtype=np.double)
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
        self.x_0 = np.hstack((pos_0, vel_0, quat_0, omega_0, self.n_init, self.r_init))


        ## Init Control Actions or equilibirum
        self.r_dot_init = np.array([0.0, 0.0, 0.0]*self.robot_num, dtype=np.double)
        self.u_equilibrium = np.hstack((self.tensions_init, self.r_dot_init))

        ### Maximum and minimun control actions
        tension_min = 0.5*self.tensions_init
        tension_max = 2.5*self.tensions_init
        r_dot_max = np.array([2.0, 2.0, 2.0]*self.robot_num, dtype=np.double)
        r_dot_min = -r_dot_max
        self.u_min =  np.hstack((tension_min, r_dot_min))
        self.u_max =  np.hstack((tension_max, r_dot_max))

        ### Define state dimension and control action
        self.n_x = self.x_0.shape[0]
        self.n_u = self.u_equilibrium.shape[0]
        print(self.n_x)
        print(self.n_u)

        #model = self.payloadModel()
        ### Create Model and OCP of the sytem        
        self.ocp = self.solver(self.x_0)

        ### Createn Model of the quadrotor
        quadrotor_position = self.quadrotor_position_c()
        quad_total_positions = np.array(quadrotor_position(self.x_0)).reshape((self.robot_num*3, ))

        self.quad_1_model, self.nx_quad, self.nu_quad = self.quadrotorModel(1)
        self.quad_2_model, _, _ = self.quadrotorModel(2)
        self.quad_3_model, _, _ = self.quadrotorModel(3)
        self.quad_4_model, _, _ = self.quadrotorModel(4)

        ### Create the initial states for quadrotor
        pos_quad_1 = np.array(quad_total_positions[0:3], dtype=np.double)
        ### Linear velocity of the sytem respect to the inertial frame
        vel_quad_1 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Angular velocity respect to the Body frame
        omega_quad_1 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Initial Orientation expressed as quaternionn
        quat_quad_1 = np.array([1.0, 0.0, 0.0, 0.0])

        pos_quad_2 = np.array(quad_total_positions[3:6], dtype=np.double)
        ### Linear velocity of the sytem respect to the inertial frame
        vel_quad_2 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Angular velocity respect to the Body frame
        omega_quad_2 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Initial Orientation expressed as quaternionn
        quat_quad_2 = np.array([1.0, 0.0, 0.0, 0.0])

        pos_quad_3 = np.array(quad_total_positions[6:9], dtype=np.double)
        ### Linear velocity of the sytem respect to the inertial frame
        vel_quad_3 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Angular velocity respect to the Body frame
        omega_quad_3 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Initial Orientation expressed as quaternionn
        quat_quad_3 = np.array([1.0, 0.0, 0.0, 0.0])

        pos_quad_4 = np.array(quad_total_positions[9:12], dtype=np.double)
        ### Linear velocity of the sytem respect to the inertial frame
        vel_quad_4 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Angular velocity respect to the Body frame
        omega_quad_4 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ### Initial Orientation expressed as quaternionn
        quat_quad_4 = np.array([1.0, 0.0, 0.0, 0.0])

        self.xq1_0 = np.hstack((pos_quad_1, vel_quad_1, quat_quad_1, omega_quad_1))
        self.xq2_0 = np.hstack((pos_quad_2, vel_quad_2, quat_quad_2, omega_quad_2))
        self.xq3_0 = np.hstack((pos_quad_3, vel_quad_3, quat_quad_3, omega_quad_3))
        self.xq4_0 = np.hstack((pos_quad_4, vel_quad_4, quat_quad_4, omega_quad_4))

        ### Control Gains quadrotor
        self.K_p = np.array([[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]], dtype=np.double)
        self.K_v =  np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]], dtype=np.double)
        self.K_omega = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]], dtype=np.double)
        self.K_orientation = np.array([[450.0, 0.0, 0.0], [0.0, 450.0, 0.0], [0.0, 0.0, 40.0]], dtype=np.double)

        ### OCP
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp_" + self.ocp.model.name + ".json", build= True, generate= True)

        #### Integration using Acados
        self.acados_integrator = AcadosSimSolver(self.ocp, json_file="acados_sim_" + self.ocp.model.name + ".json", build= True, generate= True)

        self.timer = self.create_timer(self.ts, self.run)  # 0.01 seconds = 100 Hz
        self.start_time = time.time()

        #self.timer = self.create_timer(self.ts, self.validation)  # 0.01 seconds = 100 Hz
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

    def quadrotorModel(self, number)->AcadosModel:
        # Model Name
        model_name = f"quadrotor_{number}"

        # Sample time definition
        ts = ca.MX.sym('ts')

        # Matrix for the inertia
        sparsity_I = ca.Sparsity.diag(3)
        I_sym = ca.MX(sparsity_I)
        # Assign the values from I_load
        I_sym[0, 0] = self.inertia_quad[0, 0]
        I_sym[1, 1] = self.inertia_quad[1, 1]
        I_sym[2, 2] = self.inertia_quad[2, 2]

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

        # Full states of the system
        x = ca.vertcat(x_p, v_p, quat, omega)
        
        f_cmd = ca.MX.sym("f_cmd")
        tx_cmd = ca.MX.sym("tx_cmd")
        ty_cmd = ca.MX.sym("ty_cmd")
        tz_cmd = ca.MX.sym("tz_cmd")
        tau = ca.vertcat(tx_cmd, ty_cmd, tz_cmd) 

        t = ca.MX.sym("t")
        n_x = ca.MX.sym("n_x")
        n_y = ca.MX.sym("n_y")
        n_z = ca.MX.sym("n_z")
        n = ca.vertcat(n_x, n_y, n_z)
        u_ext = ca.vertcat(t, n_x, n_y, n_z)

        # Vector of control actions
        u = ca.vertcat(f_cmd, tx_cmd, ty_cmd, tz_cmd)

        # Rotation matrix
        Rot = self.quatTorot_c(quat)

        # Linear Dynamics
        linear_velocity = v_p
        linear_acceleration = (f_cmd/self.mass_quad)*Rot@self.e3 - self.gravity*self.e3 + t*n

        # Angular dynamics
        quat_dt = self.quatdot_c(quat, omega)
        cc_forces = ca.cross(omega, I_sym @ omega)
        omega_dot = ca.solve(I_sym, -cc_forces + tau)

        # Size of the system
        nx = x.shape[0]
        nu = u.shape[0]

        # Explicit Dynamics
        f_expl = ca.vertcat(linear_velocity, linear_acceleration, quat_dt, omega_dot)

        dynamics_casadi_f = Function(f"quadrotor_dynamics_{number}",[x, u, u_ext], [f_expl])

        ## Integration method
        k1 = dynamics_casadi_f(x, u, u_ext)
        k2 = dynamics_casadi_f(x + (1/2)*ts*k1, u, u_ext)
        k3 = dynamics_casadi_f(x + (1/2)*ts*k2, u, u_ext)
        k4 = dynamics_casadi_f(x + ts*k3, u, u_ext)

        # Compute forward Euler method
        xk = x + (1/6)*self.ts*(k1 + 2*k2 + 2*k3 + k4)
        casadi_kutta = Function(f"quadrotor_dynamics_kuta_{number}",[x, u, u_ext, ts], [xk]) 

        # Create Runge Kutta Integrator
        return casadi_kutta, nx, nu

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

        ocp.model.cost_expr_ext_cost = lyapunov_position + lyapunov_orientation  + error_n1.T@error_n1 + error_n2.T@error_n2 + error_n3.T@error_n3 + error_n4.T@error_n4 + 1*(r_error.T@r_error) + 10*(tension_error.T@tension_error) + 2*(r_dot_cmd.T@r_dot_cmd)
        ocp.model.cost_expr_ext_cost_e = lyapunov_position + lyapunov_orientation + error_n1.T@error_n1 + error_n2.T@error_n2 + error_n3.T@error_n3 + error_n4.T@error_n4 + 1*(r_error.T@r_error) 

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
    def geometric_control(self, x, xd, v, vd, ad, quat, psi, omega, omega_d, t, n):
        kp = self.K_p
        kv = self.K_v

        # Control Error
        aux_variable = kp@(xd - x) + kv@(vd - v) + ad 

        force_zb = self.mass_quad * (aux_variable + self.gravity*np.array([0.0, 0.0, 1.0]) - t*n)

                # Rotation matrix of the body        
        R_b = self.quatTorot(quat)
        Zb = R_b[:, 2]

        # Force in the body frame
        force = np.dot(Zb, force_zb)

        Zb_d = force_zb/(np.linalg.norm(force_zb))
        Xc_d = np.array([ np.cos(psi), np.sin(psi), 0])
        Yb_d = (np.cross(Zb_d, Xc_d))/(np.linalg.norm(np.cross(Zb_d, Xc_d)))
        Xb_d = np.cross(Yb_d, Zb_d)

        ## Desired Rotation Matrix
        R_d = np.array([[Xb_d[0], Yb_d[0], Zb_d[0]], [Xb_d[1], Yb_d[1], Zb_d[1]], [Xb_d[2], Yb_d[2], Zb_d[2]]])

        ## Compute orientation error
        er = (1/2)*self.vee(R_d.T@R_b - R_b.T@R_d)

        ## Coriolis
        c = np.cross(omega, self.inertia_quad@omega)
        c = c.reshape((3, 1))

        ## Angular velocity error
        omega_d = omega_d.reshape((3, 1))
        omega = omega.reshape((3, 1))

        e_omega = omega - R_b.T@R_d@omega_d 

        ## Nonlinear Dynamics Inversion 
        M = c -self.inertia_quad@self.K_orientation@er - self.inertia_quad@self.K_omega@e_omega
        control = np.array([force, M[0,0], M[1, 0], M[2, 0]])
        return control

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
        x = ca.MX.sym('x', self.n_x, 1)

        x_p   = x[0:3]          # 3x1
        quat  = x[6:10]         # 4x1
        nflat = x[13:25]        # (3*m)x1   (assumes your state packs 3*m entries here)

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

    def send_odometry(self, x, odom_payload_msg, publisher_payload_odom):
        position = x[0:3]
        quat = x[6:10]

        # Function that send odometry

        odom_payload_msg.header.frame_id = "world"
        odom_payload_msg.header.stamp = self.get_clock().now().to_msg()

        odom_payload_msg.pose.pose.position.x = position[0]
        odom_payload_msg.pose.pose.position.y = position[1]
        odom_payload_msg.pose.pose.position.z = position[2]

        odom_payload_msg.pose.pose.orientation.x = quat[1]
        odom_payload_msg.pose.pose.orientation.y = quat[2]
        odom_payload_msg.pose.pose.orientation.z = quat[3]
        odom_payload_msg.pose.pose.orientation.w = quat[0]

        # Send Messag
        publisher_payload_odom.publish(odom_payload_msg)
        return None 

    def publish_transforms(self, payload, quad1, quad2, quad3, quad4, unit_vectors):
        # Payload
## -------------------------------------------------------------------------------------------------------------------
        tf_world_load = TransformStamped()
        tf_world_load.header.stamp = self.get_clock().now().to_msg()
        tf_world_load.header.frame_id = 'world'            # <-- world is the parent
        tf_world_load.child_frame_id = 'payload'          # <-- imu_link is rotated
        tf_world_load.transform.translation.x = payload[0]
        tf_world_load.transform.translation.y = payload[1]
        tf_world_load.transform.translation.z = payload[2]
        tf_world_load.transform.rotation.x = payload[7]
        tf_world_load.transform.rotation.y = payload[8]
        tf_world_load.transform.rotation.z = payload[9]
        tf_world_load.transform.rotation.w = payload[6]

        # Quadrotor
        tf_world_quad1 = TransformStamped()
        tf_world_quad1.header.stamp = self.get_clock().now().to_msg()
        tf_world_quad1.header.frame_id = 'world'            # <-- world is the parent
        tf_world_quad1.child_frame_id = 'quad1'          # <-- imu_link is rotated
        tf_world_quad1.transform.translation.x = quad1[0]
        tf_world_quad1.transform.translation.y = quad1[1]
        tf_world_quad1.transform.translation.z = quad1[2]
        tf_world_quad1.transform.rotation.x = quad1[7]
        tf_world_quad1.transform.rotation.y = quad1[8]
        tf_world_quad1.transform.rotation.z = quad1[9]
        tf_world_quad1.transform.rotation.w = quad1[6]

        tf_world_quad2 = TransformStamped()
        tf_world_quad2.header.stamp = self.get_clock().now().to_msg()
        tf_world_quad2.header.frame_id = 'world'            # <-- world is the parent
        tf_world_quad2.child_frame_id = 'quad2'          # <-- imu_link is rotated
        tf_world_quad2.transform.translation.x = quad2[0]
        tf_world_quad2.transform.translation.y = quad2[1]
        tf_world_quad2.transform.translation.z = quad2[2]
        tf_world_quad2.transform.rotation.x = quad2[7]
        tf_world_quad2.transform.rotation.y = quad2[8]
        tf_world_quad2.transform.rotation.z = quad2[9]
        tf_world_quad2.transform.rotation.w = quad2[6]

        tf_world_quad3 = TransformStamped()
        tf_world_quad3.header.stamp = self.get_clock().now().to_msg()
        tf_world_quad3.header.frame_id = 'world'            # <-- world is the parent
        tf_world_quad3.child_frame_id = 'quad3'          # <-- imu_link is rotated
        tf_world_quad3.transform.translation.x = quad3[0]
        tf_world_quad3.transform.translation.y = quad3[1]
        tf_world_quad3.transform.translation.z = quad3[2]
        tf_world_quad3.transform.rotation.x = quad3[7]
        tf_world_quad3.transform.rotation.y = quad3[8]
        tf_world_quad3.transform.rotation.z = quad3[9]
        tf_world_quad3.transform.rotation.w = quad3[6]
        
        tf_world_quad4 = TransformStamped()
        tf_world_quad4.header.stamp = self.get_clock().now().to_msg()
        tf_world_quad4.header.frame_id = 'world'            # <-- world is the parent
        tf_world_quad4.child_frame_id = 'quad4'          # <-- imu_link is rotated
        tf_world_quad4.transform.translation.x = quad4[0]
        tf_world_quad4.transform.translation.y = quad4[1]
        tf_world_quad4.transform.translation.z = quad4[2]
        tf_world_quad4.transform.rotation.x = quad4[7]
        tf_world_quad4.transform.rotation.y = quad4[8]
        tf_world_quad4.transform.rotation.z = quad4[9]
        tf_world_quad4.transform.rotation.w = quad4[6]
## --------------------------------------------------------------------------------------------------------------------
        ro = self.ro_w(payload)
        tf_payload_p1 = TransformStamped()
        tf_payload_p1.header.stamp = self.get_clock().now().to_msg()
        tf_payload_p1.header.frame_id = 'payload'
        tf_payload_p1.child_frame_id = 'p1'
        tf_payload_p1.transform.translation.x = self.p1[0]
        tf_payload_p1.transform.translation.y = self.p1[1]
        tf_payload_p1.transform.translation.z = self.p1[2]

        tf_payload_p2 = TransformStamped()
        tf_payload_p2.header.stamp = self.get_clock().now().to_msg()
        tf_payload_p2.header.frame_id = 'payload'
        tf_payload_p2.child_frame_id = 'p2'
        tf_payload_p2.transform.translation.x = self.p2[0]
        tf_payload_p2.transform.translation.y = self.p2[1]
        tf_payload_p2.transform.translation.z = self.p2[2]

        tf_payload_p3 = TransformStamped()
        tf_payload_p3.header.stamp = self.get_clock().now().to_msg()
        tf_payload_p3.header.frame_id = 'payload'
        tf_payload_p3.child_frame_id = 'p3'
        tf_payload_p3.transform.translation.x = self.p3[0]
        tf_payload_p3.transform.translation.y = self.p3[1]
        tf_payload_p3.transform.translation.z = self.p3[2]

        tf_payload_p4 = TransformStamped()
        tf_payload_p4.header.stamp = self.get_clock().now().to_msg()
        tf_payload_p4.header.frame_id = 'payload'
        tf_payload_p4.child_frame_id = 'p4'
        tf_payload_p4.transform.translation.x = self.p4[0]
        tf_payload_p4.transform.translation.y = self.p4[1]
        tf_payload_p4.transform.translation.z = self.p4[2]

        tf_world_p1 = TransformStamped()
        tf_world_p1.header.stamp = self.get_clock().now().to_msg()
        tf_world_p1.header.frame_id = 'world'
        tf_world_p1.child_frame_id = 'p1_aux'
        tf_world_p1.transform.translation.x = ro[0, 0]
        tf_world_p1.transform.translation.y = ro[1, 0]
        tf_world_p1.transform.translation.z = ro[2, 0]

        tf_world_p2 = TransformStamped()
        tf_world_p2.header.stamp = self.get_clock().now().to_msg()
        tf_world_p2.header.frame_id = 'world'
        tf_world_p2.child_frame_id = 'p2_aux'
        tf_world_p2.transform.translation.x = ro[0, 1]
        tf_world_p2.transform.translation.y = ro[1, 1]
        tf_world_p2.transform.translation.z = ro[2, 1]

        tf_world_p3 = TransformStamped()
        tf_world_p3.header.stamp = self.get_clock().now().to_msg()
        tf_world_p3.header.frame_id = 'world'
        tf_world_p3.child_frame_id = 'p3_aux'
        tf_world_p3.transform.translation.x = ro[0, 2]
        tf_world_p3.transform.translation.y = ro[1, 2]
        tf_world_p3.transform.translation.z = ro[2, 2]

        tf_world_p4 = TransformStamped()
        tf_world_p4.header.stamp = self.get_clock().now().to_msg()
        tf_world_p4.header.frame_id = 'world'
        tf_world_p4.child_frame_id = 'p4_aux'
        tf_world_p4.transform.translation.x = ro[0, 3]
        tf_world_p4.transform.translation.y = ro[1, 3]
        tf_world_p4.transform.translation.z = ro[2, 3]

        quadrotors = self.quadrotors_w(payload)
        tf_world_q1 = TransformStamped()
        tf_world_q1.header.stamp = self.get_clock().now().to_msg()
        tf_world_q1.header.frame_id = 'world'
        tf_world_q1.child_frame_id = 'quadrotor_1_world'
        tf_world_q1.transform.translation.x = quadrotors[0, 0]
        tf_world_q1.transform.translation.y = quadrotors[1, 0]
        tf_world_q1.transform.translation.z = quadrotors[2, 0]

        tf_world_q2 = TransformStamped()
        tf_world_q2.header.stamp = self.get_clock().now().to_msg()
        tf_world_q2.header.frame_id = 'world'
        tf_world_q2.child_frame_id = 'quadrotor_2_world'
        tf_world_q2.transform.translation.x = quadrotors[0, 1]
        tf_world_q2.transform.translation.y = quadrotors[1, 1]
        tf_world_q2.transform.translation.z = quadrotors[2, 1]

        tf_world_q3 = TransformStamped()
        tf_world_q3.header.stamp = self.get_clock().now().to_msg()
        tf_world_q3.header.frame_id = 'world'
        tf_world_q3.child_frame_id = 'quadrotor_3_world'
        tf_world_q3.transform.translation.x = quadrotors[0, 2]
        tf_world_q3.transform.translation.y = quadrotors[1, 2]
        tf_world_q3.transform.translation.z = quadrotors[2, 2]

        tf_world_q4 = TransformStamped()
        tf_world_q4.header.stamp = self.get_clock().now().to_msg()
        tf_world_q4.header.frame_id = 'world'
        tf_world_q4.child_frame_id = 'quadrotor_4_world'
        tf_world_q4.transform.translation.x = quadrotors[0, 3]
        tf_world_q4.transform.translation.y = quadrotors[1, 3]
        tf_world_q4.transform.translation.z = quadrotors[2, 3]

        # Extract cable direction
        n = payload[13:25]
        tension = n.reshape((3, self.p.shape[1]), order='F')

        tf_p1_q1 = TransformStamped()
        tf_p1_q1.header.stamp = self.get_clock().now().to_msg()
        tf_p1_q1.header.frame_id = 'p1_aux'
        tf_p1_q1.child_frame_id = 'quadrotor_1'
        data_p1_q1 = -(tension[:, 0])*self.length
        tf_p1_q1.transform.translation.x = data_p1_q1[0]
        tf_p1_q1.transform.translation.y = data_p1_q1[1]
        tf_p1_q1.transform.translation.z = data_p1_q1[2]

        tf_p2_q2 = TransformStamped()
        tf_p2_q2.header.stamp = self.get_clock().now().to_msg()
        tf_p2_q2.header.frame_id = 'p2_aux'
        tf_p2_q2.child_frame_id = 'quadrotor_2'
        data_p2_q2 = -(tension[:, 1])*self.length
        tf_p2_q2.transform.translation.x = data_p2_q2[0]
        tf_p2_q2.transform.translation.y = data_p2_q2[1]
        tf_p2_q2.transform.translation.z = data_p2_q2[2]

        tf_p3_q3 = TransformStamped()
        tf_p3_q3.header.stamp = self.get_clock().now().to_msg()
        tf_p3_q3.header.frame_id = 'p3_aux'
        tf_p3_q3.child_frame_id = 'quadrotor_3'
        data_p3_q3 = -(tension[:, 2])*self.length
        tf_p3_q3.transform.translation.x = data_p3_q3[0]
        tf_p3_q3.transform.translation.y = data_p3_q3[1]
        tf_p3_q3.transform.translation.z = data_p3_q3[2]

        tf_p4_q4 = TransformStamped()
        tf_p4_q4.header.stamp = self.get_clock().now().to_msg()
        tf_p4_q4.header.frame_id = 'p4_aux'
        tf_p4_q4.child_frame_id = 'quadrotor_4'
        data_p4_q4 = -(tension[:, 3])*self.length
        tf_p4_q4.transform.translation.x = data_p4_q4[0]
        tf_p4_q4.transform.translation.y = data_p4_q4[1]
        tf_p4_q4.transform.translation.z = data_p4_q4[2]

        # Real Unit vector of the system
        tf_p1_q1_real = TransformStamped()
        tf_p1_q1_real.header.stamp = self.get_clock().now().to_msg()
        tf_p1_q1_real.header.frame_id = 'p1_aux'
        tf_p1_q1_real.child_frame_id = '                      quadrotor_1_real'
        data_p1_q1_real = - unit_vectors[0:3]
        tf_p1_q1_real.transform.translation.x = data_p1_q1_real[0]
        tf_p1_q1_real.transform.translation.y = data_p1_q1_real[1]
        tf_p1_q1_real.transform.translation.z = data_p1_q1_real[2]

        tf_p2_q2_real = TransformStamped()
        tf_p2_q2_real.header.stamp = self.get_clock().now().to_msg()
        tf_p2_q2_real.header.frame_id = 'p2_aux'
        tf_p2_q2_real.child_frame_id = '                      quadrotor_2_real'
        data_p2_q2_real = - unit_vectors[3:6]
        tf_p2_q2_real.transform.translation.x = data_p2_q2_real[0]
        tf_p2_q2_real.transform.translation.y = data_p2_q2_real[1]
        tf_p2_q2_real.transform.translation.z = data_p2_q2_real[2]

        tf_p3_q3_real = TransformStamped()
        tf_p3_q3_real.header.stamp = self.get_clock().now().to_msg()
        tf_p3_q3_real.header.frame_id = 'p3_aux'
        tf_p3_q3_real.child_frame_id = '                      quadrotor_3_real'
        data_p3_q3_real = - unit_vectors[6:9]
        tf_p3_q3_real.transform.translation.x = data_p3_q3_real[0]
        tf_p3_q3_real.transform.translation.y = data_p3_q3_real[1]
        tf_p3_q3_real.transform.translation.z = data_p3_q3_real[2]

        tf_p4_q4_real = TransformStamped()
        tf_p4_q4_real.header.stamp = self.get_clock().now().to_msg()
        tf_p4_q4_real.header.frame_id = 'p4_aux'
        tf_p4_q4_real.child_frame_id = '                      quadrotor_4_real'
        data_p4_q4_real = - unit_vectors[9:12]
        tf_p4_q4_real.transform.translation.x = data_p4_q4_real[0]
        tf_p4_q4_real.transform.translation.y = data_p4_q4_real[1]
        tf_p4_q4_real.transform.translation.z = data_p4_q4_real[2]

        ## Broadcast both transforms
        self.tf_broadcaster.sendTransform([tf_world_load, tf_payload_p1, tf_payload_p2, tf_payload_p3, tf_payload_p4, tf_p1_q1, tf_p2_q2, tf_p3_q3, tf_p4_q4, tf_world_p1, tf_world_p2, tf_world_p3, tf_world_p4, tf_world_q1, tf_world_q2, tf_world_q3, tf_world_q4, tf_world_quad1, tf_world_quad2, tf_world_quad3, tf_world_quad4, tf_p1_q1_real, tf_p2_q2_real, tf_p3_q3_real, tf_p4_q4_real])
        return None

    def validation(self):
        # Simluation
        for k in range(0, self.t.shape[0] - self.N_prediction):
            tic = time.time()
            # Send Odometry ros
            self.send_odometry(self.x_0, self.odom_payload_msg, self.publisher_payload_odom_)
            self.publish_transforms(self.x_0)

            # Section to guarantee same sample times
            while (time.time() - tic <= self.ts):
                pass
            toc = time.time() - tic
            self.get_logger().info(f"Sample time: {toc:.6f} seconds")
            self.get_logger().info(f"time: {self.t[k]:.6f} seconds")
            self.get_logger().info("PAYLOAD DYNAMICS")

    def run(self):
        # Set the states to simulate
        x = np.zeros((self.n_x, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        u = np.zeros((self.n_u, self.t.shape[0] - self.N_prediction), dtype=np.double)

        # initial States of the system
        x[:, 0] = self.x_0

        ## Desired states and control actions
        xd = np.zeros((self.n_x, self.t.shape[0] + 1), dtype=np.double)
        ud = np.zeros((self.n_u, self.t.shape[0]), dtype=np.double)

        ### Set desired states
        xd[0, :] = 2
        xd[1, :] = 2
        xd[2, :] = 2

        xd[3, :] = 0.0
        xd[4, :] = 0.0
        xd[5, :] = 0.0

        theta1 = 1*np.pi/2
        n1 = np.array([0.0, 0.0, 1.0])
        qd = np.concatenate(([np.cos(theta1 / 2)], np.sin(theta1 / 2) * n1))

        xd[6, :] = qd[0]
        xd[7, :] = qd[1]
        xd[8, :] = qd[2]
        xd[9, :] = qd[3]
        ####
        xd[10, :] = 0.0
        xd[11, :] = 0.0
        xd[12, :] = 0.0

        xd[13, :] = 0.0
        xd[14, :] = 0.0
        xd[15, :] = -1.0

        xd[16, :] = 0.0
        xd[17, :] = 0.0
        xd[18, :] = -1.0

        xd[19, :] = 0.0
        xd[20, :] = 0.0
        xd[21, :] = -1.0

        xd[22, :] = 0.0
        xd[23, :] = 0.0
        xd[24, :] = -1.0

        xd[25, :] = 0.0
        xd[26, :] = 0.0
        xd[27, :] = 0.0

        xd[28, :] = 0.0
        xd[29, :] = 0.0
        xd[30, :] = 0.0

        xd[31, :] = 0.0
        xd[32, :] = 0.0
        xd[33, :] = 0.0

        xd[34, :] = 0.0
        xd[35, :] = 0.0
        xd[36, :] = 0.0

        ### Set Desired Control Actions
        ud[0, :] = self.tensions_init[0]
        ud[1, :] = self.tensions_init[1]
        ud[2, :] = self.tensions_init[2]
        ud[3, :] = self.tensions_init[3]

        ud[4, :] = 0.0
        ud[5, :] = 0.0
        ud[6, :] = 0.0

        ud[7, :] = 0.0
        ud[8, :] = 0.0
        ud[9, :] = 0.0

        ud[10, :] = 0.0
        ud[11, :] = 0.0
        ud[12, :] = 0.0

        ud[13, :] = 0.0
        ud[14, :] = 0.0
        ud[15, :] = 0.0

        u[:, 0] = ud[:, 0]

        ### Create function to compute quadrotor position
        quadrotor_position = self.quadrotor_position_c()
        quadrotor_velocity = self.quadrotor_velocity_c()
        quadrotor_acceleration = self.quadrotor_acceleration_c()
        unit_vector_from_measurements = self.quadrotor_payload_unit_vector_c()
        ### Positions and velocities from the planning
        xQ = np.zeros((self.robot_num*3, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        xQ[:, 0] = np.array(quadrotor_position(x[:, 0])).reshape((self.robot_num*3, ))

        xQ_dot = np.zeros((self.robot_num*3, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        xQ_dot[:, 0] = np.array(quadrotor_velocity(x[:, 0])).reshape((self.robot_num*3, ))

        xQ_dot_dot = np.zeros((self.robot_num*3, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        xQ_dot_dot[:, 0] = np.array(quadrotor_acceleration(x[:, 0], u[:, 0])).reshape((self.robot_num*3, ))


        ### Empty states for each quadrotor
        xq1 = np.zeros((self.nx_quad, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        uq1 = np.zeros((self.nu_quad, self.t.shape[0] - self.N_prediction), dtype=np.double)
        xq1[:, 0] = self.xq1_0

        xq2 = np.zeros((self.nx_quad, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        uq2 = np.zeros((self.nu_quad, self.t.shape[0] - self.N_prediction), dtype=np.double)
        xq2[:, 0] = self.xq2_0

        xq3 = np.zeros((self.nx_quad, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        uq3 = np.zeros((self.nu_quad, self.t.shape[0] - self.N_prediction), dtype=np.double)
        xq3[:, 0] = self.xq3_0

        xq4 = np.zeros((self.nx_quad, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        uq4 = np.zeros((self.nu_quad, self.t.shape[0] - self.N_prediction), dtype=np.double)
        xq4[:, 0] = self.xq4_0

        unit = np.zeros((self.robot_num*3, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        xquadrotors = np.hstack((xq1[0:3, 0], xq2[0:3, 0], xq3[0:3, 0], xq4[0:3, 0])) 
        unit[:, 0] = np.array(unit_vector_from_measurements(x[:, 0], xquadrotors)).reshape((self.robot_num*3, ))

        ### Reset Solver
        self.acados_ocp_solver.reset()

        ### Initial Conditions optimization problem
        for stage in range(self.N_prediction + 1):
            self.acados_ocp_solver.set(stage, "x", x[:, 0])
        for stage in range(self.N_prediction):
            self.acados_ocp_solver.set(stage, "u", ud[:, 0])

        ### Simluation
        for k in range(0, self.t.shape[0] - self.N_prediction):
            tic = time.time()
            # Send Odometry ros
            self.send_odometry(x[:, k], self.odom_payload_msg, self.publisher_payload_odom_)
            self.send_odometry(xd[:, k], self.odom_payload_desired_msg, self.publisher_payload_desired_odom_)
            self.publish_transforms(x[:, k], xq1[:, k], xq2[:, k], xq3[:, k], xq4[:, k], unit[:, k])

            self.acados_ocp_solver.set(0, "lbx", x[:, k])
            self.acados_ocp_solver.set(0, "ubx", x[:, k])

            # Desired Trajectory of the system
            for j in range(self.N_prediction):
                yref = xd[:,k+j]
                uref = ud[:,k+j]
                aux_ref = np.hstack((yref, uref))
                self.acados_ocp_solver.set(j, "p", aux_ref)

            # Desired Trayectory at the last Horizon
            yref_N = xd[:,k+self.N_prediction]
            uref_N = ud[:,k+self.N_prediction]
            aux_ref_N = np.hstack((yref_N, uref_N))
            self.acados_ocp_solver.set(self.N_prediction, "p", aux_ref_N)

            # Check Solution since there can be possible errors 
            self.acados_ocp_solver.solve()

            aux_control = self.acados_ocp_solver.get(0, "u")
            u[:, k] = aux_control

            # Update Data of the system
            self.acados_integrator.set("x", x[:, k])
            self.acados_integrator.set("u", u[:, k])

            status_integral = self.acados_integrator.solve()
            xcurrent = self.acados_integrator.get("x")
            x[:, k+1] = xcurrent

            # External Forces
            u_ext_1 = np.array([u[0, k], x[13, k+1], x[14, k+1], x[15, k+1]])
            u_ext_2 = np.array([u[1, k], x[16, k+1], x[17, k+1], x[18, k+1]])
            u_ext_3 = np.array([u[2, k], x[19, k+1], x[20, k+1], x[21, k+1]])
            u_ext_4 = np.array([u[3, k], x[22, k+1], x[23, k+1], x[24, k+1]])

            # Update QUadrotors positions
            xQ[:, k+1] = np.array(quadrotor_position(x[:, k+1])).reshape((self.robot_num*3, ))
            xQ_dot[:, k+1] = np.array(quadrotor_velocity(x[:, k+1])).reshape((self.robot_num*3, ))
            xQ_dot_dot[:, k+1] = np.array(quadrotor_acceleration(x[:, k+1], u[:, k])).reshape((self.robot_num*3, ))

            # Ccntrol of each quadrotor
            uM_q1 = self.geometric_control(xq1[0:3, k], xQ[0:3, k+1], xq1[3:6, k], xQ_dot[0:3, k+1], xQ_dot_dot[0:3, k+1], xq1[6:10, k], 0.0,  xq1[10:13, k], np.zeros((3, )), u[0, k], x[13:16, k])
            uM_q2 = self.geometric_control(xq2[0:3, k], xQ[3:6, k+1], xq2[3:6, k], xQ_dot[3:6, k+1], xQ_dot_dot[3:6, k+1], xq2[6:10, k], 0.0,  xq2[10:13, k], np.zeros((3, )), u[1, k], x[16:19, k])
            uM_q3 = self.geometric_control(xq3[0:3, k], xQ[6:9, k+1], xq3[3:6, k], xQ_dot[6:9, k+1], xQ_dot_dot[6:9, k+1], xq3[6:10, k], 0.0,  xq3[10:13, k], np.zeros((3, )), u[2, k], x[19:22, k])
            uM_q4 = self.geometric_control(xq4[0:3, k], xQ[9:12, k+1], xq4[3:6, k], xQ_dot[9:12, k+1], xQ_dot_dot[9:12, k+1], xq4[6:10, k], 0.0,  xq4[10:13, k], np.zeros((3, )), u[3, k], x[22:25, k])

            # Update quadrotors
            xq1[:, k+1] = np.array(self.quad_1_model(xq1[:, k], uM_q1, u_ext_1, self.ts)).reshape((self.nx_quad, ))
            xq2[:, k+1] = np.array(self.quad_1_model(xq2[:, k], uM_q2, u_ext_2, self.ts)).reshape((self.nx_quad, ))
            xq3[:, k+1] = np.array(self.quad_1_model(xq3[:, k], uM_q3, u_ext_3, self.ts)).reshape((self.nx_quad, ))
            xq4[:, k+1] = np.array(self.quad_1_model(xq4[:, k], uM_q4, u_ext_4, self.ts)).reshape((self.nx_quad, ))

            xquadrotors = np.hstack((xq1[0:3, k+1], xq2[0:3, k+1], xq3[0:3, k+1], xq4[0:3, k+1])) 
            unit[:, k + 1] = np.array(unit_vector_from_measurements(x[:, k+1], xquadrotors)).reshape((self.robot_num*3, ))
            
            # Section to guarantee same sample times
            while (time.time() - tic <= self.ts):
                pass
            toc = time.time() - tic
            self.get_logger().info(f"Sample time: {toc:.6f} seconds")
            self.get_logger().info(f"time: {self.t[k]:.6f} seconds")
            self.get_logger().info("PAYLOAD CONTROL")
        
        ### Plot the results 
        #plot_tensions(self.t[0:u.shape[1]], u[0:3, :])
        #plot_angular_velocities(self.t[0:u.shape[1]], u[3:6, :], u[6:9, :], u[9:12, :])
        #plot_quad_position(self.t[0:xQ.shape[1]], xQ[0:3, :], xQ[3:6, :], xQ[6:9, :])
        #plot_quad_velocity(self.t[0:xQ_dot.shape[1]], xQ_dot[0:3, :], xQ_dot[3:6, :], xQ_dot[6:9, :])
        #plot_quad_position_desired(self.t[0:xq1.shape[1]], xq1[0:3, :], xq2[0:3, :], xq3[0:3, :], xQ[0:3, :], xQ[3:6, :], xQ[6:9, :])
        #plot_quad_velocity_desired(self.t[0:xq1.shape[1]], xq1[3:6, :], xq2[3:6, :], xq3[3:6, :], xQ_dot[0:3, :], xQ_dot[3:6, :], xQ_dot[6:9, :])
        #plot_angular_velocities_aux(self.t[0:rQ.shape[1]], rQ[0:3, :], rQ[3:6, :], rQ[6:9, :])
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