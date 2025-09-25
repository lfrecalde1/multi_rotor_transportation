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
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

class PayloadControlNode(Node):
    def __init__(self):
        super().__init__('PayloadDynamics')

        # Time Definition
        self.ts = 0.03
        self.final = 20
        self.t =np.arange(0, self.final + self.ts, self.ts, dtype=np.double)

        # Prediction Node of the NMPC formulation
        self.t_N = 0.5
        self.N = np.arange(0, self.t_N + self.ts, self.ts)
        self.N_prediction = self.N.shape[0]

        # Internal parameters defintion
        self.robot_num = 3
        self.mass = 0.5
        self.M_load = self.mass *  np.eye((3))
        self.inertia = np.array([[0.013344, 0.0, 0.0], [0.0, 0.012810, 0.0], [0.0, 0.0, 0.03064]], dtype=np.double)
        self.gravity = 9.81

        # Control gains
        c1 = 1
        kv_min = c1 + 1/4 + 0.1
        kp_min = (c1*(kv_min*kv_min) + 2*kv_min*c1 - c1*c1)/((self.mass)*(4*(kv_min - c1)-1))
        kp_min = 80
        self.kp_min = kp_min
        self.kv_min = kv_min
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
        self.kw_min = kw_min
        self.c2 = c2
        print(kr_min)
        print(kw_min)
        print(c2)
        print("--------------------------")


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

        # Init Tension of the cables
        self.tensions_init = np.linalg.norm(tension_matrix, axis=0)
        self.n_init = tension_matrix/self.tensions_init
        self.n_init =  self.n_init.flatten(order='F')

        # Init states
        self.x_0 = np.hstack((pos_0, vel_0, quat_0, omega_0, self.n_init))
        print(self.x_0)

        # Init Control Actions or equilibirum
        r_init = np.array([0.0, 0.0, 0.0]*self.robot_num, dtype=np.double)
        self.u_equilibrium = np.hstack((self.tensions_init, r_init))
        print(self.u_equilibrium)

        # Maximum and minimun control actions
        tension_min = 1*self.tensions_init
        tension_max = 2.*self.tensions_init
        r_max = np.array([0.5, 0.5, 0.5]*self.robot_num, dtype=np.double)
        r_min = -r_max
        self.u_min =  np.hstack((tension_min, r_min))
        self.u_max =  np.hstack((tension_max, r_max))

        # Create Model and OCP of the sytem        
        self.payloadModel()
        self.ocp = self.solver(self.x_0)

        # OCP
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp_" + self.ocp.model.name + ".json", build= True, generate= True)

    
        # Integration using Acados
        self.acados_integrator = AcadosSimSolver(self.ocp, json_file="acados_sim_" + self.ocp.model.name + ".json", build= True, generate= True)

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
        u = ca.vertcat(t_1_cmd, t_2_cmd, t_3_cmd, rx_1_cmd, ry_1_cmd, rz_1_cmd, rx_2_cmd, ry_2_cmd, rz_2_cmd, rx_3_cmd, ry_3_cmd, rz_3_cmd)

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
        p = ca.MX.sym('p', 34, 1)

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

        # Get desired states of the system
        x_p_d = p[0:3]
        v_p_d = p[3:6]
        quat_d = p[6:10]
        omega_d = p[10:13]
        n1_d = p[13:16]
        n2_d = p[16:19]
        n3_d = p[19:22]

        # Cost Functions of the system
        cost_quaternion_f = self.cost_quaternion_c()
        cost_matrix_error_f = self.rotation_matrix_error_c()

        angular_error = cost_quaternion_f(quat_d, quat)
        print(angular_error.shape)
        angular_velocity_error = omega - cost_matrix_error_f(quat_d, quat)@omega_d
        print(angular_velocity_error.shape)

        error_position_quad = x_p - x_p_d
        error_velocity_quad = v_p - v_p_d

        # Cost functions
        lyapunov_position = (1/2)*self.kp_min*error_position_quad.T@error_position_quad + self.kv_min*(1/2)*(self.mass)*error_velocity_quad.T@error_velocity_quad + self.c1*error_position_quad.T@error_velocity_quad
        print(lyapunov_position.shape)
        lyapunov_orientation = self.kr_min*angular_error.T@angular_error + self.kw_min*(1/2)*angular_velocity_error.T@self.inertia@angular_velocity_error
        print(lyapunov_orientation.shape)

        
        ocp.model.cost_expr_ext_cost = lyapunov_position + lyapunov_orientation
        ocp.model.cost_expr_ext_cost_e = lyapunov_position + lyapunov_orientation

        ref_params = np.hstack((self.x_0, self.u_equilibrium))

        ocp.parameter_values = ref_params

        ocp.constraints.constr_type = 'BGH'

        # Set constraints
        ocp.constraints.lbu = self.u_min
        ocp.constraints.ubu = self.u_max
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        ocp.constraints.x0 = x0

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" 
        ocp.solver_options.qp_solver_cond_N = self.N_prediction // 4
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  
        ocp.solver_options.regularize_method = "CONVEXIFY"  
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.Tsim = self.ts
        ocp.solver_options.tf = self.t_N

        return ocp
        
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