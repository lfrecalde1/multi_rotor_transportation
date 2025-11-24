#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import casadi as ca
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from scipy.spatial.transform import Rotation as R
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# QP solver for CBF filter
import osqp
import scipy.sparse as sp


class QuadPointMassNMPCNode(Node):
    def __init__(self):
        super().__init__('quad_point_mass_nmpc')

        # ─────────────────────────────────────────
        # Time and horizon definition
        # ─────────────────────────────────────────
        self.ts = 0.03         # sampling time [s]
        self.t_N = 1.0         # horizon length [s]
        self.N_prediction = int(self.t_N / self.ts)  # number of shooting intervals

        # ─────────────────────────────────────────
        # ROS interfaces
        # ─────────────────────────────────────────
        # Odometry subscriber
        self.subscriber_drone_0_ = self.create_subscription(
            Odometry,
            "/drone_0/odom",
            self.callback_get_odometry_drone_0,
            10
        )

        # PositionCommand publisher (we will only use acceleration fields)
        self.publisher_ref_drone_0 = self.create_publisher(
            PositionCommand,
            "/drone_0/position_cmd",
            10
        )

        # ─────────────────────────────────────────
        # Initial state (position + velocity in inertial frame)
        # ─────────────────────────────────────────
        pos_quad_0 = np.array([-1.0, 0.0, 2.0], dtype=np.double)
        vel_quad_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        self.x_0 = np.hstack((pos_quad_0, vel_quad_0))  # 6x1: [px,py,pz,vx,vy,vz]

        self.n_x = self.x_0.shape[0]        # 6
        self.u_equilibrium = np.array([0.0, 0.0, 0.0], dtype=np.double)
        self.n_u = self.u_equilibrium.shape[0]  # 3 (ax, ay, az)

        # ─────────────────────────────────────────
        # Desired state and input (setpoint)
        # ─────────────────────────────────────────
        self.xd = np.zeros((self.n_x,), dtype=np.double)
        self.ud = np.zeros((self.n_u,), dtype=np.double)

        # Example: desired position = (1, 0, 2), desired velocity = 0
        self.xd[0] = 8.0   # p_x
        self.xd[1] = 0.0   # p_y
        self.xd[2] = 2.0   # p_z
        # velocities and desired accelerations remain zero

        # ─────────────────────────────────────────
        # Cost weights
        # ─────────────────────────────────────────
        self.kp_pos = 10.0   # position tracking weight
        self.kv_vel = 1.0    # velocity tracking weight
        self.ku_acc = 0.1    # acceleration effort weight

        # ─────────────────────────────────────────
        # Input constraints (accelerations)
        # ─────────────────────────────────────────
        a_max = 10.0  # [m/s^2], tune as needed
        self.u_min = np.array([-a_max, -a_max, -a_max], dtype=np.double)
        self.u_max = np.array([ a_max,  a_max,  a_max], dtype=np.double)

        # ─────────────────────────────────────────
        # CBF parameters and obstacles (in x–y plane)
        # ─────────────────────────────────────────
        # Obstacles at (2,-0.1), (4,0.1), (6,0)
        self.obstacles = np.array([
            [2.0, -0.1],
            [4.0,  0.1],
            [6.0,  0.0]
        ], dtype=np.double)

        # Safety radius around each obstacle
        self.R_safe = 0.3  # [m], tune as desired

        # 2nd order CBF gains (for h_ddot + k1 h_dot + k0 h >= 0)
        self.cbf_k0 = 1.0
        self.cbf_k1 = 10.0

        # ─────────────────────────────────────────
        # Internal: flag & solver handle
        # ─────────────────────────────────────────
        self.flag = 0
        self.ocp = None
        self.acados_ocp_solver = None

        # Control loop timer
        self.timer = self.create_timer(self.ts, self.run)

    # ─────────────────────────────────────────
    # ODOMETRY CALLBACK
    # ─────────────────────────────────────────
    def callback_get_odometry_drone_0(self, msg: Odometry):
        """
        Read quadrotor position and convert body linear velocity into inertial frame.
        State x = [p_x, p_y, p_z, v_x, v_y, v_z].
        """
        x = np.zeros((6,), dtype=np.double)

        # Position (already in inertial frame "world")
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Body-frame linear velocity from odom
        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        vz_b = msg.twist.twist.linear.z
        v_body = np.array([[vx_b], [vy_b], [vz_b]])

        # Orientation (to rotate body velocity into inertial frame)
        quat = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        R_ib = R.from_quat(quat).as_matrix()  # 3x3
        v_inertial = R_ib @ v_body

        x[3] = v_inertial[0, 0]
        x[4] = v_inertial[1, 0]
        x[5] = v_inertial[2, 0]

        # Update current state used as initial condition
        self.x_0 = x
        return None

    # ─────────────────────────────────────────
    # POINT-MASS MODEL
    # ─────────────────────────────────────────
    def point_mass_model(self) -> AcadosModel:
        """
        Reduced quadrotor point-mass model.
        State x = [p_x, p_y, p_z, v_x, v_y, v_z]
        Input u = [a_x, a_y, a_z]  (inertial accelerations)
        Dynamics:
            p_dot = v
            v_dot = u
        Parameters p = [x_d(6); u_d(3)] for reference tracking.
        """
        model_name = "quad_point_mass"

        # States
        p_x = ca.MX.sym('p_x')
        p_y = ca.MX.sym('p_y')
        p_z = ca.MX.sym('p_z')
        v_x = ca.MX.sym('v_x')
        v_y = ca.MX.sym('v_y')
        v_z = ca.MX.sym('v_z')
        x = ca.vertcat(p_x, p_y, p_z, v_x, v_y, v_z)

        # Controls (inertial accelerations)
        a_x = ca.MX.sym('a_x')
        a_y = ca.MX.sym('a_y')
        a_z = ca.MX.sym('a_z')
        u = ca.vertcat(a_x, a_y, a_z)

        # Dynamics
        p_dot = ca.vertcat(v_x, v_y, v_z)
        v_dot = ca.vertcat(a_x, a_y, a_z)
        f_expl = ca.vertcat(p_dot, v_dot)

        # Parameters: desired state (6) + desired input (3) = 9
        p = ca.MX.sym('p', 9, 1)

        model = AcadosModel()
        model.name = model_name
        model.x = x
        model.u = u
        model.p = p
        model.f_expl_expr = f_expl
        return model

    # ─────────────────────────────────────────
    # ACADOS OCP SETUP
    # ─────────────────────────────────────────
    def solver(self, x0: np.ndarray) -> AcadosOcp:
        """
        Build the Acados OCP for the point-mass quadrotor model:
        - External cost (quadratic tracking in p, v, and penalizing u)
        - Input constraints on accelerations
        """
        model = self.point_mass_model()

        ocp = AcadosOcp()
        ocp.model = model

        nx = model.x.size()[0]  # 6
        nu = model.u.size()[0]  # 3

        # Horizon
        ocp.dims.N = self.N_prediction

        # External cost
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        x = model.x
        u = model.u
        p = model.p

        # Split state
        p_x = x[0:3]  # position
        v_x = x[3:6]  # velocity

        # Desired from parameters
        p_x_d = p[0:3]
        v_x_d = p[3:6]
        u_d   = p[6:9]

        # Errors
        e_p = p_x - p_x_d
        e_v = v_x - v_x_d
        e_u = u   - u_d

        # Stage cost
        stage_cost = (
            self.kp_pos * ca.dot(e_p, e_p) +
            self.kv_vel * ca.dot(e_v, e_v) +
            self.ku_acc * ca.dot(e_u, e_u)
        )

        # Terminal cost (no control effort term)
        terminal_cost = (
            self.kp_pos * ca.dot(e_p, e_p) +
            self.kv_vel * ca.dot(e_v, e_v)
        )

        ocp.model.cost_expr_ext_cost   = stage_cost
        ocp.model.cost_expr_ext_cost_e = terminal_cost

        # Default parameter values
        ref_params = np.hstack((self.xd, self.ud))  # (6+3,)
        ocp.parameter_values = ref_params

        # Constraints
        ocp.constraints.constr_type = 'BGH'
        ocp.constraints.lbu = self.u_min
        ocp.constraints.ubu = self.u_max
        ocp.constraints.idxbu = np.array([0, 1, 2], dtype=np.int64)

        # Initial state constraint
        ocp.constraints.x0 = x0

        # Solver options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = max(1, self.N_prediction // 4)
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.Tsim = self.ts
        ocp.solver_options.tf = self.t_N

        return ocp

    # ─────────────────────────────────────────
    # HELPER: build solver only once
    # ─────────────────────────────────────────
    def prepare(self):
        if self.flag == 0:
            self.flag = 1

            # Build OCP
            self.ocp = self.solver(self.x_0)

            # Create acados solver (will also build & generate code)
            self.acados_ocp_solver = AcadosOcpSolver(
                self.ocp,
                json_file="acados_ocp_" + self.ocp.model.name + ".json",
                build=True,
                generate=True,
            )

            # Initialize states & controls along the horizon
            for stage in range(self.N_prediction + 1):
                self.acados_ocp_solver.set(stage, "x", self.x_0)
            for stage in range(self.N_prediction):
                self.acados_ocp_solver.set(stage, "u", self.ud)

        return None

    # ─────────────────────────────────────────
    # CBF-QP FILTER
    # ─────────────────────────────────────────
    def cbf_filter(self, x: np.ndarray, u_nom: np.ndarray) -> np.ndarray:
        """
        CBF-QP filter for 3 static obstacles in x–y plane.

        State: x = [px, py, pz, vx, vy, vz]
        Control: u = [ax, ay, az] (inertial acceleration)

        For each obstacle i with center (ox, oy), define:
            h_i = (px - ox)^2 + (py - oy)^2 - R_safe^2

        Second-order CBF condition:
            h_ddot + k1 * h_dot + k0 * h >= 0

        Using:
          h_dot  = 2(px-ox)vx + 2(py-oy)vy
          h_ddot = 2[(px-ox)ax + (py-oy)ay] + 2(vx^2 + vy^2)

        This yields a linear constraint in u:
            2[(px-ox)ax + (py-oy)ay] >= -2(vx^2+vy^2) - k1*h_dot - k0*h

        → In the form A u <= b for QP.
        """
        px, py, pz, vx, vy, vz = x

        # Build inequality constraints A u <= b
        A_rows = []
        b_vals = []

        for (ox, oy) in self.obstacles:
            dx = px - ox
            dy = py - oy

            h = dx*dx + dy*dy - self.R_safe**2
            h_dot = 2.0 * (dx*vx + dy*vy)
            v_sq = vx*vx + vy*vy

            # RHS of CBF inequality
            # 2[(dx)ax + (dy)ay] >= -2*v_sq - k1*h_dot - k0*h
            rhs = -2.0 * v_sq - self.cbf_k1 * h_dot - self.cbf_k0 * h

            # Rewrite as:
            # -2(dx)ax - 2(dy)ay <= -rhs
            Ai = np.array([
                -2.0 * dx,
                -2.0 * dy,
                0.0           # no constraint on az from obstacle
            ], dtype=np.double)

            bi = -rhs

            A_rows.append(Ai)
            b_vals.append(bi)

        # Box constraints on u: u_min <= u <= u_max
        # Encode as:
        #   u <= u_max
        #  -u <= -u_min
        A_box_upper = np.eye(3, dtype=np.double)
        b_box_upper = self.u_max.copy()

        A_box_lower = -np.eye(3, dtype=np.double)
        b_box_lower = -self.u_min.copy()

        A_rows.append(A_box_upper[0, :])
        A_rows.append(A_box_upper[1, :])
        A_rows.append(A_box_upper[2, :])
        b_vals.append(b_box_upper[0])
        b_vals.append(b_box_upper[1])
        b_vals.append(b_box_upper[2])

        A_rows.append(A_box_lower[0, :])
        A_rows.append(A_box_lower[1, :])
        A_rows.append(A_box_lower[2, :])
        b_vals.append(b_box_lower[0])
        b_vals.append(b_box_lower[1])
        b_vals.append(b_box_lower[2])

        A = np.vstack(A_rows)   # shape (m,3)
        b = np.array(b_vals, dtype=np.double)  # shape (m,)

        # QP:
        #   minimize   0.5 * (u - u_nom)^T W (u - u_nom)
        #   subject to A u <= b
        # Here W = I (can be tuned as diagonal if you want anisotropic penalty)
        W = 10*np.eye(3, dtype=np.double)

        # Rewrite as 0.5 u^T P u + q^T u
        # with P = 2W, q = -2W u_nom → equivalent to ||u - u_nom||^2 up to constant
        P = 2.0 * W
        q = -2.0 * W @ u_nom

        # Convert to sparse
        P_sp = sp.csc_matrix(P)
        A_sp = sp.csc_matrix(A)

        # OSQP form:  min 0.5 x^T P x + q^T x
        #             s.t. l <= A x <= u
        # We have A u <= b  →  l = -inf, u = b
        m = A.shape[0]
        l = -np.inf * np.ones(m)
        u_osqp = b

        try:
            prob = osqp.OSQP()
            prob.setup(P_sp, q, A_sp, l, u_osqp, verbose=False)
            res = prob.solve()

            if res.info.status_val not in [1, 2]:  # 1=solved, 2=solved inaccurate
                self.get_logger().warn(
                    f"OSQP CBF QP did not solve properly, status: {res.info.status}"
                )
                return u_nom

            u_safe = res.x
            return u_safe.astype(np.double)

        except Exception as e:
            self.get_logger().warn(f"CBF QP exception: {e}")
            return u_nom

    # ─────────────────────────────────────────
    # PUBLISH ONLY ACCELERATIONS
    # ─────────────────────────────────────────
    def send_acc_cmd(self, publisher, a: np.ndarray):
        """
        Fill only the acceleration field of PositionCommand and publish.
        """
        msg = PositionCommand()
        msg.acceleration.x = float(a[0])
        msg.acceleration.y = float(a[1])
        msg.acceleration.z = float(a[2])
        publisher.publish(msg)

    # ─────────────────────────────────────────
    # MAIN CONTROL LOOP
    # ─────────────────────────────────────────
    def run(self):
        # Build OCP/solver once
        self.prepare()

        # Update initial condition constraint using latest state x_0
        self.acados_ocp_solver.set(0, "lbx", self.x_0)
        self.acados_ocp_solver.set(0, "ubx", self.x_0)

        # Set reference trajectory (here constant) along the horizon
        for j in range(self.N_prediction):
            yref = self.xd  # desired state (6)
            uref = self.ud  # desired acceleration (3)
            aux_ref = np.hstack((yref, uref))  # (9,)
            self.acados_ocp_solver.set(j, "p", aux_ref)

        # Terminal stage parameters
        yref_N = self.xd
        uref_N = self.ud
        aux_ref_N = np.hstack((yref_N, uref_N))
        self.acados_ocp_solver.set(self.N_prediction, "p", aux_ref_N)

        # Solve NMPC
        status = self.acados_ocp_solver.solve()
        if status != 0:
            self.get_logger().warn(f"acados solver returned status {status}")

        # Extract first control input: inertial acceleration [a_x, a_y, a_z]
        u_nom = self.acados_ocp_solver.get(0, "u").copy()

        # Apply CBF-QP filter
        u_safe = self.cbf_filter(self.x_0, u_nom)

        # Publish ONLY accelerations
        self.send_acc_cmd(self.publisher_ref_drone_0, u_safe)


def main(args=None):
    rclpy.init(args=args)
    node = QuadPointMassNMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('NMPC + CBF node stopped manually.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

