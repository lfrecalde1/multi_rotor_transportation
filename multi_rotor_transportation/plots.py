import matplotlib.pyplot as plt

def plot_tensions(time_s, data):
    # Define IEEE-style font settings
    plt.rcParams.update({
        "font.size": 14,                    # Base font size
        "axes.labelsize": 16,               # Axis labels
        "legend.fontsize": 14,              # Legend
        "xtick.labelsize": 13,              # Tick labels
        "ytick.labelsize": 13
    })

    fig, ax = plt.subplots(figsize=(8, 2))
    labels = ["t1", "t2", "t3"]

    # Define line styles for each curve
    linestyles = ["-", "--", "-."]   # solid, dashed, dash-dot
    colors = ["C0", "C2", "C1"]

    for i in range(data.shape[0]):
        ax.plot(
            time_s,
            data[i, :],
            label=labels[i],
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i % len(colors)]
        )

    ax.legend(loc="upper left")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tension [N]")
    ax.grid()
    ax.autoscale(tight=True)
    fig.savefig("Tensions.pdf", dpi=300, bbox_inches="tight")
    return None

def plot_angular_velocities(time_s, r1, r2, r3):
    # Define IEEE-style font settings
    plt.rcParams.update({
        "font.size": 14,                    # Base font size
        "axes.labelsize": 16,               # Axis labels
        "legend.fontsize": 14,              # Legend
        "xtick.labelsize": 13,              # Tick labels
        "ytick.labelsize": 13
    })

    # Create three vertical subplots
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    labels = ["rx", "ry", "rz"]
    linestyles = ["-", "--", "-."]
    colors = ["C0", "C2", "C1"]

    datasets = [r1, r2, r3]
    titles = ["Cable 1", "Cable 2", "Cable 3"]

    for j, (r, ax, title) in enumerate(zip(datasets, axes, titles)):
        for i in range(r.shape[0]):
            ax.plot(
                time_s,
                r[i, :],
                label=labels[i],
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)]
            )
        ax.legend(loc="upper left")
        ax.set_ylabel("Angular Vel [rad/s]")
        ax.set_title(title)
        ax.grid()
        ax.autoscale(tight=True)

    axes[-1].set_xlabel("Time [s]")  # Only bottom subplot has x-label

    fig.tight_layout()
    fig.savefig("AngularVelocities.pdf", dpi=300, bbox_inches="tight")
    return None

def plot_quad_position(time_s, r1, r2, r3):
    # Define IEEE-style font settings
    plt.rcParams.update({
        "font.size": 14,                    # Base font size
        "axes.labelsize": 16,               # Axis labels
        "legend.fontsize": 14,              # Legend
        "xtick.labelsize": 13,              # Tick labels
        "ytick.labelsize": 13
    })

    # Create three vertical subplots
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    labels = ["x", "y", "z"]
    linestyles = ["-", "--", "-."]
    colors = ["r", "g", "b"]

    datasets = [r1, r2, r3]
    titles = ["Quadrotor 1", "Quadrotor 2", "Quadrotor 3"]

    for j, (r, ax, title) in enumerate(zip(datasets, axes, titles)):
        for i in range(r.shape[0]):
            ax.plot(
                time_s,
                r[i, :],
                label=labels[i],
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)]
            )
        ax.legend(loc="upper left")
        ax.set_ylabel("Position [m]")
        ax.set_title(title)
        ax.grid()
        ax.autoscale(tight=True)

    axes[-1].set_xlabel("Time [s]")  # Only bottom subplot has x-label

    fig.tight_layout()
    fig.savefig("Position_quadrotors.pdf", dpi=300, bbox_inches="tight")
    return None

def plot_quad_velocity(time_s, r1, r2, r3):
    # Define IEEE-style font settings
    plt.rcParams.update({
        "font.size": 14,                    # Base font size
        "axes.labelsize": 16,               # Axis labels
        "legend.fontsize": 14,              # Legend
        "xtick.labelsize": 13,              # Tick labels
        "ytick.labelsize": 13
    })

    # Create three vertical subplots
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    labels = ["vx", "vy", "vz"]
    linestyles = ["-", "--", "-."]
    colors = ["r", "g", "b"]

    datasets = [r1, r2, r3]
    titles = ["Quadrotor 1", "Quadrotor 2", "Quadrotor 3"]

    for j, (r, ax, title) in enumerate(zip(datasets, axes, titles)):
        for i in range(r.shape[0]):
            ax.plot(
                time_s,
                r[i, :],
                label=labels[i],
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)]
            )
        ax.legend(loc="upper left")
        ax.set_ylabel("Velocity [m/s]")
        ax.set_title(title)
        ax.grid()
        ax.autoscale(tight=True)

    axes[-1].set_xlabel("Time [s]")  # Only bottom subplot has x-label

    fig.tight_layout()
    fig.savefig("Velocity_quadrotors.pdf", dpi=300, bbox_inches="tight")
    return None