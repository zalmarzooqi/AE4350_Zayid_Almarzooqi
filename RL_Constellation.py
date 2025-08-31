# rl_tcr_rolling_only_and_overlay.py
# ------------------------------------------------------------
# Trains the RL environment, then:
# 1) Plots ONLY the rolling average of TCR vs. episodes.
# 2) For the LAST episode, plots an overlay: initial (hollow) → final (solid/X)
#    with arrows showing the shortest torus displacement.
# ------------------------------------------------------------
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# -----------------------------
# Tunables
# -----------------------------
N_CARS          = 12
GRID_SIZE       = 20
ROUNDS_PER_EP   = 120
EPISODES        = 200

CAP_MIN_MAX     = (3, 7)
INIT_TASKS_MIN_MAX = (12, 28)
BASE_FAIL_P     = 0.05
P_GROWTH_EVERY  = 100
P_GROWTH_FACTOR = 1.5

DETECT_DELAY    = 1
RECONF_V        = 6.0
T_MAX           = 18

ALPHA           = 0.01
GAMMA           = 0.95
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY_EP    = 200

USE_TORUS       = True

RANDOM_SEED     = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -----------------------------
# Action space + Q-table
# -----------------------------
PAIRS = [(i, j) for i in range(N_CARS) for j in range(N_CARS) if i != j]
N_ACTIONS = len(PAIRS)

Q = {}
def q_get(s, a): return Q.get((s, a), 0.0)
def q_set(s, a, v): Q[(s, a)] = v
def best_action(s):
    vals = [q_get(s, a) for a in range(N_ACTIONS)]
    mx = max(vals)
    choices = [i for i, v in enumerate(vals) if abs(v - mx) < 1e-12]
    return random.choice(choices), mx

# -----------------------------
# Helpers (init, motion, distance)
# -----------------------------
def make_initial_velocity():
    """
    Random per-spacecraft velocities from {-1, 0, +1} per axis,
    ensuring no spacecraft starts fully stationary.
    """
    v = np.zeros((N_CARS, 2), dtype=float)
    choices = np.array([-1.0, 0.0, 1.0])
    v[:, 0] = np.random.choice(choices, size=N_CARS)
    v[:, 1] = np.random.choice(choices, size=N_CARS)
    stationary = np.where((v[:, 0] == 0.0) & (v[:, 1] == 0.0))[0]
    if stationary.size > 0:
        v[stationary, 0] = np.random.choice([-1.0, 1.0], size=stationary.size)
    return v

def torus_dist(i, j, xy):
    dx = abs(xy[i, 0] - xy[j, 0])
    dy = abs(xy[i, 1] - xy[j, 1])
    dx = min(dx, GRID_SIZE - dx)
    dy = min(dy, GRID_SIZE - dy)
    return math.hypot(dx, dy)

def euclid_dist(i, j, xy):
    dx = xy[i, 0] - xy[j, 0]
    dy = xy[i, 1] - xy[j, 1]
    return math.hypot(dx, dy)

def dist(i, j, xy):
    return torus_dist(i, j, xy) if USE_TORUS else euclid_dist(i, j, xy)

def move_cars(env):
    env["xy"][:] = (env["xy"] + env["v"]) % GRID_SIZE

# -----------------------------
# State abstraction (tiny, discrete)
# -----------------------------
def state_id(env):
    status = env["status"]
    tasks  = env["tasks"]
    cap    = env["cap"]
    op_idx = np.where(status == 1)[0]
    if len(op_idx) == 0:
        return (2, 2, 2)

    t_op = tasks[op_idx]
    imbalance = (t_op.max() - t_op.min()) if len(t_op) > 0 else 0.0
    congestion = int(np.sum(tasks > (2.0 * cap)))
    failed = N_CARS - len(op_idx)

    def b3(x, edges):
        if x <= edges[0]: return 0
        if x <= edges[1]: return 1
        return 2

    return (b3(failed, [0, 1]),
            b3(imbalance, [4, 9]),
            b3(congestion, [0, 2]))

# -----------------------------
# Environment init + failures
# -----------------------------
def init_episode(ep_idx):
    xy = np.random.randint(0, GRID_SIZE, size=(N_CARS, 2)).astype(float)
    v  = make_initial_velocity()
    cap = np.random.randint(CAP_MIN_MAX[0], CAP_MIN_MAX[1] + 1, size=N_CARS)
    tasks = np.random.randint(INIT_TASKS_MIN_MAX[0], INIT_TASKS_MIN_MAX[1] + 1, size=N_CARS).astype(float)

    bw = np.random.uniform(1.0, 5.0, size=(N_CARS, N_CARS))
    bw = (bw + bw.T) / 2.0
    np.fill_diagonal(bw, 0.0)

    mod = np.random.uniform(0.8, 1.2, size=(N_CARS, N_CARS))
    mod = (mod + mod.T) / 2.0
    np.fill_diagonal(mod, 1.0)

    status = np.ones(N_CARS, dtype=int)
    init_total_tasks = float(tasks.sum())
    pending_fail = {}

    eps = EPS_START + (min(1.0, ep_idx / EPS_DECAY_EP)) * (EPS_END - EPS_START)
    p_fail = BASE_FAIL_P * (P_GROWTH_FACTOR ** (ep_idx // P_GROWTH_EVERY))

    return {
        "xy": xy, "v": v, "cap": cap, "tasks": tasks, "bw": bw, "mod": mod,
        "status": status, "init_total": init_total_tasks,
        "pending_fail": pending_fail, "fail_times": [], "eps": eps, "p_fail": p_fail,
        "round": 0
    }

def maybe_fail(env):
    status = env["status"]; tasks = env["tasks"]; xy = env["xy"]
    p = env["p_fail"]; pend = env["pending_fail"]
    for i in range(N_CARS):
        if status[i] == 1 and random.random() < p:
            status[i] = 0
            op = np.where(status == 1)[0]
            if len(op) == 0:
                mean_d = 0.0
            else:
                ds = [dist(i, k, xy) for k in op]
                mean_d = float(np.mean(ds))
            reconf_steps = int(np.ceil(mean_d / max(1e-6, RECONF_V)))
            t_i = DETECT_DELAY + reconf_steps
            pend[i] = {"steps_left": t_i, "payload": float(tasks[i]), "t_i": t_i}

def tick_fail_timers_and_redistribute(env):
    pend = env["pending_fail"]
    if not pend: return
    status = env["status"]; tasks = env["tasks"]
    for idx, rec in list(pend.items()):
        rec["steps_left"] -= 1
        if rec["steps_left"] <= 0:
            op = np.where(status == 1)[0]
            if len(op) > 0:
                share = rec["payload"] / len(op)
                tasks[op] += share
            tasks[idx] = 0.0
            env["fail_times"].append(rec["t_i"])
            del pend[idx]

# -----------------------------
# Dynamics
# -----------------------------
def apply_transfer(env, action_idx):
    i, j = PAIRS[action_idx]
    status = env["status"]
    tasks  = env["tasks"]
    if i == j or status[i] == 0 or status[j] == 0:
        return -1.0, 0.0

    xy, bw, mod = env["xy"], env["bw"], env["mod"]
    d = dist(i, j, xy)
    effective_bw = bw[i, j] * mod[i, j]
    throughput = max(0.0, effective_bw - 0.08 * d)
    moved = min(tasks[i], max(0.0, np.floor(throughput)))
    if moved > 0:
        tasks[i] -= moved
        tasks[j] += moved
    delay = 1.0 + d / (effective_bw + 1e-6)
    return -0.02 * delay, moved

def process_tasks(env):
    status = env["status"]; tasks = env["tasks"]; cap = env["cap"]
    done_now = 0.0
    for i in range(N_CARS):
        if status[i] == 1:
            x = min(tasks[i], cap[i])
            tasks[i] -= x
            done_now += x
    return done_now

# -----------------------------
# One episode (optionally capture positions for last ep)
# -----------------------------
def run_episode(ep_idx, capture_positions=False):
    env = init_episode(ep_idx)
    s = state_id(env)
    init_total = env["init_total"]
    prev_remaining = env["tasks"].sum()

    init_xy = env["xy"].copy() if capture_positions else None
    init_status = env["status"].copy() if capture_positions else None

    for _ in range(ROUNDS_PER_EP):
        env["round"] += 1
        move_cars(env)

        # ε-greedy action
        if random.random() < env["eps"]:
            a = random.randrange(N_ACTIONS)
        else:
            a, _ = best_action(s)

        pen_delay, _moved = apply_transfer(env, a)
        _ = process_tasks(env)

        maybe_fail(env)
        tick_fail_timers_and_redistribute(env)

        # reward (learning)
        remaining = env["tasks"].sum()
        delta_done = (prev_remaining - remaining)
        prev_remaining = remaining
        op_idx = np.where(env["status"] == 1)[0]
        imb = (env["tasks"][op_idx].ptp() if len(op_idx) > 1 else 0.0)
        reward = delta_done + pen_delay - 0.01 * imb

        # Q-learning update
        s_next = state_id(env)
        _, q_best_next = best_action(s_next)
        old = q_get(s, a)
        q_set(s, a, old + ALPHA * (reward + GAMMA * q_best_next - old))
        s = s_next

        if remaining <= 1e-6 or np.all(env["status"] == 0):
            break

    # Episode TCR + final layout (if captured)
    remaining = env["tasks"].sum()
    tcr_final = ((init_total - remaining) / max(1e-9, init_total)) * 100.0

    final_xy = env["xy"].copy() if capture_positions else None
    final_status = env["status"].copy() if capture_positions else None
    final_round = env["round"] if capture_positions else None

    # Optional soft shaping on ART
    art = (np.mean(env["fail_times"]) if len(env["fail_times"]) > 0 else 0.0)
    if art > T_MAX:
        for a in range(N_ACTIONS):
            old = q_get(s, a)
            q_set(s, a, old - 0.01 * (art - T_MAX))

    return tcr_final, init_xy, init_status, final_xy, final_status, final_round

# -----------------------------
# Plotting: ONLY rolling average of TCR vs episodes
# -----------------------------
def plot_tcr_rolling_only(tcr_hist, window=25):
    if len(tcr_hist) < window:
        print(f"Not enough episodes ({len(tcr_hist)}) for rolling window={window}. Plotting skipped.")
        return
    episodes = np.arange(window, len(tcr_hist) + 1)
    roll = np.convolve(tcr_hist, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, roll, linewidth=2.4, label=f"Rolling avg TCR (win={window})")
    plt.xlabel("Episode")
    plt.ylabel("Task Completion Rate (TCR) [%]")
    plt.title("Rolling Average of TCR vs. Episode")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Plotting: overlay initial & final with arrows
# -----------------------------
def shortest_torus_vector(p_from, p_to, G):
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    if dx >  G/2: dx -= G
    if dx < -G/2: dx += G
    if dy >  G/2: dy -= G
    if dy < -G/2: dy += G
    return dx, dy

def plot_overlay_initial_final(init_xy, init_status, final_xy, final_status, final_round):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(N_CARS):
        xi, yi = init_xy[i]
        xf, yf = final_xy[i]
        dx, dy = shortest_torus_vector((xi, yi), (xf, yf), GRID_SIZE)
        x_end = xi + dx
        y_end = yi + dy

        # Initial marker (hollow)
        ax.scatter([xi], [yi], s=90, facecolors='none', edgecolors='C0', linewidths=1.8, zorder=3)
        ax.text(xi+0.2, yi+0.2, str(i), fontsize=9, color='C0')

        # Arrow initial -> final
        ax.annotate("", xy=(x_end, y_end), xytext=(xi, yi),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color='0.3'), zorder=2)

        # Final marker (solid for operational, red X for failed)
        if final_status[i] == 1:
            ax.scatter([x_end], [y_end], s=40, color='C1', zorder=4)
        else:
            ax.scatter([x_end], [y_end], s=90, marker='x', color='crimson', zorder=4)

    ax.set_title(f"Last Episode Overlay: Initial (hollow) → Final (solid/X)  |  t={final_round}")
    ax.set_xlim(-1, GRID_SIZE + 1)
    ax.set_ylim(-1, GRID_SIZE + 1)
    ax.set_xticks(range(0, GRID_SIZE+1, max(1, GRID_SIZE // 5)))
    ax.set_yticks(range(0, GRID_SIZE+1, max(1, GRID_SIZE // 5)))
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='C0', label='Initial (hollow)',
               markerfacecolor='none', markersize=8, lw=0, markeredgewidth=1.8),
        Line2D([0], [0], marker='o', color='C1', label='Final (operational)',
               markerfacecolor='C1', markersize=6, lw=0),
        Line2D([0], [0], marker='x', color='crimson', label='Final (failed)',
               markersize=9, lw=0),
        Line2D([0], [0], color='0.3', lw=1.6, label='Displacement (shortest torus)'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.show()

# -----------------------------
# Main: train, plot rolling TCR, plot overlay
# -----------------------------
if __name__ == "__main__":
    tcr_hist = []
    init_xy_last = init_status_last = final_xy_last = final_status_last = None
    final_round_last = None

    for ep in range(EPISODES):
        capture = (ep == EPISODES - 1)  # capture positions only on last episode
        tcr, init_xy, init_status, final_xy, final_status, final_round = run_episode(
            ep, capture_positions=capture
        )
        tcr_hist.append(tcr)

        if capture:
            init_xy_last, init_status_last = init_xy, init_status
            final_xy_last, final_status_last = final_xy, final_status
            final_round_last = final_round

        if (ep + 1) % 25 == 0:
            avg_25 = np.mean(tcr_hist[-25:])
            print(f"Ep {ep+1:4d} | Avg TCR (last 25): {avg_25:6.2f}%")

    # Plot ONLY the rolling average of TCR
    plot_tcr_rolling_only(tcr_hist, window=25)

    # Overlay of last episode
    if init_xy_last is None:
        print("No last-episode layouts captured.")
    else:
        plot_overlay_initial_final(init_xy_last, init_status_last,
                                   final_xy_last, final_status_last, final_round_last)
