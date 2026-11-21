import code
import numpy as np
import matplotlib.pyplot as plt

def simulate_fer(polar, p_values, num_frames=5000, max_frame_errors=100, seed=None):
    """
    Monte Carlo simulation of the SC decoder over a BSC.

    polar: PolarCode instance (fixed frozen set)
    p_values: BSC crossover probabilities to simulate (each must be in (0, 0.5])
    num_frames: max frames per p
    max_frame_errors: stop a point early once this many frame errors are seen
                      (keeps high-error points fast, low-error points accurate)
    Returns (fer, ber, frames_run) as numpy arrays aligned with p_values.
    """
    rng = np.random.default_rng(seed)
    fer, ber, frames_run = [], [], []

    for p in p_values:
        llr_mag = np.log((1.0 - p) / p)
        frame_errs = bit_errs = frames = 0

        while frames < num_frames and frame_errs < max_frame_errors:
            info_bits = rng.integers(0, 2, polar.K)
            _, x = polar.encode(info_bits)

            flips = (rng.random(polar.N) < p).astype(int)
            y = (x + flips) % 2
            llr = (1 - 2 * y) * llr_mag

            u_hat = polar.decode_sc(llr)
            n_err = int(np.sum(u_hat[polar.info_indices] != info_bits))

            frames += 1
            bit_errs += n_err
            frame_errs += n_err > 0

        fer.append(frame_errs / frames)
        ber.append(bit_errs / (frames * polar.K))
        frames_run.append(frames)

    return np.array(fer), np.array(ber), np.array(frames_run)


def plot_fer(polar, p_values=None, num_frames=5000, max_frame_errors=100,
             seed=None, ax=None):
    """
    Simulate and plot the frame error rate (FER) of the SC decoder vs the
    BSC crossover probability p. The x-axis runs from p = 0.5 on the left
    (useless channel) to p = 0 on the right (noiseless channel).
    Returns (p_values, fer).

    Points with zero observed frame errors are not drawn on the log axis
    (their FER is below 1/frames); raise num_frames to resolve them.
    """
    import matplotlib.pyplot as plt

    if p_values is None:
        p_values = np.linspace(0.5, 0.01, 17)
    p_values = np.asarray(p_values, dtype=float)

    fer, _, _ = simulate_fer(
        polar, p_values, num_frames, max_frame_errors, seed)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))

    mask = fer > 0
    ax.semilogy(p_values[mask], fer[mask], "o-")
    ax.set_xlim(0.5, 0)  # origin at 0.5, ends at 0 on the right

    ax.set_xlabel("BSC crossover probability p")
    ax.set_ylabel("Frame error rate (FER)")
    ax.set_title(f"SC decoder, N={polar.N}, K={polar.K} (rate {polar.K/polar.N:.2f})")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return p_values, fer


def plot_ber_rate_half(block_lengths=(2, 64, 256, 1024), p_values=None, p_design=0.1,
                       num_frames=2000, max_frame_errors=100, seed=None, ax=None):
    """
    Plot the bit error rate (BER) of the SC decoder vs BSC crossover
    probability p for several block lengths N, all at rate 1/2 (K = N/2).
    Log-log axes. The x-axis runs from p = 0.5 on the left toward small p on
    the right (p = 0 itself cannot be shown on a log axis).

    block_lengths: powers of 2 (default 2, 64, 256, 1024). Longer N gives a
                   sharper waterfall but is slower (~17 ms/frame at N=1024).
    p_design: crossover probability used to pick each code's frozen set
              (kept fixed across the sweep; ~0.1 suits rate 1/2)
    Returns {N: ber_array}.

    Points with zero observed bit errors are not drawn on the log axis.
    """
    import matplotlib.pyplot as plt

    if p_values is None:
        # coarse far from capacity (p ~ 0.11), dense around the waterfall
        p_values = np.array([0.5, 0.4, 0.3, 0.25, 0.2, 0.17, 0.15, 0.13, 0.12,
                             0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
                             0.03, 0.02, 0.01])
    p_values = np.asarray(p_values, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))

    results = {}
    for N in block_lengths:
        K = N // 2
        frozen = design_frozen_indices(N, K, p_design)
        polar = PolarCode(N, K, frozen)
        _, ber, _ = simulate_fer(polar, p_values, num_frames, max_frame_errors, seed)
        results[N] = ber

        mask = ber > 0
        ax.loglog(p_values[mask], ber[mask], "o-", label=f"N={N}, K={K}")

    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.set_xlim(0.5, p_values.min() / 1.25)  # reversed: 0.5 on the left
    ticks = [t for t in (0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001)
             if p_values.min() / 1.25 <= t <= 0.5]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("crossover probability p")
    ax.set_ylabel("Bit error rate (BER)")
    ax.set_title("SC decoder")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return results
