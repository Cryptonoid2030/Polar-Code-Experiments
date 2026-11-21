import numpy as np

class PolarCode:
  def __init__(self, N, K, frozen_indices):
    """
    N: Block length (must be a power of 2)
    K: Number of information bits
    frozen_indices: List of indices (0 to N-1) designated as frozen bits
    """
    self.N = N
    self.K = K
    self.frozen_indices = set(frozen_indices)
    self.info_indices = [i for i in range(N) if i not in self.frozen_indices]
    
    self.G = self._build_generator_matrix(N)

  def _build_generator_matrix(self, N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    n = int(np.log2(N))
    G = F
    for _ in range(n - 1):
      G = np.kron(G, F)
    return G

  def encode(self, u_info):
    u = np.zeros(self.N, dtype=int)
    u[self.info_indices] = u_info

    x = np.dot(u, self.G) % 2

    return u, x

  def decode_sc(self, y_llr):
    """
    Successive Cancellation (SC) Decoder with correct global offset tracking.
    y_llr: Received Log-Likelihood Ratios from the channel.
    Returns the estimated full input vector u_hat.
    """
    def _sc_recursive(llr, offset, length):
        # Base case: single bit
        if length == 1:
            idx = offset
            if idx in self.frozen_indices:
                val = 0  # Frozen bits are always known to be 0
            else:
                # Hard decision: LLR >= 0 -> bit 0, LLR < 0 -> bit 1
                val = 0 if llr[0] >= 0 else 1
            return np.array([val], dtype=int), np.array([val], dtype=int)
        
        half = length // 2
        l1 = llr[:half]
        l2 = llr[half:]
        
        # Left branch LLR calculation (f function)
        # here, the min function is applied to find the worst case - least certain channel 
        # the sgn function computes the xor of bit 1 and bit 2; representing the G matrix input
        left_llr = np.sign(l1) * np.sign(l2) * np.minimum(np.abs(l1), np.abs(l2))  
        u_left, s_left = _sc_recursive(left_llr, offset, half)
        
        # Right branch LLR calculation (g function using partial sums s_left)
        # s_left is applied here to predict (better channel)
        right_llr = l2 + (1 - 2 * s_left) * l1
        u_right, s_right = _sc_recursive(right_llr, offset + half, half)
        
        # Combine both halves
        u_hat = np.concatenate([u_left, u_right])
        s_hat = np.concatenate([(s_left + s_right) % 2, s_right])
        return u_hat, s_hat

    # Start recursion from the root (offset 0, full length N)
    u_hat, _ = _sc_recursive(y_llr, 0, self.N)
    return u_hat

def design_frozen_indices(N, K, p_design):
    """
    Pick the N-K least reliable subchannels (Bhattacharyya bound) for a BSC
    with crossover probability p_design. Use ONE fixed p_design for a whole
    FER sweep so the code does not change from point to point.
    """
    z = np.array([2.0 * np.sqrt(p_design * (1.0 - p_design))])
    while len(z) < N:
        y = z - z**2
        new_z = np.empty(2 * len(z))
        new_z[0::2] = np.clip(z + y, 0, 1)  # degraded subchannel
        new_z[1::2] = z - y                 # upgraded subchannel
        z = new_z
    ranked = np.argsort(z)
    return np.sort(ranked[K:])


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


if __name__ == "__main__":
    # N = 8
    # K = 4

    N = 1024
    K = int(N/8) #roughly 50% code rate

    p_crossover = 0.9 # choose a crossover probability for this
    # Arbitrarily choosing some indices for frozen bits (usually chosen via channel capacity analysis)
    # frozen_indices = [0, 1, 2, 4] 
    frozen_indices = design_frozen_indices(N, K, p_crossover)
    
    polar = PolarCode(N, K, frozen_indices)
    
    # 1. Generate random info bits
    rng = np.random.default_rng()
    info_bits = rng.integers(0, 2, size=K)
    # info_bits = np.array([1, 0, 1, 1])
    print(f"Original Info Bits: {info_bits}")
    
    # 2. Encode
    u, x = polar.encode(info_bits)
    print(f"Full Input Vector u (with frozen bits): {u}")
    print(f"Transmitted Codeword x: {x}")
    
    # 3. Simulate transmission over AWGN channel (BPSK modulation: 0 -> +1, 1 -> -1)
    bpsk_signal = 1 - 2 * x.astype(float)
    noise = np.random.normal(0, 0.5, N) # Add some Gaussian noise
    received_signal = bpsk_signal + noise
    
    # Convert received signal to Log-Likelihood Ratios (LLRs): LLR = 2 * y / sigma^2
    sigma_squared = 0.5**2
    llr_input = (2 * received_signal) / sigma_squared

    # 4. Decode
    decoded_u = polar.decode_sc(llr_input)
    extracted_info = decoded_u[polar.info_indices]
    print(f"Decoded Info Bits:     {extracted_info}")
    
    # 5. Verify success
    success = np.array_equal(info_bits, extracted_info)
    print(f"Decoding Successful?   {success}")
