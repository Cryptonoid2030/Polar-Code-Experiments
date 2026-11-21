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
    
    self.G = self.generate_matrix(N)

  def generate_matrix(self, N, generator=[[1, 0], [1, 1]]):
    F = np.array(generator, dtype=int)
    n = int(np.log2(N))
    G = F
    for _ in range(n - 1):
      G = np.kron(G, F)
    return G

  # apply generator matrix
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



if __name__ == "__main__":
    from frozen import *
    from performance import *

    # N = 8
    # K = 4

    N = 1024
    K = int(N/8) #roughly 50% code rate

    p_crossover = 0.4 # choose a crossover probability for this
    # Arbitrarily choosing some indices for frozen bits (usually chosen via channel capacity analysis)
    # frozen_indices = [0, 1, 2, 4] 
    frozen_indices = Bhattacharyya(N, K, p_crossover)
    
    polar = PolarCode(N, K, frozen_indices)
    
    # 1. Generate random info bits
    rng = np.random.default_rng()
    info_bits = rng.integers(0, 2, size=K)
    # info_bits = np.array([1, 0, 1, 1])
    print(f"Original Info Bits: {info_bits}")
    
    # 2. Encode
    polar.generate_matrix(N, [[1, 0], [1, 1]])
    u, x = polar.encode(info_bits)
    print(f"Full Input Vector u (with frozen bits): {u}")
    print(f"Transmitted Codeword x: {x}")
    
    # 3. Simulate transmission over AWGN channel (BPSK modulation: 0 -> +1, 1 -> -1)
    bpsk_signal = 1 - 2 * x.astype(float)
    sigma = 1
    noise = np.random.normal(0, sigma, N) # Add some Gaussian noise
    received_signal = bpsk_signal + noise
    
    # Convert received signal to Log-Likelihood Ratios (LLRs): LLR = 2 * y / sigma^2
    sigma_squared = sigma**2
    llr_input = (2 * received_signal) / sigma_squared
    print(f"llr:        {llr_input}")

    # 4. Decode
    decoded_u = polar.decode_sc(llr_input)
    extracted_info = decoded_u[polar.info_indices]
    print(f"Decoded Info Bits:     {extracted_info}")
    
    # 5. Verify success
    success = np.array_equal(info_bits, extracted_info)
    print(f"Decoding Successful?   {success}")
