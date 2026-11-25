import numpy as np
import decodeHelper
import sc
import scl


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

  # decoders are implemented here
  def decode_sc(self, y_llr):
     return sc.decode(y_llr, self.frozen_indices, self.N)
  
  def decode_scl(self, y_llr, L):
     return scl.decode(y_llr, self.frozen_indices, self.N, L)



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
    print(f"Decoded Info Bits (SC):     {extracted_info}")

    decoded_u = polar.decode_scl(llr_input, L=4)
    extracted_info = decoded_u[polar.info_indices]
    print(f"Decoded Info Bits (SCL):     {extracted_info}")
    
    # 5. Verify success
    success = np.array_equal(info_bits, extracted_info)
    print(f"Decoding Successful?   {success}")
