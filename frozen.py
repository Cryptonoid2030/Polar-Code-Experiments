import numpy as np

def Bhattacharyya(N, K, p_design, return_z=False):
    """
    Pick the N total bits with K send bits (Bhattacharyya bound) for a BSC
    with crossover probability p_design.
    """
    z = np.array([2.0 * np.sqrt(p_design * (1.0 - p_design))])
    while len(z) < N:
        y = z - z**2
        new_z = np.empty(2 * len(z))
        new_z[0::2] = np.clip(z + y, 0, 1)  # degraded subchannel
        new_z[1::2] = z - y                 # upgraded subchannel
        z = new_z
        
    ranked = np.argsort(z)
    frozen_indices = np.sort(ranked[K:])
    
    if return_z:
        return frozen_indices, z
    return frozen_indices

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 1024
    K = int(N / 8)
    p = 0.3

    channels, z_values = Bhattacharyya(N, K, p, return_z=True)
    
    reliabilities = 1.0 - z_values
    
    print("Channel Reliabilities (Index 0 to 1023):")
    for i in range(10): 
        print(f"Channel {i}: {reliabilities[i]:.4f} (Z = {z_values[i]:.4f})")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5)) 

    ax[0].plot(reliabilities, marker='.', linestyle='none')

    ax[0].set_title("Polarization of Sub-Channels")
    ax[0].set_xlabel("Channel Index")
    ax[0].set_ylabel("Approximate Reliability (1 - Z)")
    ax[0].grid(True)

    ax[1].hist(reliabilities, bins=int(np.sqrt(len(reliabilities))))
    ax[1].set_title("Distribution of Reliabilities")
    ax[1].set_xlabel("Approximate Reliability")
    ax[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()