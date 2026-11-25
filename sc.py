import numpy as np
import decodeHelper

def decode(y_llr, frozen_indices, N):
    """
    Successive Cancellation (SC) Decoder with correct global offset tracking.
    y_llr: Received Log-Likelihood Ratios from the channel.
    Returns the estimated full input vector u_hat.
    """
    def _sc_recursive(llr, offset, length):
        # Base case: single bit
        if length == 1:
            idx = offset
            if idx in frozen_indices:
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
        left_llr = decodeHelper.f(l1, l2)
        u_left, s_left = _sc_recursive(left_llr, offset, half)
        
        # Right branch LLR calculation (g function using partial sums s_left)
        # s_left is applied here to predict (better channel)
        right_llr = decodeHelper.g(l1, l2, s_left)
        u_right, s_right = _sc_recursive(right_llr, offset + half, half)
        
        # Combine both halves
        u_hat = np.concatenate([u_left, u_right])
        s_hat = np.concatenate([(s_left + s_right) % 2, s_right])
        return u_hat, s_hat

    # Start recursion from the root (offset 0, full length N)
    u_hat, _ = _sc_recursive(y_llr, 0, N)
    return u_hat