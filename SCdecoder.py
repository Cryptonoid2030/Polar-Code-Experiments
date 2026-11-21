import numpy as np
def sgn(val):
    if val == 0: return 1
    return abs(val) / val

# f function makes first channel estimate
def f_func_vec(L1, L2): # left child node (eq 3)

    L1 = np.clip(L1, -30, 30)
    L2 = np.clip(L2, -30, 30)
    # return 2 * np.arctanh(np.tanh(L1 / 2.0) * np.tanh(L2 / 2.0))
    # below in min approx for left node
    return [sgn(a) * sgn(b) * min(abs(a), abs(b)) for a, b in zip(L1, L2)]

# based on prediction v, apply guess for second bit
def g_func_vec(L1, L2, v): # right child node (eq 4)
    return [b + (1 - 2 * u_hat) * a for a, b, u_hat in zip(L1, L2, v)]

def polar_transform(u):
    if len(u) == 1:
        return u
    half = len(u) // 2
    left = polar_transform(u[:half])
    right = polar_transform(u[half:])
    return [l ^ r for l, r in zip(left, right)] + right

def recursive_sc(llrs, frozen_indices, start_idx=0):
    N = len(llrs)
    
    # BASE CASE: at leaf nodes, llrs is used to estimate bit
    if N == 1:
        if start_idx in frozen_indices:
            return [0]  # Frozen bit
        else:
            return [0] if llrs[0] > 0 else [1]  # Information bit

    # RECURSIVE CASE (N > 1)
    half_N = N // 2
    # split data into two trees
    L1 = llrs[:half_N]
    L2 = llrs[half_N:]
    
    # 1. Left Branch
    left_llrs = f_func_vec(L1, L2)
    u_left = recursive_sc(left_llrs, frozen_indices, start_idx)
    
    # 2. Partial Sums
    # The partial sums passed to the right branch are exactly the 
    # polar encoded version of the left decoded bits.
    v = polar_transform(u_left)
    
    # 3. Right Branch
    right_llrs = g_func_vec(L1, L2, v)
    u_right = recursive_sc(right_llrs, frozen_indices, start_idx + half_N)
    
    # Combine the decoded halves
    return u_left + u_right