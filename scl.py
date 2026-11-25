import numpy as np
import copy
import decodeHelper


def decode(y_llr, frozen_indices, N, L=4):
        """
        Iterative Successive Cancellation List (SCL) Decoder.
        Replaces the recursive SC decoder.
        
        y_llr: Received Log-Likelihood Ratios from the channel.
        L: Maximum list size (number of concurrent paths).
        Returns the estimated full input vector u_hat.
        """
        import copy
        n = int(np.log2(N))
        
        # Initialize the first (and only) active path
        # llrs: stores LLRs at each depth. Layer 0 is root (len N), Layer n is leaf (len 1)
        # bits: stores partial sums propagating back up.
        init_path = {
            'llrs': np.zeros((n + 1, N)),
            'bits': np.zeros((n + 1, N), dtype=int),
            'u_hat': np.zeros(N, dtype=int),
            'pm': 0.0  # Path Metric (lower is better)
        }
        # Load the channel LLRs into the root layer
        init_path['llrs'][0, :] = y_llr
        
        active_paths = [init_path]

        # Iterative decoding bit-by-bit from left to right
        for i in range(N):
            
            # ==========================================
            # 1. DOWNWARD PHASE: Calculate LLRs to leaf
            # ==========================================
            for d in range(n):
                layer_size = N // (2**d)
                half = layer_size // 2
                parent_offset = (i // layer_size) * layer_size
                
                # Only calculate when we enter a new left or right branch
                is_left_branch = (i % layer_size == 0)
                is_right_branch = (i % layer_size == half)
                
                if is_left_branch or is_right_branch:
                    for path in active_paths:
                        l1 = path['llrs'][d, parent_offset : parent_offset + half]
                        l2 = path['llrs'][d, parent_offset + half : parent_offset + layer_size]
                        
                        if is_left_branch:
                            # f-function (min-sum approximation)
                            path['llrs'][d+1, parent_offset : parent_offset + half] = decodeHelper.f(l1, l2)
                        elif is_right_branch:
                            # g-function
                            s_left = path['bits'][d+1, parent_offset : parent_offset + half]
                            path['llrs'][d+1, parent_offset + half : parent_offset + layer_size] = decodeHelper.g(l1, l2, s_left)

            # ==========================================
            # 2. LEAF NODE: Branching and Pruning
            # ==========================================
            new_paths = []
            for path in active_paths:
                leaf_llr = path['llrs'][n, i]
                expected_bit = 0 if leaf_llr >= 0 else 1
                
                if i in frozen_indices:
                    # Frozen bit: forced to 0
                    path['u_hat'][i] = 0
                    path['bits'][n, i] = 0
                    if expected_bit != 0:
                        path['pm'] += np.abs(leaf_llr)
                    new_paths.append(path)
                else:
                    # Information bit: Split into two paths (guess 0 and guess 1)
                    
                    # Path guessing 0
                    path0 = copy.deepcopy(path)
                    path0['u_hat'][i] = 0
                    path0['bits'][n, i] = 0
                    if expected_bit != 0:
                        path0['pm'] += np.abs(leaf_llr)
                    new_paths.append(path0)
                    
                    # Path guessing 1
                    path1 = copy.deepcopy(path) # Duplicate arrays
                    path1['u_hat'][i] = 1
                    path1['bits'][n, i] = 1
                    if expected_bit != 1:
                        path1['pm'] += np.abs(leaf_llr)
                    new_paths.append(path1)
            
            # Sort paths by Path Metric (lowest penalty is best) and prune to size L
            new_paths.sort(key=lambda x: x['pm'])
            active_paths = new_paths[:L]

            # ==========================================
            # 3. UPWARD PHASE: Propagate Partial Sums
            # ==========================================
            # Traverse from bottom layer back up to update partial sums
            for d in range(n - 1, -1, -1):
                layer_size = N // (2**d)
                half = layer_size // 2
                parent_offset = (i // layer_size) * layer_size
                
                # If we just finished processing the rightmost leaf of a node, 
                # we can compute its parent's partial sums.
                if i % layer_size == layer_size - 1:
                    for path in active_paths:
                        s_left = path['bits'][d+1, parent_offset : parent_offset + half]
                        s_right = path['bits'][d+1, parent_offset + half : parent_offset + layer_size]
                        
                        # Apply inverse kronecker transform modulo 2
                        path['bits'][d, parent_offset : parent_offset + half] = (s_left + s_right) % 2
                        path['bits'][d, parent_offset + half : parent_offset + layer_size] = s_right

        # ==========================================
        # 4. FINISH: Return best candidate
        # ==========================================
        best_path = active_paths[0]  # Already sorted, index 0 has the lowest PM
        return best_path['u_hat']