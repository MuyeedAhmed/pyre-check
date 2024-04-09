# static_analysis_example/source.py

import os
# import from sklearn.cluster import AffinityPropagation
import numpy as np



# def get_image(url):
#     command = "wget -q https:{}".format(url)
#     return os.system(command)

# def convert():
#     image_link = input("image link: ")
#     image = get_image(image_link)
# @pyre_dump_call_graph
def get_label():
    # pyre_dump_call_graph()
    # X = input("")

    # def input(__prompt) -> TaintSource[UserControlled]: ...

    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    # clustering = AffinityPropagation(random_state=5).fit(X)
    # c = clustering.labels_
    # reveal_taint(X)
    c = _affinity_propagation(X)
    # result = eval(f"c[2] * c[1]")
    return os.system(c)
    
def _affinity_propagation(
    S,
    preference=None,
    convergence_iter=15,
    max_iter=200,
    damping=0.5,
    copy=True,
    verbose=False,
    return_n_iter=False,
    random_state=None,
):
    pyre_dump_call_graph()
    """Main affinity propagation algorithm."""
    n_samples = S.shape[0]

    # Place preference on the diagonal of S
    S.flat[:: (n_samples + 1)] = preference

    A = np.zeros((n_samples, n_samples))
    R = np.zeros((n_samples, n_samples))  # Initialize messages
    # Intermediate results
    tmp = np.zeros((n_samples, n_samples))

    # Remove degeneracies
    S += (
        np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100
    ) * random_state.standard_normal(size=(n_samples, n_samples))
    
    # reveal_taint(S)

    # Execute parallel affinity propagation updates
    e = np.zeros((n_samples, convergence_iter))

    ind = np.arange(n_samples)

    for it in range(max_iter):
        # tmp = A + S; compute responsibilities
        np.add(A, S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)

        # tmp = Rnew
        np.subtract(S, Y[:, None], tmp)
        tmp[ind, I] = S[ind, I] - Y2

        # Damping
        tmp *= 1 - damping
        R *= damping
        R += tmp

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp.flat[:: n_samples + 1] = R.flat[:: n_samples + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[:: n_samples + 1] = dA

        # Damping
        tmp *= 1 - damping
        A *= damping
        A -= tmp

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)

        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            unconverged = np.sum((se == convergence_iter) + (se == 0)) != n_samples
            if (not unconverged and (K > 0)) or (it == max_iter):
                never_converged = False
                if verbose:
                    print("Converged after %d iterations." % it)
                break
    else:
        never_converged = True
        if verbose:
            print("Did not converge")

    I = np.flatnonzero(E)
    K = I.size  # Identify exemplars

    if K > 0:
        
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)  # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        
        labels = np.array([-1] * n_samples)
        cluster_centers_indices = []

    if return_n_iter:
        return cluster_centers_indices, labels, it + 1
    else:
        return cluster_centers_indices, labels