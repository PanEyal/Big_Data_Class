import numpy as np

if __name__ == '__main__':

    # a.
    d = 4
    k = 2
    X = np.matrix([[1,-2,5,4],
                   [3,2,1,-5],
                   [-10,1,-4,6]])

    A = X.transpose() @ X

    eig_vals, eig_vecs = np.linalg.eig(A)

    zipped_eig = zip(eig_vals, eig_vecs.transpose())
    sorted_zipped_eigs = np.asarray(sorted(zipped_eig, key = lambda zipped_eig: zipped_eig[0]))
    eig_vals = sorted_zipped_eigs[:, 0]
    eig_vecs = sorted_zipped_eigs[:, 1]
    distortion = np.sum(eig_vals[:(d-k)])

    print(f"The distortion is: {distortion}")


    # b.
    u_trans = np.flip(np.stack(eig_vecs[(d-k):]), axis=0)

    for eig_vec in range(k):
        norm = np.linalg.norm(u_trans[eig_vec, :])
        if norm != 0.:
            u_trans[eig_vec, :] = u_trans[eig_vec, :] / norm

    print("\nU^T is:")
    print(u_trans)
    print(f"U^T is orthogonal? {np.isclose(np.dot(u_trans[0,:], u_trans[1,:].transpose()), 0)}")

    restored_X = np.transpose(u_trans.transpose() @ u_trans @ X.transpose())

    print("\nrestored_X is:")
    print(restored_X)

    distortion = 0.
    for i in range(len(X)):
        distortion += pow(np.linalg.norm(X[i] - restored_X[i]), 2)

    print(f"\nThe distortion after restore is: {distortion}")
