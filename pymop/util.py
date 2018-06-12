import numpy as np
from scipy import special


def get_weights(n_points, n_dim, func_random=np.random.rand, method="uniform", **kwargs):
    if method == "dirichlet":
        M = dirichlet(n_points, n_dim, func_random, **kwargs)
    elif method == "uniform":
        M = get_uniform(n_points, n_dim, func_random, **kwargs)
    M[M == 0] = 1e-30
    return M


# https://stats.stackexchange.com/questions/14059/generate-uniformly-distributed-weights-that-sum-to-unity?
# utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
def dirichlet(n_points, n_dim, func_random):
    y = -np.log(func_random(n_points - n_dim, n_dim))
    w = y / np.sum(y, axis=1)[:, None]
    return np.concatenate([np.eye(n_dim), w], axis=0)


# returns the closest possible number of references lines to given one
def get_uniform(n_points, n_dim, func_random, fill_with_random=True, max_sections=300):
    def get_ref_dirs_from_section(n_obj, n_sections):

        if n_obj == 1:
            return np.array([1.0])

        # all possible values for the vector
        sections = np.linspace(0, 1, num=n_sections + 1)[::-1]

        ref_dirs = []
        ref_recursive([], sections, 0, n_obj, ref_dirs)
        return np.array(ref_dirs)

    def ref_recursive(v, sections, level, max_level, result):
        v_sum = np.sum(np.array(v))

        # sum slightly above or below because of numerical issues
        if v_sum > 1.0001:
            return
        elif level == max_level:
            if 1.0 - v_sum < 0.0001:
                result.append(v)
        else:
            for e in sections:
                next = list(v)
                next.append(e)
                ref_recursive(next, sections, level + 1, max_level, result)

    def get_number_of_reference_directions(n_obj, n_sections):
        return int(special.binom(n_obj + n_sections - 1, n_sections))

    # Generates uniform distribution of reference points
    n_sections = np.array([get_number_of_reference_directions(n_dim, i) for i in range(max_sections)])
    idx = np.argmin((n_sections < n_points).astype(np.int))
    M = get_ref_dirs_from_section(n_dim, idx - 1)

    if fill_with_random and M.shape[0] < n_points:
        M = np.concatenate([M, func_random(n_points - M.shape[0], n_dim)], axis=0)

    return M


if __name__ == "__main__":
    w = get_weights(300, 3, np.random.rand, method="section")

    import matplotlib.pyplot as plt

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w[:, 0], w[:, 1], w[:, 2])
    plt.show()
