import numpy as np
from scipy import special



# returns the closest possible number of references lines to given one
def get_uniform_weights(n_dim, n_points, max_sections=100):

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

    # Generates uniform distrbution of reference points
    n_sections = np.array([get_number_of_reference_directions(n_dim, i) for i in range(max_sections)])
    idx = np.argmin((n_sections < n_points).astype(np.int))
    M = get_ref_dirs_from_section(n_dim, idx-1)
    M[M == 0] = 0.000001

    return M
