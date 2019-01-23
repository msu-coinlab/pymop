import autograd.numpy as np
from autograd.core import VJPNode, vspace, backward_pass
from autograd.tracer import trace_stack, new_box, isbox


def calc_and_trace(fun, x, *args, **kwargs):
    # create the starting node
    start_node = VJPNode.new_root(x)

    # trace the calculations
    with trace_stack.new_trace() as t:
        start_box = new_box(x, t, start_node)
        fun(start_box, *args, **kwargs)
    return start_box


def calc_jacobian(start_box, end_box, x):
    # check if the graph makes sense for derivation
    if isbox(end_box) and end_box._trace == start_box._trace:
        ans, end_node = end_box._value, end_box._node
    else:
        #warnings.warn("Output seems independent of input.")
        ans, end_node = end_box, None

    if end_node is None:
        def vjp(g):
            return vspace(x).zeros()
    else:
        def vjp(g):
            return backward_pass(g, end_node)

    # initialize the jacobian matrix
    jac = np.full((x.shape[0], ans.shape[1], x.shape[1]), np.nan)

    for j in range(ans.shape[1]):
        m = np.zeros(ans.shape)
        m[:, j] = 1
        jac[:, j, :] = vjp(m)

    return jac