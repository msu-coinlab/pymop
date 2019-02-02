import autograd.numpy as np
from autograd.core import VJPNode, vspace, backward_pass
from autograd.tracer import new_box, isbox


def calc_and_trace(fun, x, *args, **kwargs):
    # create the starting node
    start_node = VJPNode.new_root(x)

    #with trace_stack.new_trace() as t:
    start_box = new_box(x, 0, start_node)
    out = fun(start_box, *args, **kwargs)

    return start_box, out


def calc_jacobian(start_box, end_box):
    # check if the graph makes sense for derivation
    if isbox(end_box):# and end_box._trace == start_box._trace:
        ans, end_node = end_box._value, end_box._node
    else:
        # warnings.warn("Output seems independent of input.")
        ans, end_node = end_box, None

    if end_node is None:
        def vjp(g):
            return vspace(start_box.shape).zeros()
    else:
        def vjp(g):
            return backward_pass(g, end_node)

    jac = []

    for j in range(ans.shape[1]):
        m = np.zeros(ans.shape)
        m[:, j] = 1

        _jac = vjp(new_box(m, 0, VJPNode.new_root(m)))

        m = np.zeros(_jac.shape)
        m[0,0] = 1

        test = backward_pass(m, end_node)

        #root, _jac = calc_and_trace(vjp, m)

        jac.append(_jac)


    jac = np.stack(jac, axis=1)

    return jac


def calc_hessian(start_box, end_box):
    # check if the graph makes sense for derivation
    if isbox(end_box): # and end_box._trace == start_box._trace:
        ans, end_node = end_box._value, end_box._node
    else:
        # warnings.warn("Output seems independent of input.")
        ans, end_node = end_box, None

    if end_node is None:
        def vjp(g):
            return vspace(start_box.shape).zeros()
    else:
        def vjp(g):
            return backward_pass(g, end_node)

    hessian = []

    # iterate over each function
    for i in range(ans.shape[1]):

        hessian_of_func = []

        # iterate over each variable
        for j in range(ans.shape[2]):
            m = np.zeros(ans.shape)
            m[:, i, j] = 1

            _hessian = vjp(m)

            hessian_of_func.append(_hessian)

        hessian_of_func = np.stack(hessian_of_func, axis=1)
        hessian.append(hessian_of_func)

    hessian = np.stack(hessian, axis=1)

    return hessian
