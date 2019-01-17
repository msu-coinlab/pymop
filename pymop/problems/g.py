import numpy as np
from pymop.problems import Problem
import math


class G1(Problem):
    def __init__(self):
        self.n_var = 13
        self.n_constr = 9
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = np.zeros(self.n_var)
        self.xu = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1])
        super(G1, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                 type_var=np.double)

    def _evaluate(self, x, f, g, *args, **kwargs):
        x1 = x[:, 0: 4]
        x2 = x[:, 4: 13]
        f[:, 0] = 5 * np.sum(x1, axis=1) - 5 * np.sum(np.multiply(x1, x1), axis=1) - np.sum(x2, axis=1)

        # Constraints
        g[:, 0] = 2 * x[:, 0] + 2 * x[:, 1] + x[:, 9] + x[:, 10] - 10
        g[:, 1] = 2 * x[:, 0] + 2 * x[:, 2] + x[:, 9] + x[:, 11] - 10
        g[:, 2] = 2 * x[:, 1] + 2 * x[:, 2] + x[:, 10] + x[:, 11] - 10
        g[:, 3] = -8 * x[:, 0] + x[:, 9]
        g[:, 4] = -8 * x[:, 1] + x[:, 10]
        g[:, 5] = -8 * x[:, 2] + x[:, 11]
        g[:, 6] = -2 * x[:, 3] - x[:, 4] + x[:, 9]
        g[:, 7] = -2 * x[:, 5] - x[:, 6] + x[:, 10]
        g[:, 8] = -2 * x[:, 7] - x[:, 8] + x[:, 11]

    def _calc_pareto_front(self):
        return -15

    def _calc_pareto_set(self):
        return np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1])


class G2(Problem):
    def __init__(self):
        self.n_var = 20
        self.n_constr = 2
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = np.zeros(self.n_var)
        self.xu = 10 * np.ones(self.n_var)
        super(G2, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                 type_var=np.double)

    def _evaluate(self, x, f, g, *args, **kwargs):
        sum_jx = np.zeros((x.shape[0], 1))
        for j in range(self.n_var):
            sum_jx[:, 0] = sum_jx[:, 0] + (j + 1) * x[:, j] ** 2

        a = np.sum(np.cos(x) ** 4, axis=1)
        b = 2 * np.prod(np.cos(x) ** 2, axis=1)
        c = (np.sqrt(sum_jx)).flatten()
        f[:, 0] = -np.absolute((a - b) / c)

        # Constraints
        g[:, 0] = -np.prod(x, 1) + 0.75
        g[:, 1] = np.sum(x, axis=1) - 7.5 * self.n_var

    def _calc_pareto_front(self):
        return -0.80361910412559

    def _calc_pareto_set(self):
        return np.array(
            [3.16246061572185, 3.12833142812967, 3.09479212988791, 3.06145059523469, 3.02792915885555, 2.99382606701730,
             2.95866871765285, 2.92184227312450, 0.49482511456933, 0.48835711005490, 0.48231642711865, 0.47664475092742,
             0.47129550835493, 0.46623099264167, 0.46142004984199, 0.45683664767217, 0.45245876903267, 0.44826762241853,
             0.44424700958760, 0.44038285956317])


class G3(Problem):

    def __init__(self):
        self.n_var = 10
        self.n_constr = 1
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)
        super(G3, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                 type_var=np.double)

    def _evaluate(self, x, f, g, *args, **kwargs):
        f[:, 0] = -np.sqrt(self.n_var) ** self.n_var * np.prod(x, axis=1)

        # Constraints
        g[:, 0] = np.absolute(np.sum(x ** 2, axis=1) - 1) - 1e-4

    def _calc_pareto_front(self):
        return -1.00050010001000

    def _calc_pareto_set(self):
        return np.array([0.31624357647283069,
                         0.316243577414338339, 0.316243578012345927, 0.316243575664017895, 0.316243578205526066,
                         0.31624357738855069, 0.316243575472949512, 0.316243577164883938, 0.316243578155920302,
                         0.316243576147374916])


class G4(Problem):

    def __init__(self):
        self.n_var = 5
        self.n_constr = 6
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = np.array([78, 33, 27, 27, 27])
        self.xu = np.array([102, 45, 45, 45, 45])
        super(G4, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                 type_var=np.double)

    def _evaluate(self, x, f, g, *args, **kwargs):
        f[:, 0] = 5.3578547 * x[:, 2] ** 2 + 0.8356891 * x[:, 0] * x[:, 4] + 37.293239 * x[:, 0] - 40792.141

        # Constraints
        u = 85.334407 + 0.0056858 * x[:, 1] * x[:, 4] + 0.0006262 * x[:, 0] * x[:, 3] - 0.0022053 * x[:, 2] * x[:, 4]
        g[:, 0] = -u
        g[:, 1] = u - 92
        v = 80.51249 + 0.0071317 * x[:, 1] * x[:, 4] + 0.0029955 * x[:, 0] * x[:, 1] + 0.0021813 * x[:, 2] ** 2
        g[:, 2] = -v + 90
        g[:, 3] = v - 110
        w = 9.300961 + 0.0047026 * x[:, 2] * x[:, 4] + 0.0012547 * x[:, 0] * x[:, 2] + 0.0019085 * x[:, 2] * x[:, 3]
        g[:, 4] = -w + 20
        g[:, 5] = w - 25

    def _calc_pareto_front(self):
        return -3.066553867178332 * (10 ** 4)

    def _calc_pareto_set(self):
        return np.array([78, 33, 29.9952560256815985, 45, 36.7758129057882073])


class G5(Problem):

    def __init__(self):
        self.n_var = 4
        self.n_constr = 5
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = np.array([0, 0, -0.55, -0.55])
        self.xu = np.array([1200, 1200, 0.55, 0.55])
        super(G5, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                 type_var=np.double)

    def _evaluate(self, x, f, g, *args, **kwargs):
        f[:, 0] = 3 * x[:, 0] + (10 ** -6) * x[:, 0] ** 3 + 2 * x[:, 1] + (2 * 10 ** (-6)) / 3 * x[:, 1] ** 3

        # Constraints
        g[:, 0] = x[:, 2] - x[:, 3] - 0.55
        g[:, 1] = x[:, 3] - x[:, 2] - 0.55

        g[:, 2] = np.absolute(1000 * (np.sin(-x[:, 2] - 0.25) + np.sin(-x[:, 3] - 0.25)) + 894.8 - x[:, 0]) - 10 ** (-4)
        g[:, 3] = np.absolute(
            1000 * (np.sin(x[:, 2] - 0.25) + np.sin(x[:, 2] - x[:, 3] - 0.25)) + 894.8 - x[:, 1]) - 10 ** (-4)
        g[:, 4] = np.absolute(1000 * (np.sin(x[:, 3] - 0.25) + np.sin(x[:, 3] - x[:, 2] - 0.25)) + 1294.8) - 10 ** (-4)

    def _calc_pareto_front(self):
        return 5126.4967140071

    def _calc_pareto_set(self):
        return np.array([679.945148297028709, 1026.06697600004691, 0.118876369094410433, -0.39623348521517826])


class G6(Problem):

    def __init__(self):
        self.n_var = 2
        self.n_constr = 2
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = np.array([13, 0])
        self.xu = np.array([100, 100])
        super(G6, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                 type_var=np.double)

    def _evaluate(self, x, f, g, *args, **kwargs):
        f[:, 0] = (x[:, 0] - 10) ** 3 + (x[:, 1] - 20) ** 3

        # Constraints
        g[:, 0] = -(x[:, 0] - 5) ** 2 - (x[:, 1] - 5) ** 2 + 100
        g[:, 1] = (x[:, 0] - 6) ** 2 + (x[:, 1] - 5) ** 2 - 82.81

    def _calc_pareto_front(self):
        return -6961.81387558015

    def _calc_pareto_set(self):
        return np.array([14.09500000000000064, 0.8429607892154795668])


class G7(Problem):

    def __init__(self):
        self.n_var = 10
        self.n_constr = 8
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = -10 * np.ones(self.n_var)
        self.xu = 10 * np.ones(self.n_var)
        super(G7, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                 type_var=np.double)

    def _evaluate(self, x, f, g, *args, **kwargs):
        f[:, 0] = x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 0] * x[:, 1] - 14 * x[:, 0] - 16 * x[:, 1] + (x[:, 2] - 10) ** 2 \
                  + 4 * (x[:, 3] - 5) ** 2 + (x[:, 4] - 3) ** 2 + 2 * (x[:, 5] - 1) ** 2 + 5 * x[:, 6] ** 2 \
                  + 7 * (x[:, 7] - 11) ** 2 + 2 * (x[:, 8] - 10) ** 2 + (x[:, 9] - 7) ** 2 + 45

        # Constraints
        g[:, 0] = 4 * x[:, 0] + 5 * x[:, 1] - 3 * x[:, 6] + 9 * x[:, 7] - 105
        g[:, 1] = 10 * x[:, 0] - 8 * x[:, 1] - 17 * x[:, 6] + 2 * x[:, 7]
        g[:, 2] = -8 * x[:, 0] + 2 * x[:, 1] + 5 * x[:, 8] - 2 * x[:, 9] - 12
        g[:, 3] = 3 * (x[:, 0] - 2) ** 2 + 4 * (x[:, 1] - 3) ** 2 + 2 * x[:, 2] ** 2 - 7 * x[:, 3] - 120
        g[:, 4] = 5 * x[:, 0] ** 2 + 8 * x[:, 1] + (x[:, 2] - 6) ** 2 - 2 * x[:, 3] - 40
        g[:, 5] = 0.5 * (x[:, 0] - 8) ** 2 + 2 * (x[:, 1] - 4) ** 2 + 3 * x[:, 4] ** 2 - x[:, 5] - 30
        g[:, 6] = x[:, 0] ** 2 + 2 * (x[:, 1] - 2) ** 2 - 2 * x[:, 0] * x[:, 1] + 14 * x[:, 4] - 6 * x[:, 5]
        g[:, 7] = -3 * x[:, 0] + 6 * x[:, 1] + 12 * (x[:, 8] - 8) ** 2 - 7 * x[:, 9]

    def _calc_pareto_front(self):
        return 24.30620906818

    def _calc_pareto_set(self):
        return np.array([2.17199634142692, 2.3636830416034,
                         8.77392573913157, 5.09598443745173, 0.990654756560493, 1.43057392853463, 1.32164415364306,
                         9.82872576524495, 8.2800915887356, 8.3759266477347])


class G8(Problem):

    def __init__(self):
        self.n_var = 2
        self.n_constr = 2
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = np.zeros(self.n_var)
        self.xu = 10 * np.ones(self.n_var)
        super(G8, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                 type_var=np.double)

    def _evaluate(self, x, f, g, *args, **kwargs):
        f[:, 0] = -(np.sin(2 * math.pi * x[:, 0]) ** 3 * np.sin(2 * math.pi * x[:, 1])) / (
                    x[:, 0] ** 3 * (x[:, 0] + x[:, 1]))

        # Constraints
        g[:, 0] = x[:, 0] ** 2 - x[:, 1] + 1
        g[:, 1] = 1 - x[:, 0] + (x[:, 1] - 4) ** 2

    def _calc_pareto_front(self):
        return -0.0958250414180359

    def _calc_pareto_set(self):
        return np.array([1.22797135260752599, 4.24537336612274885])


class G9(Problem):

    def __init__(self):
        self.n_var = 7
        self.n_constr = 4
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = -10 * np.zeros(self.n_var)
        self.xu = 10 * np.ones(self.n_var)
        super(G9, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                 type_var=np.double)

    def _evaluate(self, x, f, g, *args, **kwargs):
        f[:, 0] = (x[:, 0] - 10) ** 2 + 5 * (x[:, 1] - 12) ** 2 + x[:, 2] ** 4 \
                  + 3 * (x[:, 3] - 11) ** 2 + 10 * x[:, 4] ** 6 + 7 * x[:, 5] ** 2 \
                  + x[:, 6] ** 4 - 4 * x[:, 5] * x[:, 6] - 10 * x[:, 5] - 8 * x[:, 6]

        # Constraints
        v1 = 2 * x[:, 0] ** 2
        v2 = x[:, 1] ** 2
        g[:, 0] = v1 + 3 * v2 ** 2 + x[:, 2] + 4 * x[:, 3] ** 2 + 5 * x[:, 4] - 127
        g[:, 1] = 7 * x[:, 0] + 3 * x[:, 1] + 10 * x[:, 2] ** 2 + x[:, 3] - x[:, 4] - 282
        g[:, 2] = 23 * x[:, 0] + v2 + 6 * x[:, 5] ** 2 - 8 * x[:, 6] - 196
        g[:, 3] = 2 * v1 + v2 - 3 * x[:, 0] * x[:, 1] + 2 * x[:, 2] ** 2 + 5. * x[:, 5] - 11 * x[:, 6]

    def _calc_pareto_front(self):
        return 680.630057374402

    def _calc_pareto_set(self):
        return np.array([2.33049935147405174, 1.95137236847114592, -0.477541399510615805,
                         4.36572624923625874, -0.624486959100388983, 1.03813099410962173, 1.5942266780671519])


class G10(Problem):

    def __init__(self):
        self.n_var = 8
        self.n_constr = 6
        self.n_obj = 1
        self.func = self._evaluate
        self.xl = np.array([100, 1000, 1000, 10, 10, 10, 10, 10])
        self.xu = np.array([10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000])
        super(G10, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                  type_var=np.double)

    def _evaluate(self, x, f, g, *args, **kwargs):
        f[:, 0] = x[:, 0] + x[:, 1] + x[:, 2]

        # Constraints
        g[:, 0] = -1 + 0.0025 * (x[:, 3] + x[:, 5])
        g[:, 1] = -1 + 0.0025 * (-x[:, 3] + x[:, 4] + x[:, 6])
        g[:, 2] = -1 + 0.01 * (-x[:, 4] + x[:, 7])
        g[:, 3] = 100 * x[:, 0] - x[:, 0] * x[:, 5] + 833.33252 * x[:, 3] - 83333.333
        g[:, 4] = x[:, 1] * x[:, 3] - x[:, 1] * x[:, 6] - 1250 * x[:, 3] + 1250 * x[:, 4]
        g[:, 5] = x[:, 2] * x[:, 4] - x[:, 2] * x[:, 7] - 2500. * x[:, 4] + 1250000

    def _calc_pareto_front(self):
        return 7049.24802052867

    def _calc_pareto_set(self):
        return np.array(
            [579.306685017979589, 1359.97067807935605, 5109.97065743133317, 182.01769963061534, 295.601173702746792,
             217.982300369384632, 286.41652592786852, 395.601173702746735])
