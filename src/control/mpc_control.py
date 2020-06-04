# https://apmonitor.com/wiki/index.php/Main/GekkoPythonOptimization
# https://www.youtube.com/watch?v=Gh8R4PVg1Zc&t=70s

# https://medium.com/@jonathan_hui/self-driving-car-path-planning-to-maneuver-the-traffic-ac63f5a620e2
# https://medium.com/@jonathan_hui/lane-keeping-in-autonomous-driving-with-model-predictive-control-50f06e989bc9

# https://arxiv.org/pdf/1805.08551.pdf
# https://sci-hub.tw/https://www.sciencedirect.com/science/article/pii/S2405896319303647

# Read implementation , try to apply gekko in order to optimize

# Implementation in torch https://locuslab.github.io/mpc.pytorch/

#THIS IS WITH GEKKO
import random

import carla
import numpy as np

from scipy.optimize import minimize
import sympy as sym
from sympy.tensor.array import derive_by_array
sym.init_printing()

#FIX RELATIVE IMPORTS
from control.abstract_control import Controller
from config import STEER_BOUNDS, THROTTLE_BOUNDS
from spawn import numpy_to_transform, transform_to_numpy, velocity_to_kmh


class _EqualityConstraints(object):
    """Class for storing equality constraints in the MPC."""

    def __init__(self, N, state_vars):
        self.dict = {}
        for symbol in state_vars:
            self.dict[symbol] = N*[None]

    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value


class MPCController(Controller):
    def __init__(self, target_speed, steps_ahead=10, dt=0.1):
        self.target_speed = target_speed
        self.state_vars = ('x', 'y', 'v', 'ψ', 'cte', 'eψ')

        self.steps_ahead = steps_ahead
        self.dt = dt

        # Cost function coefficients
        self.cte_coeff = 100 # 100
        self.epsi_coeff = 100 # 100
        self.speed_coeff = 0.4  # 0.2
        self.acc_coeff = 1  # 1
        self.steer_coeff = 0.1  # 0.1
        self.consec_acc_coeff = 50
        self.consec_steer_coeff = 50

        # Front wheel L
        self.Lf = 2.5

        # How the polynomial fitting the desired curve is fitted
        self.steps_poly = 30
        self.poly_degree = 3

        # Bounds for the optimizer
        self.bounds = (
            6*self.steps_ahead * [(None, None)]
            + self.steps_ahead * [THROTTLE_BOUNDS]
            + self.steps_ahead * [STEER_BOUNDS]
        )

        # State 0 placeholder
        num_vars = (len(self.state_vars) + 2)  # State variables and two actuators
        self.state0 = np.zeros(self.steps_ahead*num_vars)

        # Lambdify and minimize stuff
        self.evaluator = 'numpy'
        self.tolerance = 1
        self.cost_func, self.cost_grad_func, self.constr_funcs = self.get_func_constraints_and_bounds()

        # To keep the previous state
        self.steer = None
        self.throttle = None

    @property
    def name(self) -> str:
        return f'{self.__class__.__name__}_ts{self.target_speed}_sa{self.steps_ahead}_dt{self.dt}'

    def get_func_constraints_and_bounds(self):
        """The most important method of this class, defining the MPC's cost
        function and constraints.
        """
        # Polynomial coefficients will also be symbolic variables
        poly = self.create_array_of_symbols('poly', self.poly_degree+1)

        # Initialize the initial state
        x_init = sym.symbols('x_init')
        y_init = sym.symbols('y_init')
        ψ_init = sym.symbols('ψ_init')
        v_init = sym.symbols('v_init')
        cte_init = sym.symbols('cte_init')
        eψ_init = sym.symbols('eψ_init')

        init = (x_init, y_init, ψ_init, v_init, cte_init, eψ_init)

        # State variables
        x = self.create_array_of_symbols('x', self.steps_ahead)
        y = self.create_array_of_symbols('y', self.steps_ahead)
        ψ = self.create_array_of_symbols('ψ', self.steps_ahead)
        v = self.create_array_of_symbols('v', self.steps_ahead)
        cte = self.create_array_of_symbols('cte', self.steps_ahead)
        eψ = self.create_array_of_symbols('eψ', self.steps_ahead)

        # Actuators
        a = self.create_array_of_symbols('a', self.steps_ahead)
        δ = self.create_array_of_symbols('δ', self.steps_ahead)

        vars_ = (
            # Symbolic arrays (but NOT actuators)
            *x, *y, *ψ, *v, *cte, *eψ,

            # Symbolic arrays (actuators)
            *a, *δ,
        )

        cost = 0
        for t in range(self.steps_ahead):
            cost += (
                # Reference state penalties
                self.cte_coeff * cte[t]**2
                + self.epsi_coeff * eψ[t]**2 +
                + self.speed_coeff * (v[t] - self.target_speed)**2

                # # Actuator penalties
                + self.acc_coeff * a[t]**2
                + self.steer_coeff * δ[t]**2
            )

        # Penalty for differences in consecutive actuators
        for t in range(self.steps_ahead-1):
            cost += (
                self.consec_acc_coeff * (a[t+1] - a[t])**2
                + self.consec_steer_coeff * (δ[t+1] - δ[t])**2
            )

        # Initialize constraints
        eq_constr = _EqualityConstraints(self.steps_ahead, self.state_vars)
        eq_constr['x'][0] = x[0] - x_init
        eq_constr['y'][0] = y[0] - y_init
        eq_constr['ψ'][0] = ψ[0] - ψ_init
        eq_constr['v'][0] = v[0] - v_init
        eq_constr['cte'][0] = cte[0] - cte_init
        eq_constr['eψ'][0] = eψ[0] - eψ_init

        for t in range(1, self.steps_ahead):
            curve = sum(poly[-(i+1)] * x[t-1]**i for i in range(len(poly)))
            # The desired ψ is equal to the derivative of the polynomial curve at
            #  point x[t-1]
            ψdes = sum(poly[-(i+1)] * i*x[t-1]**(i-1) for i in range(1, len(poly)))

            eq_constr['x'][t] = x[t] - (x[t-1] + v[t-1] * sym.cos(ψ[t-1]) * self.dt)
            eq_constr['y'][t] = y[t] - (y[t-1] + v[t-1] * sym.sin(ψ[t-1]) * self.dt)
            eq_constr['ψ'][t] = ψ[t] - (ψ[t-1] - v[t-1] * δ[t-1] / self.Lf * self.dt)
            eq_constr['v'][t] = v[t] - (v[t-1] + a[t-1] * self.dt)
            eq_constr['cte'][t] = cte[t] - (curve - y[t-1] + v[t-1] * sym.sin(eψ[t-1]) * self.dt)
            eq_constr['eψ'][t] = eψ[t] - (ψ[t-1] - ψdes - v[t-1] * δ[t-1] / self.Lf * self.dt)

        # Generate actual functions from
        cost_func = self.generate_fun(cost, vars_, init, poly)
        cost_grad_func = self.generate_grad(cost, vars_, init, poly)

        constr_funcs = []
        for symbol in self.state_vars:
            for t in range(self.steps_ahead):
                func = self.generate_fun(eq_constr[symbol][t], vars_, init, poly)
                grad_func = self.generate_grad(eq_constr[symbol][t], vars_, init, poly)
                constr_funcs.append(
                    {'type': 'eq', 'fun': func, 'jac': grad_func, 'args': None},
                )

        return cost_func, cost_grad_func, constr_funcs

    def control(self, state, **kwargs):

        pts_3D = kwargs['pts_3D']
        which_closest, _, location = self._calc_closest_dists_and_location(
        # which_closest, _, location = _calc_closest_dists_and_location(
            state['location'], #without yaws
            pts_3D
        )

        # Stabilizes polynomial fitting
        which_closest_shifted = which_closest - 5
        # NOTE: `which_closest_shifted` might become < 0, but the modulo operation below fixes that

        indeces = which_closest_shifted + self.steps_poly*np.arange(self.poly_degree+1)
        indeces = indeces % pts_3D.shape[0]
        pts = pts_3D[:,:2][indeces] #converting to 2d points

        v = state['velocity'] # km / h #how to get speed?????????
        ψ = np.radians(state['yaw']) #adding 180 as carla returns yaw degrees in (-180, 180) range

        cos_ψ = np.cos(ψ) #wartość cosinusa z radianów
        sin_ψ = np.sin(ψ)

        x, y = location[0], location[1]
        pts_car = MPCController.transform_into_cars_coordinate_system(pts, x, y, cos_ψ, sin_ψ)

        poly = np.polyfit(pts_car[:, 0], pts_car[:, 1], self.poly_degree)

        cte = poly[-1]
        eψ = -np.arctan(poly[-2])

        # return cte, eψ
        init = (0, 0, 0, v, cte, eψ, *poly)
        self.state0 = self.get_state0(v, cte, eψ, self.steer, self.throttle, poly)
        result = self.minimize_cost(self.bounds, self.state0, init)

        if 'success' in result.message:
            self.steer = result.x[-self.steps_ahead]
            self.throttle = result.x[-2*self.steps_ahead]
        else:
            print('Unsuccessful optimization')

        actions = {
            'steer': round(self.steer, 3),
            'gas_brake': round(self.throttle, 3),
        }

        return actions

    def get_state0(self, v, cte, epsi, a, delta, poly):
        a = a or 0
        delta = delta or 0
        # "Go as the road goes"
        # x = np.linspace(0, self.steps_ahead*self.dt*v, self.steps_ahead)
        # y = np.polyval(poly, x)
        x = np.linspace(0, 1, self.steps_ahead)
        y = np.polyval(poly, x)
        psi = 0

        self.state0[:self.steps_ahead] = x
        self.state0[self.steps_ahead:2*self.steps_ahead] = y
        self.state0[2*self.steps_ahead:3*self.steps_ahead] = psi
        self.state0[3*self.steps_ahead:4*self.steps_ahead] = v
        self.state0[4*self.steps_ahead:5*self.steps_ahead] = cte
        self.state0[5*self.steps_ahead:6*self.steps_ahead] = epsi
        self.state0[6*self.steps_ahead:7*self.steps_ahead] = a
        self.state0[7*self.steps_ahead:8*self.steps_ahead] = delta
        return self.state0

    def generate_fun(self, symb_fun, vars_, init, poly):
        '''This function generates a function of the form `fun(x, *args)` because
        that's what the scipy `minimize` API expects (if we don't want to minimize
        over certain variables, we pass them as `args`)
        '''
        args = init + poly
        return sym.lambdify((vars_, *args), symb_fun, self.evaluator)
        # Equivalent to (but faster than):
        # func = sym.lambdify(vars_+init+poly, symb_fun, evaluator)
        # return lambda x, *args: func(*np.r_[x, args])

    def generate_grad(self, symb_fun, vars_, init, poly):
        args = init + poly
        return sym.lambdify(
            (vars_, *args),
            derive_by_array(symb_fun, vars_+args)[:len(vars_)],
            self.evaluator
        )
        # Equivalent to (but faster than):
        # cost_grad_funcs = [
        #     generate_fun(symb_fun.diff(var), vars_, init, poly)
        #     for var in vars_
        # ]
        # return lambda x, *args: [
        #     grad_func(np.r_[x, args]) for grad_func in cost_grad_funcs
        # ]

    def minimize_cost(self, bounds, x0, init):
        for constr_func in self.constr_funcs:
            constr_func['args'] = init

        return minimize(
            fun=self.cost_func,
            x0=x0,
            args=init,
            jac=self.cost_grad_func,
            bounds=bounds,
            constraints=self.constr_funcs,
            method='SLSQP',
            tol=self.tolerance,
        )

    @staticmethod
    def create_array_of_symbols(str_symbol, N):
        return sym.symbols('{symbol}0:{N}'.format(symbol=str_symbol, N=N))

    @staticmethod
    def transform_into_cars_coordinate_system(pts, x, y, cos_ψ, sin_ψ):
        diff = (pts - [x, y])
        pts_car = np.zeros_like(diff)
        pts_car[:, 0] = cos_ψ * diff[:, 0] + sin_ψ * diff[:, 1]
        pts_car[:, 1] = sin_ψ * diff[:, 0] - cos_ψ * diff[:, 1]
        return pts_car
