import casadi as cd
import do_mpc
import numpy as np

def CSTR_model():
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # States struct (optimization variables):
    C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1,1))
    C_b = model.set_variable(var_type='_x', var_name='C_b', shape=(1,1))
    T_R = model.set_variable(var_type='_x', var_name='T_R', shape=(1,1))
    T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1,1))

    # Input struct (optimization variables):
    F = model.set_variable(var_type='_u', var_name='F')
    Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

    # Certain parameters
    K0_ab = 1.287e12 # K0 [h^-1]
    K0_bc = 1.287e12 # K0 [h^-1]
    K0_ad = 9.043e9 # K0 [l/mol.h]
    E_A_ab = 9758.3*1.00 #* R_gas# [kj/mol]
    E_A_bc = 9758.3*1.00 #* R_gas# [kj/mol]
    E_A_ad = 8560.0*1.0 #* R_gas# [kj/mol]
    H_R_ab = 4.2 # [kj/mol A]
    H_R_bc = -11.0 # [kj/mol B] Exothermic
    H_R_ad = -41.85 # [kj/mol A] Exothermic
    Rou = 0.9342 # Density [kg/l]
    Cp = 3.01 # Specific Heat capacity [kj/Kg.K]
    Cp_k = 2.0 # Coolant heat capacity [kj/kg.k]
    A_R = 0.215 # Area of reactor wall [m^2]
    V_R = 10.01 #0.01 # Volume of reactor [l]
    m_k = 5.0 # Coolant mass[kg]
    T_in = 130.0 # Temp of inflow [Celsius]
    K_w = 4032.0 # [kj/h.m^2.K]
    C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]

    # Uncertain parameters:
    alpha = model.set_variable(var_type='_p', var_name='alpha')
    beta = model.set_variable(var_type='_p', var_name='beta')

    # Auxiliary terms
    K_1 = beta * K0_ab * cd.exp((-E_A_ab)/((T_R+273.15)))
    K_2 =  K0_bc * cd.exp((-E_A_bc)/((T_R+273.15)))
    K_3 = K0_ad * cd.exp((-alpha*E_A_ad)/((T_R+273.15)))

    T_dif = model.set_expression(expr_name='T_dif', expr=T_R-T_K)

    model.set_rhs('C_a', F*(C_A0 - C_a) -K_1*C_a - K_3*(C_a**2))
    model.set_rhs('C_b', -F*C_b + K_1*C_a - K_2*C_b)
    model.set_rhs('T_R', ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R)))
    model.set_rhs('T_K', (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k))

    # Build the model
    model.setup()

    return model



def CSTR_msMPC(model, n_horizon: int = 20, n_robust: int = 1, t_step: float = 0.005,
                x_ub :np.array = np.array([2.0,2.0,140,140]), x_lb :np.array = np.array([0.1, 0.1, 100,100]),
                u_ub:np.array = np.array([100,0]), u_lb: np.array = np.array([5,-8500]), 
                alpha_var:np.array = np.array([0.8, 1.2]), beta_var:np.array = np.array([0.8, 1.2])): 
    
    mpc = do_mpc.controller.MPC(model)
    
    setup_mpc = {
    'n_horizon': n_horizon,
    'n_robust': n_robust,
    'open_loop': 0,
    't_step': t_step,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    mpc.scaling['_x', 'T_R'] = 100
    mpc.scaling['_x', 'T_K'] = 100
    mpc.scaling['_u', 'Q_dot'] = 2000
    mpc.scaling['_u', 'F'] = 100

    _x = model.x
    mterm = (_x['C_b'] - 0.6)**2 # terminal cost
    lterm = (_x['C_b'] - 0.6)**2 # stage cost

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(F=1e-1, Q_dot = 1e-3) # input penalty

    # # lower bounds of the states
    mpc.bounds['lower', '_x', 'C_a'] = x_lb[0]
    mpc.bounds['lower', '_x', 'C_b'] = x_lb[1]
    mpc.bounds['lower', '_x', 'T_R'] = x_lb[2]
    mpc.bounds['lower', '_x', 'T_K'] = x_lb[3]

    # upper bounds of the states
    mpc.bounds['upper', '_x', 'C_a'] = x_ub[0]
    mpc.bounds['upper', '_x', 'C_b'] = x_ub[1]
    mpc.bounds['upper', '_x', 'T_K'] = x_ub[2]
    mpc.bounds['upper', '_x', 'T_R'] = x_ub[3]

    # lower bounds of the inputs
    mpc.bounds['lower', '_u', 'F'] = u_lb[0]
    mpc.bounds['lower', '_u', 'Q_dot'] = u_lb[1]

    # upper bounds of the inputs
    mpc.bounds['upper', '_u', 'F'] = u_ub[0]
    mpc.bounds['upper', '_u', 'Q_dot'] = u_ub[1]

    mpc.set_uncertainty_values(alpha = alpha_var, beta = beta_var)

    mpc.setup()

    return mpc


def CSTR_simulator(model, t_step: float = 0.005, p: list = [1.0,1.0]):
    simulator = do_mpc.simulator.Simulator(model)
    params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': t_step
    }

    simulator.set_param(**params_simulator)

    # realization of uncertain parameters
    p_num = simulator.get_p_template()
    tvp_num = simulator.get_tvp_template()

    # function for time-varying parameters
    def tvp_fun(t_now):
        return tvp_num

    # uncertain parameters
    p_num['alpha'] = p[0]
    p_num['beta'] = p[1]

    def p_fun(t_now):
        return p_num
    
    simulator.set_tvp_fun(tvp_fun)
    simulator.set_p_fun(p_fun)
    
    simulator.setup()

    return simulator