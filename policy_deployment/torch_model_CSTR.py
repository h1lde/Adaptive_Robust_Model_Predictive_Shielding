import torch as th
from torch import jit

# fourth-order Runge-Kutta implementation of CSTR model for parallel execution of robust rollouts

class CSTR_model_torch(th.nn.Module):
    def __init__(self):
        super(CSTR_model_torch, self).__init__()
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        # Model parameters - register as buffer to keep them on the same device
        self.register_buffer("K0_ab", th.tensor(1.287e12, device=self.device))
        self.register_buffer("K0_bc", th.tensor(1.287e12, device=self.device))
        self.register_buffer("K0_ad", th.tensor(9.043e9, device=self.device))
        self.register_buffer("E_A_ab", th.tensor(9758.3 * 1.00, device=self.device))
        self.register_buffer("E_A_bc", th.tensor(9758.3 * 1.00, device=self.device))
        self.register_buffer("E_A_ad", th.tensor(8560.0 * 1.0, device=self.device))
        self.register_buffer("H_R_ab", th.tensor(4.2, device=self.device))
        self.register_buffer("H_R_bc", th.tensor(-11.0, device=self.device))
        self.register_buffer("H_R_ad", th.tensor(-41.85, device=self.device))
        self.register_buffer("Rou", th.tensor(0.9342, device=self.device))
        self.register_buffer("Cp", th.tensor(3.01, device=self.device))
        self.register_buffer("Cp_k", th.tensor(2.0, device=self.device))
        self.register_buffer("A_R", th.tensor(0.215, device=self.device))
        self.register_buffer("V_R", th.tensor(10.01, device=self.device))
        self.register_buffer("K_w", th.tensor(4032.0, device=self.device))
        self.register_buffer("m_k", th.tensor(5.0, device=self.device))
        self.register_buffer("C_A0", th.tensor((5.7+4.5)/2.0*1.0, device=self.device))
        self.register_buffer("T_in", th.tensor(130.0, device=self.device))
        
        # Precompute constants to avoid repeated calculations
        self.register_buffer("neg_rou_cp", -self.Rou * self.Cp)
        self.register_buffer("kw_ar", self.K_w * self.A_R)
        self.register_buffer("rou_cp_vr", self.Rou * self.Cp * self.V_R)
        self.register_buffer("mk_cpk", self.m_k * self.Cp_k)
        self.register_buffer("temp_adj", th.tensor(273.15, device=self.device))
                
        # Constant for temperature adjustment in exponential
        self.register_buffer("temp_adj", th.tensor(273.15, device=self.device))

    @jit.export
    def derivatives(self, x: th.Tensor, u: th.Tensor, p: th.Tensor) -> th.Tensor:

        # Extract values (no new allocations)
        C_a = x[:, 0]
        C_b = x[:, 1]
        T_R = x[:, 2]
        T_K = x[:, 3]
        
        F = u[:, 0]
        Q_dot = u[:, 1]
        
        alpha = p[:, 0]
        beta = p[:, 1]
        
        # Pre-allocate result tensor
        result = th.empty_like(x)
        
        # Temperature adjustment (use in-place add to avoid allocation)
        T_R_adj = T_R + self.temp_adj
        
        # #Compute reaction constants (reuse common exponential terms)
        neg_E_A_ab_div_T = -self.E_A_ab / T_R_adj
        neg_E_A_bc_div_T = -self.E_A_bc / T_R_adj
        neg_alpha_E_A_ad_div_T = -alpha * self.E_A_ad / T_R_adj
        
        K_1 = beta * self.K0_ab * th.exp(neg_E_A_ab_div_T)
        K_2 = self.K0_bc * th.exp(neg_E_A_bc_div_T)
        K_3 = self.K0_ad * th.exp(neg_alpha_E_A_ad_div_T)
        
        # Temperature difference (reused in calculations)
        T_dif = T_R - T_K
        
        # Compute derivatives efficiently
        # C_a derivative
        C_a_sq = C_a * C_a  # Compute C_a^2 once
        result[:, 0] = F * (self.C_A0 - C_a) - K_1 * C_a - K_3 * C_a_sq
        
        # C_b derivative
        result[:, 1] = -F * C_b + K_1 * C_a - K_2 * C_b
        
        # T_R derivative (break into parts for clarity)
        heat_rxn = (K_1 * C_a * self.H_R_ab + K_2 * C_b * self.H_R_bc + K_3 * C_a_sq * self.H_R_ad) / self.neg_rou_cp
        heat_flow = F * (self.T_in - T_R)
        heat_exchange = (self.kw_ar * (-T_dif)) / self.rou_cp_vr
        result[:, 2] = heat_rxn + heat_flow + heat_exchange
        
        # T_K derivative
        result[:, 3] = (Q_dot + self.kw_ar * T_dif) / self.mk_cpk
        
        return result


class CSTR_simulator_torch(th.nn.Module):
    def __init__(self, model, dt):
        super(CSTR_simulator_torch, self).__init__()
        self.model = model
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        # Convert dt to tensor if it's not already
        if not isinstance(dt, th.Tensor):
            dt = th.tensor(dt, device=self.device, dtype=th.float32)
        self.register_buffer("dt", dt)
        
        # Pre-calculate RK4 coefficients
        self.register_buffer("dt_half", self.dt / 2.0)
        self.register_buffer("dt_sixth", self.dt / 6.0)
        self.register_buffer("two", th.tensor(2.0, device=self.device))

    @jit.export
    def make_step_batch(self, x: th.Tensor, u: th.Tensor, p: th.Tensor) -> th.Tensor:

        if self.device.type == 'cuda':
            with th.amp.autocast(device_type='cuda', enabled=True):
                # First RK stage
                k1 = self.model.derivatives(x, u, p)
                
                # Second RK stage
                x2 = x + self.dt_half * k1
                k2 = self.model.derivatives(x2, u, p)
                
                # Third RK stage
                x3 = x + self.dt_half * k2
                k3 = self.model.derivatives(x3, u, p)
                
                # Fourth RK stage
                x4 = x + self.dt * k3
                k4 = self.model.derivatives(x4, u, p)
                
                # Final integration step
                k1 = k1 + self.two * k2 + self.two * k3 + k4
                x_next = x + self.dt_sixth * k1
                
                return x_next
     
        elif self.device.type == 'cpu':
            with th.amp.autocast(device_type='cpu', enabled=True):
                # First RK stage
                k1 = self.model.derivatives(x, u, p)
                
                # Second RK stage
                x2 = x + self.dt_half * k1
                k2 = self.model.derivatives(x2, u, p)
                
                # Third RK stage
                x3 = x + self.dt_half * k2
                k3 = self.model.derivatives(x3, u, p)
                
                # Fourth RK stage
                x4 = x + self.dt * k3
                k4 = self.model.derivatives(x4, u, p)
                
                # Final integration step
                k1 = k1 + self.two * k2 + self.two * k3 + k4
                x_next = x + self.dt_sixth * k1
                
                return x_next
        else:
            raise ValueError(f"Unsupported device type: {self.device.type}")