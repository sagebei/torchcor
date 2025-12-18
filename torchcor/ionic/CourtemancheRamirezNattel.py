import torch
import torchcor as tc
from math import log, exp, expm1

@torch.jit.script
class CourtemancheRamirezNattel:
    def __init__(self, dt: float, device: torch.device=tc.get_device(), dtype: torch.dtype=torch.float64):
        self.name = "CourtemancheRamirezNattel"
        self.dt = dt
        self.device = device
        self.dtype = dtype

        # Constants
        self.C_B1a = 3.79138232501097e-05
        self.C_B1b = 0.0811764705882353
        self.C_B1c = 0.00705882352941176
        self.C_B1d = 0.00537112496043221
        self.C_B1e = 11.5
        self.C_Fn1 = 9.648e-13
        self.C_Fn2 = 2.5910306809e-13
        self.C_dCa_rel = 8.
        self.C_dCaup = 0.0869565217391304
        self.Ca_rel_init = 1.49
        self.Ca_up_init = 1.49
        self.Cai_init = 1.02e-1
        self.F = 96.4867
        self.K_Q10 = 3.
        self.K_up = 0.00092
        self.Ki_init = 139.0
        self.KmCa = 1.38
        self.KmCmdn = 0.00238
        self.KmCsqn = 0.8
        self.KmKo = 1.5
        self.KmNa = 87.5
        self.KmNa3 = 669921.875
        self.KmNai = 10.
        self.KmTrpn = 0.0005
        self.Nai = 11.2
        self.R = 8.3143
        self.T = 310.
        self.V_init = -81.2
        self.Volcell = 20100.
        self.Voli = 13668.
        self.Volrel = 96.48
        self.Volup = 1109.52
        self.d_init = 1.37e-4
        self.f_Ca_init = 0.775
        self.f_init = 0.999
        self.gamma = 0.35
        self.h_init = 0.965
        self.j_init = 0.978
        self.k_rel = 30.
        self.k_sat = 0.1
        self.m_init = 2.91e-3
        self.maxCmdn = 0.05
        self.maxCsqn = 10.
        self.maxTrpn = 0.07
        self.oa_init = 3.04e-2
        self.oi_init = 0.999
        self.tau_f_Ca = 2.
        self.tau_tr = 180.
        self.tau_u = 8.
        self.u_init = 0.
        self.ua_init = 4.96e-3
        self.ui_init = 0.999
        self.v_init = 1.
        self.w_init = 0.999
        self.xr_init = 3.29e-5
        self.xs_init = 1.87e-2

        # Parameters
        self.ACh = 0.000001
        self.Cao = 1.8
        self.Cm = 100.
        self.GACh = 0.
        self.GCaL = 0.1238
        self.GK1 = 0.09
        self.GKr = 0.0294
        self.GKs = 0.129
        self.GNa = 7.8
        self.GbCa = 0.00113
        self.GbNa = 0.000674
        self.Gto = 0.1652
        self.Ko = 5.4
        self.Nao = 140.
        self.factorGKur = 1.
        self.factorGrel = 1.
        self.factorGtr = 1.
        self.factorGup = 1.
        self.factorhGate = 0.
        self.factormGate = 0.
        self.factoroaGate = 0.
        self.factorxrGate = 1.
        self.maxCaup = 15.
        self.maxINaCa = 1600.
        self.maxINaK = 0.60
        self.maxIpCa = 0.275
        self.maxIup = 0.005

        # Cai_TableIndex
        self.carow_1_idx = 0
        self.carow_2_idx = 1
        self.carow_3_idx = 2
        self.conCa_idx = 3
        self.f_Ca_rush_larsen_A_idx = 4
        self.Cai_NROWS = 5

        # V_TableIndex
        self.GKur_idx = 0
        self.INaK_idx = 1
        self.IbNa_idx = 2
        self.d_rush_larsen_A_idx = 3
        self.d_rush_larsen_B_idx = 4
        self.f_rush_larsen_A_idx = 5
        self.f_rush_larsen_B_idx = 6
        self.h_rush_larsen_A_idx = 7
        self.h_rush_larsen_B_idx = 8
        self.j_rush_larsen_A_idx = 9
        self.j_rush_larsen_B_idx = 10
        self.m_rush_larsen_A_idx = 11
        self.m_rush_larsen_B_idx = 12
        self.oa_rush_larsen_A_idx = 13
        self.oa_rush_larsen_B_idx = 14
        self.oi_rush_larsen_A_idx = 15
        self.oi_rush_larsen_B_idx = 16
        self.ua_rush_larsen_A_idx = 17
        self.ua_rush_larsen_B_idx = 18
        self.ui_rush_larsen_A_idx = 19
        self.ui_rush_larsen_B_idx = 20
        self.vrow_29_idx = 21
        self.vrow_31_idx = 22
        self.vrow_32_idx = 23
        self.vrow_36_idx = 24
        self.vrow_7_idx = 25
        self.w_rush_larsen_A_idx = 26
        self.w_rush_larsen_B_idx = 27
        self.xr_rush_larsen_A_idx = 28
        self.xr_rush_larsen_B_idx = 29
        self.xs_rush_larsen_A_idx = 30
        self.xs_rush_larsen_B_idx = 31
        self.V_NROWS = 32

        # fn_TableIndex
        self.u_rush_larsen_A_idx = 0
        self.v_rush_larsen_A_idx = 1
        self.v_rush_larsen_B_idx = 2
        self.fn_NROWS = 3

        # Cai_TableParam
        self.Cai_T_mn = 3e-4
        self.Cai_T_mx = 30.0
        self.Cai_T_res = 3e-4
        self.Cai_T_step = 1 / self.Cai_T_res
        self.Cai_T_mn_idx = 0
        self.Cai_T_mx_idx = int((self.Cai_T_mx - self.Cai_T_mn) * self.Cai_T_step) - 1

        # V_TableParam
        self.V_T_mn = -200.
        self.V_T_mx = 200.
        self.V_T_res = 0.1
        self.V_T_step = 1 / self.V_T_res
        self.V_T_mn_idx = 0.
        self.V_T_mx_idx = int((self.V_T_mx - self.V_T_mn) * self.V_T_step) - 1

        # fn_TableParam
        self.fn_T_mn = -2e-11
        self.fn_T_mx = 10.0e-11
        self.fn_T_res = 2e-15
        self.fn_T_step = 1 / self.fn_T_res
        self.fn_T_mn_idx = 0.
        self.fn_T_mx_idx = int((self.fn_T_mx - self.fn_T_mn) * self.fn_T_step) - 1

        # 3 lookup tables
        self.Cai_tab = torch.tensor([1.0])
        self.V_tab = torch.tensor([1.0])
        self.fn_tab = torch.tensor([1.0])

        # 19 states variables
        self.Ca_rel = torch.tensor([self.Ca_rel_init])
        self.Ca_up = torch.tensor([self.Ca_up_init])
        self.Cai = torch.tensor([self.Cai_init])
        self.Ki = torch.tensor([self.Ki_init])
        self.d = torch.tensor([self.d_init])
        self.f = torch.tensor([self.f_init])
        self.f_Ca = torch.tensor([self.f_Ca_init])
        self.h = torch.tensor([self.h_init])
        self.j = torch.tensor([self.j_init])
        self.m = torch.tensor([self.m_init])
        self.oa = torch.tensor([self.oa_init])
        self.oi = torch.tensor([self.oi_init])
        self.u = torch.tensor([self.u_init])
        self.ua = torch.tensor([self.ua_init])
        self.ui = torch.tensor([self.ui_init])
        self.v = torch.tensor([self.v_init])
        self.w = torch.tensor([self.w_init])
        self.xr = torch.tensor([self.xr_init])
        self.xs = torch.tensor([self.xs_init])

        self.f_Ca_rush_larsen_B = exp(((-self.dt)/self.tau_f_Ca))
        self.u_rush_larsen_B = exp(((-self.dt)/self.tau_u))


    def interpolate(self, X, table, mn: float, mx: float, res: float, step: float, mx_idx: int):
        X = torch.clamp(X, mn, mx)
        idx = ((X - mn) * step).to(torch.long)
        lower_idx = torch.clamp(idx, 0, mx_idx - 1)
        higher_idx = lower_idx + 1
        lower_pos = lower_idx * res + mn
        w = ((X - lower_pos) / res).unsqueeze(1)
        return (1 - w) * table[lower_idx] + w * table[higher_idx]

    def construct_tables(self):
        E_Na = ((self.R*self.T)/self.F)*(log((self.Nao/self.Nai)))
        f_Ca_rush_larsen_C = expm1(((-self.dt)/self.tau_f_Ca))
        sigma = ((exp((self.Nao/67.3)))-1.)/7.
        u_rush_larsen_C = expm1(((-self.dt)/self.tau_u))

        # construct Cai Lookup Table
        Cai = torch.arange(self.Cai_T_mn, self.Cai_T_mx, self.Cai_T_res).to(self.device).to(self.dtype)
        Cai_tab = torch.zeros((Cai.shape[0], self.Cai_NROWS)).to(self.device).to(self.dtype)

        Cai_tab[:, self.conCa_idx] = Cai / 1000
        Cai_tab[:, self.carow_1_idx] = (self.factorGup*self.maxIup)/(1.+(self.K_up/Cai_tab[:, self.conCa_idx]))
        Cai_tab[:, self.carow_2_idx] = (((((((self.maxTrpn*self.KmTrpn)/(Cai_tab[:, self.conCa_idx]+self.KmTrpn))/(Cai_tab[:, self.conCa_idx]+self.KmTrpn))+(((self.maxCmdn*self.KmCmdn)/(Cai_tab[:, self.conCa_idx]+self.KmCmdn))/(Cai_tab[:, self.conCa_idx]+self.KmCmdn)))+1.)/self.C_B1c)/1000.)
        Cai_tab[:, self.carow_3_idx] = (((self.maxIpCa*Cai_tab[:, self.conCa_idx])/(0.0005+Cai_tab[:, self.conCa_idx]))-(((((self.GbCa*self.R)*self.T)/2.)/self.F)*(torch.log((self.Cao/Cai_tab[:, self.conCa_idx])))))
        f_Ca_inf = 1./(1.+(Cai_tab[:, self.conCa_idx]/0.00035))
        Cai_tab[:, self.f_Ca_rush_larsen_A_idx] = (-f_Ca_inf)*f_Ca_rush_larsen_C

        self.Cai_tab = Cai_tab

        # construct V Lookup Table
        V = torch.arange(self.V_T_mn, self.V_T_mx, self.V_T_res).to(self.device).to(self.dtype)
        V_tab = torch.zeros((V.shape[0], self.V_NROWS)).to(self.device).to(self.dtype)

        V_tab[:, self.GKur_idx] = (0.005 + (0.05 / (1. + (torch.exp(((15. - V) / 13.))))))
        V_tab[:, self.IbNa_idx] = (self.GbNa * (V - E_Na))
        a_h = torch.where(V >= -40., 0.0, (0.135 * (torch.exp((((V+80.) - self.factorhGate) / -6.8)))))
        a_j = torch.where(V < -40., ((((-127140.*(torch.exp((0.2444*V))))-(3.474e-5*(torch.exp((-0.04391*V)))))*(V+37.78))/(1.+(torch.exp((0.311*(V+79.23)))))), 0.)
        a_m = torch.where(V == -47.13, 3.2, ((0.32*(V+47.13))/(1.-(torch.exp((-0.1*(V+47.13)))))))
        aa_oa = (0.65 / ((torch.exp(((V+10.)/-8.5))) + (torch.exp(((30. - V) / 59.0)))))
        aa_oi = (1./(18.53+(torch.exp(((V+113.7)/10.95)))))
        aa_ua = (0.65 / ((torch.exp(((V+10.)/-8.5))) + (torch.exp(((V - 30.) / -59.0)))))
        aa_ui = (1. / (21. + (torch.exp(((V - 185.) / -28.)))))
        aa_xr = (self.factorxrGate * ((0.0003*(V+14.1)) / (1. - (torch.exp(((V + 14.1) / -5.))))))
        aa_xs = ((4.e-5*(V-(19.9))) / (1. - (torch.exp(((19.9 - (V)) / 17.)))))
        b_h = torch.where((V >= -40.), ((1./0.13)/(1.+(torch.exp(((-(V+10.66))/11.1))))), ((3.56*(torch.exp((0.079*V))))+(3.1e5*(torch.exp((0.35*V))))))
        b_j = torch.where((V >= -40.), ((0.3*(torch.exp((-2.535e-7*V))))/(1.+(torch.exp((-0.1*(V+32.)))))), ((0.1212*(torch.exp((-0.01052*V))))/(1.+(torch.exp((-0.1378*(V+40.14)))))))
        b_m = (0.08 * (torch.exp(((-(V - self.factormGate)) / 11.))))
        bb_oa = (0.65/(2.5+(torch.exp(((V+82.)/17.)))))
        bb_oi = (1./(35.56+(torch.exp(((-(V+1.26))/7.44)))))
        bb_ua = (0.65/(2.5+(torch.exp(((V+82.)/17.)))))
        bb_ui = (torch.exp(((V - 158.) / 16.)))
        bb_xr = ((1.0/self.factorxrGate) * ((7.3898e-5 * (V - 3.3328)) / ((torch.exp(((V - 3.3328) / 5.1237))) - (1.))))
        bb_xs = ((3.5e-5 * (V - 19.9)) / ((torch.exp(((V - 19.9) / 9.))) - 1.))
        d_inf = (1./(1.+(torch.exp(((-(V+10.))/8.)))))
        f_NaK = (1./((1.+(0.1245*(torch.exp(((((-0.1*self.F)*V)/self.R)/self.T)))))+((0.0365*sigma)*(torch.exp(((((-self.F)*V)/self.R)/self.T))))))
        f_inf = (1./(1.+(torch.exp(((V+28.)/6.9)))))
        oa_inf = (1. / (1. + (torch.exp(((-((V+20.47) - self.factoroaGate)) / 17.54)))))
        oi_inf = (1./(1.+(torch.exp(((V+43.1)/5.3)))))
        tau_d = torch.where(V == -10., (((1./6.24)/0.035)/2.), ((((1.-((torch.exp(((-(V+10.))/6.24)))))/0.035)/(V+10.))/(1.+(torch.exp(((-(V+10.))/6.24))))))
        tau_f = (9./((0.0197*(torch.exp((((-0.0337*0.0337)*(V+10.))*(V+10.)))))+0.02))
        tau_w = (((6. * (1. - (torch.exp(((7.9 - V) / 5.))))) / (1. + (0.3 * (torch.exp(((7.9 - (V)) / 5.)))))) / (V - (7.9)))
        ua_inf = (1./(1.+(torch.exp(((V+30.3)/-9.6)))))
        ui_inf = (1. / (1. + (torch.exp(((V - 99.45) / 27.48)))))
        V_tab[:, self.vrow_29_idx] = (self.GCaL * (V - 65.))
        V_tab[:, self.vrow_31_idx] = ((((((((self.maxINaCa*(torch.exp(((((self.gamma*self.F)*V)/self.R)/self.T))))*self.Nai)*self.Nai)*self.Nai)*self.Cao)/(self.KmNa3+((self.Nao*self.Nao)*self.Nao)))/(self.KmCa+self.Cao))/(1.+(self.k_sat*(torch.exp((((((self.gamma-(1.))*self.F)*V)/self.R)/self.T))))))
        V_tab[:, self.vrow_32_idx] = ((((((((self.maxINaCa * (torch.exp((((((self.gamma - 1.) * self.F) * V) / self.R) / self.T)))) * self.Nao) * self.Nao) * self.Nao) / (self.KmNa3 + ((self.Nao * self.Nao) * self.Nao))) / (self.KmCa + self.Cao)) / (1. + (self.k_sat * (torch.exp((((((self.gamma - (1.)) * self.F) * V) / self.R) / self.T)))))) / 1000.)
        V_tab[:, self.vrow_36_idx] = (V*self.GbCa)
        V_tab[:, self.vrow_7_idx] = (self.GNa * (V - E_Na))
        w_inf = (1. - (1. / (1. + (torch.exp(((40. - V) / 17.))))))
        xr_inf = (1./(1.+(torch.exp(((V+14.1)/-6.5)))))
        xs_inf = (1. / (torch.sqrt((1. + (torch.exp(((V - 19.9) / -12.7)))))))
        V_tab[:, self.INaK_idx] = ((((self.maxINaK*f_NaK)/(1.+(pow((self.KmNai/self.Nai),1.5))))*self.Ko)/(self.Ko+self.KmKo))
        V_tab[:, self.d_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_d)))
        d_rush_larsen_C = (torch.expm1(((-self.dt)/tau_d)))
        V_tab[:, self.f_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_f)))
        f_rush_larsen_C = (torch.expm1(((-self.dt)/tau_f)))
        V_tab[:, self.h_rush_larsen_A_idx] = (((-a_h)/(a_h+b_h))*(torch.expm1(((-self.dt)*(a_h+b_h)))))
        V_tab[:, self.h_rush_larsen_B_idx] = (torch.exp(((-self.dt)*(a_h+b_h))))
        V_tab[:, self.j_rush_larsen_A_idx] = (((-a_j)/(a_j+b_j))*(torch.expm1(((-self.dt)*(a_j+b_j)))))
        V_tab[:, self.j_rush_larsen_B_idx] = (torch.exp(((-self.dt)*(a_j+b_j))))
        V_tab[:, self.m_rush_larsen_A_idx] = (((-a_m)/(a_m+b_m))*(torch.expm1(((-self.dt)*(a_m+b_m)))))
        V_tab[:, self.m_rush_larsen_B_idx] = (torch.exp(((-self.dt)*(a_m+b_m))))
        tau_oa = ((1./(aa_oa+bb_oa))/self.K_Q10)
        tau_oi = ((1./(aa_oi+bb_oi))/self.K_Q10)
        tau_ua = ((1./(aa_ua+bb_ua))/self.K_Q10)
        tau_ui = ((1./(aa_ui+bb_ui))/self.K_Q10)
        tau_xr = (1./(aa_xr+bb_xr))
        tau_xs = (0.5/(aa_xs+bb_xs))
        V_tab[:, self.w_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_w)))
        w_rush_larsen_C = (torch.expm1(((-self.dt)/tau_w)))
        V_tab[:, self.d_rush_larsen_A_idx] = ((-d_inf)*d_rush_larsen_C)
        V_tab[:, self.f_rush_larsen_A_idx] = ((-f_inf)*f_rush_larsen_C)
        V_tab[:, self.oa_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_oa)))
        oa_rush_larsen_C = (torch.expm1(((-self.dt)/tau_oa)))
        V_tab[:, self.oi_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_oi)))
        oi_rush_larsen_C = (torch.expm1(((-self.dt)/tau_oi)))
        V_tab[:, self.ua_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_ua)))
        ua_rush_larsen_C = (torch.expm1(((-self.dt)/tau_ua)))
        V_tab[:, self.ui_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_ui)))
        ui_rush_larsen_C = (torch.expm1(((-self.dt)/tau_ui)))
        V_tab[:, self.w_rush_larsen_A_idx] = ((-w_inf)*w_rush_larsen_C)
        V_tab[:, self.xr_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_xr)))
        xr_rush_larsen_C = (torch.expm1(((-self.dt)/tau_xr)))
        V_tab[:, self.xs_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_xs)))
        xs_rush_larsen_C = (torch.expm1(((-self.dt)/tau_xs)))
        V_tab[:, self.oa_rush_larsen_A_idx] = ((-oa_inf)*oa_rush_larsen_C)
        V_tab[:, self.oi_rush_larsen_A_idx] = ((-oi_inf)*oi_rush_larsen_C)
        V_tab[:, self.ua_rush_larsen_A_idx] = ((-ua_inf)*ua_rush_larsen_C)
        V_tab[:, self.ui_rush_larsen_A_idx] = ((-ui_inf)*ui_rush_larsen_C)
        V_tab[:, self.xr_rush_larsen_A_idx] = ((-xr_inf)*xr_rush_larsen_C)
        V_tab[:, self.xs_rush_larsen_A_idx] = ((-xs_inf)*xs_rush_larsen_C)

        self.V_tab = V_tab

        # Create the fn lookup table
        fn = torch.arange(self.fn_T_mn, self.fn_T_mx, self.fn_T_res).to(self.device).to(self.dtype)
        fn_tab = torch.zeros((fn.shape[0], self.fn_NROWS)).to(self.device).to(self.dtype)

        tau_v = (1.91+(2.09/(1.+(torch.exp(((3.4175e-13-fn)/13.67e-16))))))
        u_inf = (1./(1.+(torch.exp(((3.4175e-13-fn)/13.67e-16)))))
        v_inf = (1. - (1. / (1. + (torch.exp(((6.835e-14 - fn) / 13.67e-16))))))
        fn_tab[:, self.u_rush_larsen_A_idx] = ((-u_inf)*u_rush_larsen_C)
        fn_tab[:, self.v_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_v)))
        v_rush_larsen_C = (torch.expm1(((-self.dt)/tau_v)))
        fn_tab[:, self.v_rush_larsen_A_idx] = ((-v_inf)*v_rush_larsen_C)

        self.fn_tab = fn_tab

    def initialize(self, n_nodes: int):
        self.construct_tables()
        
        V = torch.full((n_nodes,), self.V_init, device=self.device, dtype=self.dtype)

        self.Ca_rel = torch.full((n_nodes,), self.Ca_rel_init, device=self.device, dtype=self.dtype)
        self.Ca_up = torch.full((n_nodes,), self.Ca_up_init, device=self.device, dtype=self.dtype)
        self.Cai = torch.full((n_nodes,), self.Cai_init, device=self.device, dtype=self.dtype)
        self.Ki = torch.full((n_nodes,), self.Ki_init, device=self.device, dtype=self.dtype)
        self.d = torch.full((n_nodes,), self.d_init, device=self.device, dtype=self.dtype)
        self.f = torch.full((n_nodes,), self.f_init, device=self.device, dtype=self.dtype)
        self.f_Ca = torch.full((n_nodes,), self.f_Ca_init, device=self.device, dtype=self.dtype)
        self.h = torch.full((n_nodes,), self.h_init, device=self.device, dtype=self.dtype)
        self.j = torch.full((n_nodes,), self.j_init, device=self.device, dtype=self.dtype)
        self.m = torch.full((n_nodes,), self.m_init, device=self.device, dtype=self.dtype)
        self.oa = torch.full((n_nodes,), self.oa_init, device=self.device, dtype=self.dtype)
        self.oi = torch.full((n_nodes,), self.oi_init, device=self.device, dtype=self.dtype)
        self.u = torch.full((n_nodes,), self.u_init, device=self.device, dtype=self.dtype)
        self.ua = torch.full((n_nodes,), self.ua_init, device=self.device, dtype=self.dtype)
        self.ui = torch.full((n_nodes,), self.ui_init, device=self.device, dtype=self.dtype)
        self.v = torch.full((n_nodes,), self.v_init, device=self.device, dtype=self.dtype)
        self.w = torch.full((n_nodes,), self.w_init, device=self.device, dtype=self.dtype)
        self.xr = torch.full((n_nodes,), self.xr_init, device=self.device, dtype=self.dtype)
        self.xs = torch.full((n_nodes,), self.xs_init, device=self.device, dtype=self.dtype)

        return V


    def differentiate(self, V):
        V_row = self.interpolate(V, self.V_tab, self.V_T_mn, self.V_T_mx, self.V_T_res, self.V_T_step, self.V_T_mx_idx)
        Cai_row = self.interpolate(self.Cai, self.Cai_tab, self.Cai_T_mn, self.Cai_T_mx, self.Cai_T_res, self.Cai_T_step, self.Cai_T_mx_idx)

        # Compute storevars and external modvars
        E_K = (((self.R*self.T)/self.F)*(torch.log((self.Ko/self.Ki))))
        ICaL = (((V_row[:, self.vrow_29_idx]*self.d)*self.f)*self.f_Ca)
        IKACh = (((self.GACh*(10.0/(1.0+(9.13652/(pow(self.ACh,0.477811))))))*(0.0517+(0.4516/(1.0+(torch.exp(((V+59.53)/17.18))))))) * (V - E_K))
        INa = (((((V_row[:, self.vrow_7_idx]*self.m)*self.m)*self.m)*self.h)*self.j)
        INaCa = (V_row[:, self.vrow_31_idx] - (self.Cai * V_row[:, self.vrow_32_idx]))
        vrow_13 = (self.Gto * (V - E_K))
        vrow_18 = ((self.factorGKur*V_row[:, self.GKur_idx]) * (V - E_K))
        vrow_21 = ((self.GKr * (V - E_K)) / (1. + (torch.exp(((V + 15.) / 22.4)))))
        vrow_24 = (self.GKs * (V - E_K))
        vrow_8 = ((self.GK1 * (V - E_K)) / (1. + (torch.exp((0.07 * (V + 80.))))))
        IK1 = vrow_8
        IKr = (vrow_21*self.xr)
        IKs = ((vrow_24*self.xs)*self.xs)
        IKur = ((((vrow_18*self.ua)*self.ua)*self.ua)*self.ui)
        IpCa = (Cai_row[:, self.carow_3_idx]+V_row[:, self.vrow_36_idx])
        Ito = ((((vrow_13*self.oa)*self.oa)*self.oa)*self.oi)
        Iion = INa+IK1+Ito+IKur+IKr+IKs+ICaL+IpCa+INaCa+V_row[:, self.IbNa_idx]+V_row[:, self.INaK_idx]+IKACh

        # Complete Forward Euler Update
        Itr = ((self.factorGtr * (self.Ca_up - self.Ca_rel)) / self.tau_tr)
        Irel = ((((((self.factorGrel*self.u)*self.u)*self.v)*self.k_rel)*self.w)*(self.Ca_rel-(Cai_row[:, self.conCa_idx])))
        dIups = (Cai_row[:, self.carow_1_idx] - ((self.maxIup / self.maxCaup) * self.Ca_up))
        diff_Ca_rel = ((Itr - Irel) / (1. + ((self.C_dCa_rel / (self.Ca_rel + self.KmCsqn)) / (self.Ca_rel + self.KmCsqn))))
        diff_Ki = ((-((((((Ito+IKr)+IKur)+IKs)+IK1)+IKACh) - (2.0 * V_row[:, self.INaK_idx]))) / (self.F * self.Voli))
        diff_Ca_up = (dIups - (Itr * self.C_dCaup))
        diff_Cai = ((((self.C_B1d * (((INaCa+INaCa) - IpCa) - ICaL)) - (self.C_B1e * dIups)) + Irel) / Cai_row[:, self.carow_2_idx])

        self.Ca_rel = self.Ca_rel+diff_Ca_rel*self.dt
        self.Ca_up = self.Ca_up+diff_Ca_up*self.dt
        self.Cai = self.Cai+diff_Cai*self.dt
        self.Ki = self.Ki+diff_Ki*self.dt

        # Complete Rush Larsen Update
        fn = ((self.C_Fn1*Irel) - (self.C_Fn2 * (ICaL - (0.4 * INaCa))))
        fn_row = self.interpolate(fn, self.fn_tab, self.fn_T_mn, self.fn_T_mx, self.fn_T_res, self.fn_T_step, self.fn_T_mx_idx)

        d_rush_larsen_B = V_row[:, self.d_rush_larsen_B_idx]
        f_rush_larsen_B = V_row[:, self.f_rush_larsen_B_idx]
        h_rush_larsen_A = V_row[:, self.h_rush_larsen_A_idx]
        h_rush_larsen_B = V_row[:, self.h_rush_larsen_B_idx]
        j_rush_larsen_A = V_row[:, self.j_rush_larsen_A_idx]
        j_rush_larsen_B = V_row[:, self.j_rush_larsen_B_idx]
        m_rush_larsen_A = V_row[:, self.m_rush_larsen_A_idx]
        m_rush_larsen_B = V_row[:, self.m_rush_larsen_B_idx]
        w_rush_larsen_B = V_row[:, self.w_rush_larsen_B_idx]
        d_rush_larsen_A = V_row[:, self.d_rush_larsen_A_idx]
        f_Ca_rush_larsen_A = Cai_row[:, self.f_Ca_rush_larsen_A_idx]
        f_rush_larsen_A = V_row[:, self.f_rush_larsen_A_idx]
        oa_rush_larsen_B = V_row[:, self.oa_rush_larsen_B_idx]
        oi_rush_larsen_B = V_row[:, self.oi_rush_larsen_B_idx]
        u_rush_larsen_A = fn_row[:, self.u_rush_larsen_A_idx]
        ua_rush_larsen_B = V_row[:, self.ua_rush_larsen_B_idx]
        ui_rush_larsen_B = V_row[:, self.ui_rush_larsen_B_idx]
        v_rush_larsen_B = fn_row[:, self.v_rush_larsen_B_idx]
        w_rush_larsen_A = V_row[:, self.w_rush_larsen_A_idx]
        xr_rush_larsen_B = V_row[:, self.xr_rush_larsen_B_idx]
        xs_rush_larsen_B = V_row[:, self.xs_rush_larsen_B_idx]
        oa_rush_larsen_A = V_row[:, self.oa_rush_larsen_A_idx]
        oi_rush_larsen_A = V_row[:, self.oi_rush_larsen_A_idx]
        ua_rush_larsen_A = V_row[:, self.ua_rush_larsen_A_idx]
        ui_rush_larsen_A = V_row[:, self.ui_rush_larsen_A_idx]
        v_rush_larsen_A = fn_row[:, self.v_rush_larsen_A_idx]
        xr_rush_larsen_A = V_row[:, self.xr_rush_larsen_A_idx]
        xs_rush_larsen_A = V_row[:, self.xs_rush_larsen_A_idx]

        self.d = d_rush_larsen_A + d_rush_larsen_B * self.d
        self.f = f_rush_larsen_A + f_rush_larsen_B * self.f
        self.f_Ca = f_Ca_rush_larsen_A + self.f_Ca_rush_larsen_B * self.f_Ca
        self.h = h_rush_larsen_A + h_rush_larsen_B * self.h
        self.j = j_rush_larsen_A + j_rush_larsen_B * self.j
        self.m = m_rush_larsen_A + m_rush_larsen_B * self.m
        self.oa = oa_rush_larsen_A + oa_rush_larsen_B * self.oa
        self.oi = oi_rush_larsen_A + oi_rush_larsen_B * self.oi
        self.u = u_rush_larsen_A + self.u_rush_larsen_B * self.u
        self.ua = ua_rush_larsen_A + ua_rush_larsen_B * self.ua
        self.ui = ui_rush_larsen_A + ui_rush_larsen_B * self.ui
        self.v = v_rush_larsen_A + v_rush_larsen_B * self.v
        self.w = w_rush_larsen_A + w_rush_larsen_B * self.w
        self.xr = xr_rush_larsen_A + xr_rush_larsen_B * self.xr
        self.xs = xs_rush_larsen_A + xs_rush_larsen_B * self.xs

        return -Iion



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    dt = 0.02
    stimulus = 50
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    ionic = CourtemancheRamirezNattel(dt=dt, 
                                      device=device, 
                                      dtype=torch.float32)
    V = ionic.initialize(n_nodes=1)

    V_list = []
    ctime = 0.0
    for _ in range(int(1000/dt)):
        V_list.append([ctime, V.item()])

        dV = ionic.differentiate(V)
        V = V + dt * dV
        ctime += dt
        if ctime >= 0 and ctime <= (0+2.0): 
            V = V + dt * stimulus
    
    plt.figure()
    V_list = np.array(V_list)    
    plt.plot(V_list[:, 0], V_list[:, 1])
    plt.savefig("V_CourtemancheRamirezNattel.png")