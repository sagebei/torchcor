import torch
import torchcor as tc
from math import exp, sqrt


@torch.jit.script
class TenTusscherPanfilov:
    def __init__(self, cell_type: str, dt: float, device: torch.device=tc.get_device(), dtype: torch.dtype=torch.float64):
        self.name = "TenTusscherPanfilov"
        self.dt = dt
        self.device = device
        self.dtype = dtype

        self.cell_type = "EPI" if cell_type is None else cell_type
        # Constants
        self.CaSR_init = 1.3
        self.CaSS_init = 0.00007
        self.Cai_init = 0.00007
        self.EC = 1.5
        self.ENDO = 2.
        self.EPI = 0.
        self.F2_init = 1.
        self.FCaSS_init = 1.
        self.F_init = 1.
        self.H_init = 0.75
        self.J_init = 0.75
        self.Ki_init = 138.3
        self.MCELL = 1.
        self.M_init = 0.
        self.Nai_init = 7.67
        self.O_init = 0.
        self.R__init = 1.
        self.R_init = 0.
        self.S_init = 1.
        self.V_init = -86.2
        self.Vxfer = 0.0038
        self.Xr1_init = 0.
        self.Xr2_init = 1.
        self.Xs_init = 0.
        self.d_init = 0.
        self.k1_ = 0.15
        self.k2_ = 0.045
        self.k3 = 0.060
        self.k4 = 0.005
        self.maxsr = 2.5
        self.minsr = 1.

        # Parameters
        self.Bufc = 0.2
        self.Bufsr = 10.
        self.Bufss = 0.4
        self.CAPACITANCE = 0.185
        self.Cao = 2.0
        self.D_CaL_off = 0.
        self.Fconst = 96485.3415
        self.GK1 = 5.405
        self.GNa = 14.838
        self.GbCa = 0.000592
        self.GbNa = 0.00029
        self.GpCa = 0.1238
        self.GpK = 0.0146
        self.Kbufc = 0.001
        self.Kbufsr = 0.3
        self.Kbufss = 0.00025
        self.KmCa = 1.38
        self.KmK = 1.0
        self.KmNa = 40.0
        self.KmNai = 87.5
        self.Ko = 5.4
        self.KpCa = 0.0005
        self.Kup = 0.00025
        self.Nao = 140.0
        self.Rconst = 8314.472
        self.T = 310.0
        self.Vc = 0.016404
        self.Vleak = 0.00036
        self.Vmaxup = 0.006375
        self.Vrel = 0.102
        self.Vsr = 0.001094
        self.Vss = 0.00005468
        self.knaca = 1000.
        self.knak = 2.724
        self.ksat = 0.1
        self.n = 0.35
        self.pKNa = 0.03
        self.scl_tau_f = 1.
        self.vHalfXs = 5.
        self.xr2_off = 0.
        self.GCaL_init = 0.00003980
        self.GKr_init = 0.153
        self.GKs_init =  0.392 if cell_type == "EPI" else 0.098 if cell_type == "MCELL" else 0.392
        self.Gto_init = 0.294 if cell_type == "EPI" else 0.294 if cell_type == "MCELL" else 0.073

        # CaSS_TableIndex
        self.CaSS2_idx = 0
        self.FCaSS_rush_larsen_A_idx = 1
        self.FCaSS_rush_larsen_B_idx = 2
        self.CaSS_NROWS = 3

        # V_TableIndex
        self.D_rush_larsen_A_idx = 0
        self.D_rush_larsen_B_idx = 1
        self.F2_rush_larsen_A_idx = 2
        self.F2_rush_larsen_B_idx = 3
        self.F_rush_larsen_A_idx = 4
        self.F_rush_larsen_B_idx = 5
        self.H_rush_larsen_A_idx = 6
        self.H_rush_larsen_B_idx = 7
        self.INaCa_A_idx = 8
        self.INaCa_B_idx = 9
        self.J_rush_larsen_A_idx = 10
        self.J_rush_larsen_B_idx = 11
        self.M_rush_larsen_A_idx = 12
        self.M_rush_larsen_B_idx = 13
        self.R_rush_larsen_A_idx = 14
        self.R_rush_larsen_B_idx = 15
        self.S_rush_larsen_A_idx = 16
        self.S_rush_larsen_B_idx = 17
        self.Xr1_rush_larsen_A_idx = 18
        self.Xr1_rush_larsen_B_idx = 19
        self.Xr2_rush_larsen_A_idx = 20
        self.Xr2_rush_larsen_B_idx = 21
        self.Xs_rush_larsen_A_idx = 22
        self.Xs_rush_larsen_B_idx = 23
        self.a2_idx = 24
        self.rec_iNaK_idx = 25 
        self.rec_ipK_idx = 26
        self.V_NROWS = 27

        # VEk_TableIndex
        self.rec_iK1_idx = 0
        self.VEk_NROWS =1

        # CaSS_TableParam
        self.CaSS_T_mn = 0.00001
        self.CaSS_T_mx = 10.0
        self.CaSS_T_res = 0.00001
        self.CaSS_T_step = 1 / self.CaSS_T_res
        self.CaSS_T_mx_idx = int((self.CaSS_T_mx - self.CaSS_T_mn) * self.CaSS_T_step) - 1

        # V_TableParam
        self.V_T_mn = -800.
        self.V_T_mx = 800.
        self.V_T_res = 0.05
        self.V_T_step = 1 / self.V_T_res
        self.V_T_mx_idx = int((self.V_T_mx - self.V_T_mn) * self.V_T_step) - 1

        # VEk_TableParam
        self.VEk_T_mn = -800.
        self.VEk_T_mx = 800.
        self.VEk_T_res = 0.05
        self.VEk_T_step = 1 / self.VEk_T_res
        self.VEk_T_mx_idx = int((self.VEk_T_mx - self.VEk_T_mn) * self.VEk_T_step) - 1

        # 3 lookup tables
        self.CaSS_tab = torch.tensor([1.0])
        self.V_tab = torch.tensor([1.0])
        self.VEk_tab = torch.tensor([1.0])

        # 22 states variables
        self.GCaL = torch.tensor([self.GCaL_init])
        self.GKr = torch.tensor([self.GKr_init])
        self.GKs = torch.tensor([self.GKs_init])
        self.Gto = torch.tensor([self.Gto_init])
        self.CaSR = torch.tensor([self.CaSR_init])
        self.CaSS = torch.tensor([self.CaSS_init])
        self.Cai = torch.tensor([self.Cai_init])
        self.F = torch.tensor([self.F_init])
        self.F2 = torch.tensor([self.F2_init])
        self.FCaSS = torch.tensor([self.FCaSS_init])
        self.H = torch.tensor([self.H_init])
        self.J = torch.tensor([self.J_init])
        self.Ki = torch.tensor([self.Ki_init])
        self.M = torch.tensor([self.M_init])
        self.Nai = torch.tensor([self.Nai_init])
        self.R = torch.tensor([self.R_init])
        self.R_ = torch.tensor([self.R__init])
        self.S = torch.tensor([self.S_init])
        self.Xr1 = torch.tensor([self.Xr1_init])
        self.Xr2 = torch.tensor([self.Xr2_init])
        self.Xs = torch.tensor([self.Xs_init])
        self.D = torch.tensor([self.d_init])

        self.V_row = None


    def interpolate(self, X, table, mn: float, mx: float, res: float, step: float, mx_idx: int):
        X = torch.clamp(X, mn, mx)
        idx = ((X - mn) * step).to(torch.long)
        lower_idx = torch.clamp(idx, 0, mx_idx - 1)
        higher_idx = lower_idx + 1
        lower_pos = lower_idx * res + mn
        w = ((X - lower_pos) / res).unsqueeze(1)
        return (1 - w) * table[lower_idx] + w * table[higher_idx]

    def construct_tables(self):
        KmNai3 = ((self.KmNai*self.KmNai)*self.KmNai)
        Nao3 = ((self.Nao*self.Nao)*self.Nao)
        RTONF = ((self.Rconst*self.T)/self.Fconst)
        invKmCa_Cao = (1./(self.KmCa+self.Cao))
        F_RT = (1./RTONF)
        invKmNai3_Nao3 = (1./(KmNai3+Nao3))
        pmf_INaCa = ((self.knaca*invKmNai3_Nao3)*invKmCa_Cao)

        # construct the CaSS lookup table
        CaSS = torch.arange(self.CaSS_T_mn, self.CaSS_T_mx, self.CaSS_T_res).to(self.device).to(self.dtype)
        CaSS_tab = torch.zeros((CaSS.shape[0], self.CaSS_NROWS)).to(self.device).to(self.dtype)

        FCaSS_inf = ((0.6/(1.+((CaSS/0.05)*(CaSS/0.05))))+0.4)
        tau_FCaSS = ((80./(1.+((CaSS/0.05)*(CaSS/0.05))))+2.)
        CaSS_tab[:, self.FCaSS_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_FCaSS)))
        FCaSS_rush_larsen_C = (torch.expm1(((-self.dt)/tau_FCaSS)))
        CaSS_tab[:, self.FCaSS_rush_larsen_A_idx] = ((-FCaSS_inf)*FCaSS_rush_larsen_C)
        self.CaSS_tab = CaSS_tab


        # construct V Lookup Table
        V = torch.arange(self.V_T_mn, self.V_T_mx, self.V_T_res).to(self.device).to(self.dtype)
        V_tab = torch.zeros((V.shape[0], self.V_NROWS)).to(self.device).to(self.dtype)

        D_inf = (1./(1.+(torch.exp((((-8.+self.D_CaL_off)-(V))/7.5)))))
        F2_inf = ((0.67/(1.+(torch.exp(((V+35.)/7.)))))+0.33)
        F_inf = (1./(1.+(torch.exp(((V+20.)/7.)))))
        H_inf = (1./((1.+(torch.exp(((V+71.55)/7.43))))*(1.+(torch.exp(((V+71.55)/7.43))))))
        M_inf = (1./((1.+(torch.exp(((-56.86-(V))/9.03))))*(1.+(torch.exp(((-56.86-(V))/9.03))))))
        R_inf = (1./(1.+(torch.exp(((20.-(V))/6.)))))
        S_inf = (1./(1.+(torch.exp(((V+20.)/5.)))))
        Xr1_inf = (1./(1.+(torch.exp(((-26.-(V))/7.)))))
        Xr2_inf = (1./(1.+(torch.exp(((V-((-88.+self.xr2_off)))/24.)))))
        Xs_inf = (1./(1.+(torch.exp((((-self.vHalfXs)-(V))/14.)))))
        V_tab[:, self.a2_idx] = (0.25*(torch.exp(((2.*(V-(15.)))*F_RT))))
        aa_D = ((1.4/(1.+(torch.exp(((-35.-(V))/13.)))))+0.25)
        aa_F = (1102.5*(torch.exp((((-(V+27.))*(V+27.))/225.))))
        aa_F2 = (562.*(torch.exp((((-(V+27.))*(V+27.))/240.))))
        aa_H = torch.where((V>=-40.), 0., (0.057*(torch.exp(((-(V+80.))/6.8)))))
        aa_J = torch.where((V>=-40.), 0., ((((-2.5428e4*(torch.exp((0.2444*V))))-((6.948e-6*(torch.exp((-0.04391*V))))))*(V+37.78))/(1.+(torch.exp((0.311*(V+79.23)))))))
        aa_M = (1./(1.+(torch.exp(((-60.-(V))/5.)))))
        aa_Xr1 = (450./(1.+(torch.exp(((-45.-(V))/10.)))))
        aa_Xr2 = (3./(1.+(torch.exp(((-60.-(V))/20.)))))
        aa_Xs = (1400./(torch.sqrt((1.+(torch.exp(((5.-(V))/6.)))))))
        bb_D = (1.4/(1.+(torch.exp(((V+5.)/5.)))))
        bb_F = (200./(1.+(torch.exp(((13.-(V))/10.)))))
        bb_F2 = (31./(1.+(torch.exp(((25.-(V))/10.)))))
        bb_H = torch.where((V>=-40.), (0.77/(0.13*(1.+(torch.exp(((-(V+10.66))/11.1)))))), ((2.7*(torch.exp((0.079*V))))+(3.1e5*(torch.exp((0.3485*V))))))
        bb_J = torch.where((V>=-40.), ((0.6*(torch.exp((0.057*V))))/(1.+(torch.exp((-0.1*(V+32.)))))), ((0.02424*(torch.exp((-0.01052*V))))/(1.+(torch.exp((-0.1378*(V+40.14)))))))
        bb_M = ((0.1/(1.+(torch.exp(((V+35.)/5.)))))+(0.10/(1.+(torch.exp(((V-(50.))/200.))))))
        bb_Xr1 = (6./(1.+(torch.exp(((V-(-30.))/11.5)))))
        bb_Xr2 = (1.12/(1.+(torch.exp(((V-(60.))/20.)))))
        bb_Xs = (1./(1.+(torch.exp(((V-(35.))/15.)))))
        cc_D = (1./(1.+(torch.exp(((50.-(V))/20.)))))
        cc_F = ((180./(1.+(torch.exp(((V+30.)/10.)))))+20.)
        cc_F2 = (80./(1.+(torch.exp(((V+30.)/10.)))))
        den = (pmf_INaCa/(1.+(self.ksat*(torch.exp((((self.n-(1.))*V)*F_RT))))))
        V_tab[:, self.rec_iNaK_idx] = (1./((1.+(0.1245*(torch.exp(((-0.1*V)*F_RT)))))+(0.0353*(torch.exp(((-V)*F_RT))))))
        V_tab[:, self.rec_ipK_idx] = (1./(1.+(torch.exp(((25.-(V))/5.98)))))
        tau_R = ((9.5*(torch.exp((((-(V+40.))*(V+40.))/1800.))))+0.8)
        tau_S = ((1000.*(torch.exp((((-(V+67.))*(V+67.))/1000.))))+8.) if self.cell_type=="ENDO" else (((85.*(torch.exp((((-(V+45.))*(V+45.))/320.))))+(5./(1.+(torch.exp(((V-(20.))/5.))))))+3.)
        V_tab[:, self.INaCa_A_idx] = ((den*self.Cao)*(torch.exp(((self.n*V)*F_RT))))
        V_tab[:, self.INaCa_B_idx] = (((den*(torch.exp((((self.n-(1.))*V)*F_RT))))*Nao3)*2.5)
        J_inf = H_inf
        V_tab[:, self.R_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_R)))
        R_rush_larsen_C = (torch.expm1(((-self.dt)/tau_R)))
        V_tab[:, self.S_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_S)))
        S_rush_larsen_C = (torch.expm1(((-self.dt)/tau_S)))
        tau_D = ((aa_D*bb_D)+cc_D)
        tau_F2 = ((aa_F2+bb_F2)+cc_F2)
        tau_F_factor = ((aa_F+bb_F)+cc_F)
        tau_H = (1.0/(aa_H+bb_H))
        tau_J = (1.0/(aa_J+bb_J))
        tau_M = (aa_M*bb_M)
        tau_Xr1 = (aa_Xr1*bb_Xr1)
        tau_Xr2 = (aa_Xr2*bb_Xr2)
        tau_Xs = ((aa_Xs*bb_Xs)+80.)
        V_tab[:, self.D_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_D)))
        D_rush_larsen_C = (torch.expm1(((-self.dt)/tau_D)))
        V_tab[:, self.F2_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_F2)))
        F2_rush_larsen_C = (torch.expm1(((-self.dt)/tau_F2)))
        V_tab[:, self.H_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_H)))
        H_rush_larsen_C = (torch.expm1(((-self.dt)/tau_H)))
        V_tab[:, self.J_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_J)))
        J_rush_larsen_C = (torch.expm1(((-self.dt)/tau_J)))
        V_tab[:, self.M_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_M)))
        M_rush_larsen_C = (torch.expm1(((-self.dt)/tau_M)))
        V_tab[:, self.R_rush_larsen_A_idx] = ((-R_inf)*R_rush_larsen_C)
        V_tab[:, self.S_rush_larsen_A_idx] = ((-S_inf)*S_rush_larsen_C)
        V_tab[:, self.Xr1_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_Xr1)))
        Xr1_rush_larsen_C = (torch.expm1(((-self.dt)/tau_Xr1)))
        V_tab[:, self.Xr2_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_Xr2)))
        Xr2_rush_larsen_C = (torch.expm1(((-self.dt)/tau_Xr2)))
        V_tab[:, self.Xs_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_Xs)))
        Xs_rush_larsen_C = (torch.expm1(((-self.dt)/tau_Xs)))
        tau_F = torch.where((V>0.), (tau_F_factor*self.scl_tau_f), tau_F_factor)
        V_tab[:, self.D_rush_larsen_A_idx] = ((-D_inf)*D_rush_larsen_C)
        V_tab[:, self.F2_rush_larsen_A_idx] = ((-F2_inf)*F2_rush_larsen_C)
        V_tab[:, self.F_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_F)))
        F_rush_larsen_C = (torch.expm1(((-self.dt)/tau_F)))
        V_tab[:, self.H_rush_larsen_A_idx] = ((-H_inf)*H_rush_larsen_C)
        V_tab[:, self.J_rush_larsen_A_idx] = ((-J_inf)*J_rush_larsen_C)
        V_tab[:, self.M_rush_larsen_A_idx] = ((-M_inf)*M_rush_larsen_C)
        V_tab[:, self.Xr1_rush_larsen_A_idx] = ((-Xr1_inf)*Xr1_rush_larsen_C)
        V_tab[:, self.Xr2_rush_larsen_A_idx] = ((-Xr2_inf)*Xr2_rush_larsen_C)
        V_tab[:, self.Xs_rush_larsen_A_idx] = ((-Xs_inf)*Xs_rush_larsen_C)
        V_tab[:, self.F_rush_larsen_A_idx] = ((-F_inf)*F_rush_larsen_C)

        self.V_tab = V_tab

        # construct VEk Lookup Table
        VEk = torch.arange(self.VEk_T_mn, self.VEk_T_mx, self.VEk_T_res).to(self.device).to(self.dtype)
        VEk_tab = torch.zeros((VEk.shape[0], self.VEk_NROWS)).to(self.device).to(self.dtype)

        a_K1 = (0.1/(1.+(torch.exp((0.06*(VEk-(200.)))))))
        b_K1 = (((3.*(torch.exp((0.0002*(VEk+100.)))))+(torch.exp((0.1*(VEk-(10.))))))/(1.+(torch.exp((-0.5*VEk)))))
        VEk_tab[:, self.rec_iK1_idx] = (a_K1/(a_K1+b_K1))

        self.VEk_tab = VEk_tab

    def initialize(self, n_nodes: int):
        self.construct_tables()
        
        V = torch.full((n_nodes,), self.V_init, device=self.device, dtype=self.dtype)
    
        self.GCaL = torch.full((n_nodes,), self.GCaL_init, device=self.device, dtype=self.dtype)
        self.GKr = torch.full((n_nodes,), self.GKr_init, device=self.device, dtype=self.dtype)
        self.GKs = torch.full((n_nodes,), self.GKs_init, device=self.device, dtype=self.dtype)
        self.Gto = torch.full((n_nodes,), self.Gto_init, device=self.device, dtype=self.dtype)
        self.CaSR = torch.full((n_nodes,), self.CaSR_init, device=self.device, dtype=self.dtype)
        self.CaSS = torch.full((n_nodes,), self.CaSS_init, device=self.device, dtype=self.dtype)
        self.Cai = torch.full((n_nodes,), self.Cai_init, device=self.device, dtype=self.dtype)
        # self.Cai *= 1e3
        self.F = torch.full((n_nodes,), self.F_init, device=self.device, dtype=self.dtype)
        self.F2 = torch.full((n_nodes,), self.F2_init, device=self.device, dtype=self.dtype)
        self.FCaSS = torch.full((n_nodes,), self.FCaSS_init, device=self.device, dtype=self.dtype)
        self.H = torch.full((n_nodes,), self.H_init, device=self.device, dtype=self.dtype)
        self.J = torch.full((n_nodes,), self.J_init, device=self.device, dtype=self.dtype)
        self.Ki = torch.full((n_nodes,), self.Ki_init, device=self.device, dtype=self.dtype)
        self.M = torch.full((n_nodes,), self.M_init, device=self.device, dtype=self.dtype)
        self.Nai = torch.full((n_nodes,), self.Nai_init, device=self.device, dtype=self.dtype)
        self.R = torch.full((n_nodes,), self.R_init, device=self.device, dtype=self.dtype)
        self.R_ = torch.full((n_nodes,), self.R__init, device=self.device, dtype=self.dtype)
        self.S = torch.full((n_nodes,), self.S_init, device=self.device, dtype=self.dtype)
        self.Xr1 = torch.full((n_nodes,), self.Xr1_init, device=self.device, dtype=self.dtype)
        self.Xr2 = torch.full((n_nodes,), self.Xr2_init, device=self.device, dtype=self.dtype)
        self.Xs = torch.full((n_nodes,), self.Xs_init, device=self.device, dtype=self.dtype)
        self.D = torch.full((n_nodes,), self.d_init, device=self.device, dtype=self.dtype)   # (1./(1.+(torch.exp((((-8.+self.D_CaL_off)-(V))/7.5)))))

        return V

    def differentiate(self, V):
        # self.Cai *= 1e-3

        # Define the constants that depend on the parameters.
        RTONF = ((self.Rconst*self.T)/self.Fconst)
        inverseVcF = (1./(self.Vc*self.Fconst))
        inverseVcF2 = (1./((2.*self.Vc)*self.Fconst))
        inverseVssF2 = (1./((2.*self.Vss)*self.Fconst))
        pmf_INaK = (self.knak*(self.Ko/(self.Ko+self.KmK)))
        sqrt_Ko = (sqrt((self.Ko/5.4)))
        F_RT = (1./RTONF)
        invVcF_Cm = (inverseVcF*self.CAPACITANCE)

        V_row = self.interpolate(V, self.V_tab, self.V_T_mn, self.V_T_mx, self.V_T_res, self.V_T_step, self.V_T_mx_idx)
        CaSS_row = self.interpolate(self.CaSS, self.CaSS_tab, self.CaSS_T_mn, self.CaSS_T_mx, self.CaSS_T_res, self.CaSS_T_step, self.CaSS_T_mx_idx)
        
        Eca = ((0.5*RTONF)*(torch.log((self.Cao/self.Cai))))
        Ek = (RTONF*(torch.log((self.Ko/self.Ki))))
        Eks = (RTONF*(torch.log(((self.Ko+(self.pKNa*self.Nao))/(self.Ki+(self.pKNa*self.Nai))))))
        Ena = (RTONF*(torch.log((self.Nao/self.Nai))))
        IpCa = ((self.GpCa*self.Cai)/(self.KpCa+self.Cai))
        a1 = ((((self.GCaL*self.Fconst)*F_RT)*4.)*torch.where((V==15.), ((1./2.)*F_RT), ((V-(15.))/(torch.expm1(((2.*(V-(15.)))*F_RT))))))
        ICaL_A = (a1*V_row[:, self.a2_idx])
        ICaL_B = (a1*self.Cao)
        IKr = ((((self.GKr*sqrt_Ko)*self.Xr1)*self.Xr2)*(V-(Ek)))
        IKs = (((self.GKs*self.Xs)*self.Xs)*(V-(Eks)))
        INa = ((((((self.GNa*self.M)*self.M)*self.M)*self.H)*self.J)*(V-(Ena)))
        INaK = ((pmf_INaK*(self.Nai/(self.Nai+self.KmNa)))*V_row[:, self.rec_iNaK_idx])
        IbCa = (self.GbCa*(V-(Eca)))
        IbNa = (self.GbNa*(V-(Ena)))
        IpK = ((self.GpK*V_row[:, self.rec_ipK_idx])*(V-(Ek)))
        Ito = (((self.Gto*self.R)*self.S)*(V-(Ek)))
        VEk = (V-(Ek))

        VEk_row = self.interpolate(VEk, self.VEk_tab, self.VEk_T_mn, self.VEk_T_mx, self.VEk_T_res, self.VEk_T_step, self.VEk_T_mx_idx)
        ICaL = ((((self.D*self.F)*self.F2)*self.FCaSS)*((ICaL_A*self.CaSS)-(ICaL_B)))
        INaCa = ((((V_row[:, self.INaCa_A_idx]*self.Nai)*self.Nai)*self.Nai)-((V_row[:, self.INaCa_B_idx]*self.Cai)))
        IK1 = ((self.GK1*VEk_row[:, self.rec_iK1_idx])*(V-(Ek)))
        Iion = (((((((((((IKr+IKs)+IK1)+Ito)+INa)+IbNa)+ICaL)+IbCa)+INaK)+INaCa)+IpCa)+IpK)

        # Complete Forward Euler Update
        Ileak = (self.Vleak*(self.CaSR-(self.Cai)))
        Iup = (self.Vmaxup/(1.+((self.Kup*self.Kup)/(self.Cai*self.Cai))))
        Ixfer = (self.Vxfer*(self.CaSS-(self.Cai)))
        diff_Ki = ((-(((((IK1+Ito)+IKr)+IKs)-((2.*INaK)))+IpK))*invVcF_Cm)
        diff_Nai = ((-(((INa+IbNa)+(3.*INaK))+(3.*INaCa)))*invVcF_Cm)
        kCaSR = (self.maxsr-(((self.maxsr-(self.minsr))/(1.+((self.EC/self.CaSR)*(self.EC/self.CaSR))))))
        diff_Cai = (((Ixfer-(((((IbCa+IpCa)-((2.*INaCa)))*inverseVcF2)*self.CAPACITANCE)))-(((Iup-(Ileak))*(self.Vsr/self.Vc))))/(1.+(((self.Bufc*self.Kbufc)/(self.Kbufc+self.Cai))/(self.Kbufc+self.Cai))))
        diff_R_ = ((self.k4*(1.-(self.R_)))-((((self.k2_*kCaSR)*self.CaSS)*self.R_)))
        k1 = (self.k1_/kCaSR)
        O = (((k1*CaSS_row[:, self.CaSS2_idx])*self.R_)/(self.k3+(k1*CaSS_row[:, self.CaSS2_idx])))
        Irel = ((self.Vrel*O)*(self.CaSR-(self.CaSS)))
        diff_CaSR = (((Iup-(Irel))-(Ileak))/(1.+(((self.Bufsr*self.Kbufsr)/(self.CaSR+self.Kbufsr))/(self.CaSR+self.Kbufsr))))
        diff_CaSS = (((((-Ixfer)*(self.Vc/self.Vss))+(Irel*(self.Vsr/self.Vss)))+(((-ICaL)*inverseVssF2)*self.CAPACITANCE))/(1.+(((self.Bufss*self.Kbufss)/(self.CaSS+self.Kbufss))/(self.CaSS+self.Kbufss))))
        self.CaSR = self.CaSR+diff_CaSR*self.dt
        self.CaSS = self.CaSS+diff_CaSS*self.dt
        self.Cai = self.Cai+diff_Cai*self.dt
        self.Ki = self.Ki+diff_Ki*self.dt
        self.Nai = self.Nai+diff_Nai*self.dt
        self.R_ = self.R_+diff_R_*self.dt
        
        # Complete Rush Larsen Update
        FCaSS_rush_larsen_B = CaSS_row[:, self.FCaSS_rush_larsen_B_idx]
        R_rush_larsen_B = V_row[:, self.R_rush_larsen_B_idx]
        S_rush_larsen_B = V_row[:, self.S_rush_larsen_B_idx]
        D_rush_larsen_B = V_row[:, self.D_rush_larsen_B_idx]
        F2_rush_larsen_B = V_row[:, self.F2_rush_larsen_B_idx]
        FCaSS_rush_larsen_A = CaSS_row[:, self.FCaSS_rush_larsen_A_idx]
        H_rush_larsen_B = V_row[:, self.H_rush_larsen_B_idx]
        J_rush_larsen_B = V_row[:, self.J_rush_larsen_B_idx]
        M_rush_larsen_B = V_row[:, self.M_rush_larsen_B_idx]
        R_rush_larsen_A = V_row[:, self.R_rush_larsen_A_idx]
        S_rush_larsen_A = V_row[:, self.S_rush_larsen_A_idx]
        Xr1_rush_larsen_B = V_row[:, self.Xr1_rush_larsen_B_idx]
        Xr2_rush_larsen_B = V_row[:, self.Xr2_rush_larsen_B_idx]
        Xs_rush_larsen_B = V_row[:, self.Xs_rush_larsen_B_idx]
        D_rush_larsen_A = V_row[:, self.D_rush_larsen_A_idx]
        F2_rush_larsen_A = V_row[:, self.F2_rush_larsen_A_idx]
        F_rush_larsen_B = V_row[:, self.F_rush_larsen_B_idx]
        H_rush_larsen_A = V_row[:, self.H_rush_larsen_A_idx]
        J_rush_larsen_A = V_row[:, self.J_rush_larsen_A_idx]
        M_rush_larsen_A = V_row[:, self.M_rush_larsen_A_idx]
        Xr1_rush_larsen_A = V_row[:, self.Xr1_rush_larsen_A_idx]
        Xr2_rush_larsen_A = V_row[:, self.Xr2_rush_larsen_A_idx]
        Xs_rush_larsen_A = V_row[:, self.Xs_rush_larsen_A_idx]
        F_rush_larsen_A = V_row[:, self.F_rush_larsen_A_idx]
        self.D = D_rush_larsen_A+D_rush_larsen_B*self.D
        self.F = F_rush_larsen_A+F_rush_larsen_B*self.F
        self.F2 = F2_rush_larsen_A+F2_rush_larsen_B*self.F2
        self.FCaSS = FCaSS_rush_larsen_A+FCaSS_rush_larsen_B*self.FCaSS
        self.H = H_rush_larsen_A+H_rush_larsen_B*self.H
        self.J = J_rush_larsen_A+J_rush_larsen_B*self.J
        self.M = M_rush_larsen_A+M_rush_larsen_B*self.M
        self.R = R_rush_larsen_A+R_rush_larsen_B*self.R
        self.S = S_rush_larsen_A+S_rush_larsen_B*self.S
        self.Xr1 = Xr1_rush_larsen_A+Xr1_rush_larsen_B*self.Xr1
        self.Xr2 = Xr2_rush_larsen_A+Xr2_rush_larsen_B*self.Xr2
        self.Xs = Xs_rush_larsen_A+Xs_rush_larsen_B*self.Xs

        # self.Cai *= 1e3

        return -Iion


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    dt = 0.02
    stimulus = 20
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    ionic = TenTusscherPanfilov(cell_type="EPI", 
                                dt=dt, 
                                device=device, 
                                dtype=torch.float32)
    V = ionic.initialize(n_nodes=1)


    V_list = []
    Cai_list = []
    CaSS_list = []
    dV_list = []

    ctime = 0.0
    for _ in range(int(1000/dt)):
        V_list.append([ctime, V.item()])
        Cai_list.append([ctime, ionic.Cai.item()])
        CaSS_list.append([ctime, ionic.CaSS.item()])

        dV = ionic.differentiate(V)
        V = V + dt * dV
        ctime += dt
        if ctime >= 0 and ctime <= (0+2.0): 
            V = V + dt * stimulus
        dV_list.append([ctime, dV.item()])
    
    plt.figure()
    V_list = np.array(V_list)    
    plt.plot(V_list[:, 0], V_list[:, 1])
    plt.savefig("V_TenTusscherPanfilov.png")

    plt.figure()
    dV_list = np.array(dV_list)    
    plt.plot(dV_list[:, 0], dV_list[:, 1])
    plt.savefig("dV.png")

    plt.figure()
    Cai_list = np.array(Cai_list)    
    plt.plot(Cai_list[:,0], Cai_list[:,1])
    plt.savefig("Cai.png")

    plt.figure()
    CaSS_list = np.array(CaSS_list)    
    plt.plot(CaSS_list[:,0], CaSS_list[:,1])
    plt.savefig("CaSS.png")