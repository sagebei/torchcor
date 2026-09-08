import torch
import torchcor as tc
from math import exp, log, sqrt, expm1
from typing import Optional, List


@torch.jit.script
class OHaraRudy:
    def __init__(self, 
                 dt: float, 
                 region_ids: Optional[List[int]] = None, 
                 cell_type: str = "ENDO", 
                 ina: str = "ORdINa", 
                 formulation: str = "ORIGINAL", 
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float64):
        
        self.name = "OHaraRudy"
        self.dt = dt
        self.region_ids = region_ids
        self.node_indices = torch.tensor([0])
        self.device = device
        self.dtype = dtype

        self.cell_type = "ENDO" if cell_type is None else cell_type

        # Constants
        self.L = 0.01
        self.rad = 0.0011
        self.Ageo = ((((2.*3.14)*self.rad)*self.rad)+(((2.*3.14)*self.rad)*self.L))
        self.Acap = (2.*self.Ageo)
        self.Aff = 0.6
        self.Afs = (1.0-(self.Aff))
        self.Ahf = 0.99
        self.Ahs = (1.0-(self.Ahf))
        self.BSLmax = 1.124
        self.BSRmax = 0.047
        self.CLERX = 1.
        self.CaMKo = 0.05
        self.CaMKt_init = 0.0
        self.Cai_init = 1.0e-4
        self.ENDO = 0.
        self.EPI = 1.
        self.F = 96485.0
        self.Jrelnp_init = 0.0
        self.Jrelp_init = 0.0
        self.KKo = 0.3582
        self.KNao0 = 27.78
        self.Khp = 1.698e-7
        self.Ki_init = 145.0
        self.Kki = 0.5
        self.KmBSL = 0.0087
        self.KmBSR = 0.00087
        self.KmCaAct_i = 150.0e-6
        self.KmCaAct_ss = 150.0e-6
        self.KmCaM = 0.0015
        self.KmCaMK = 0.15
        self.Kmgatp = 1.698e-7
        self.Kmn = 0.002
        self.Knai0 = 9.073
        self.Knap = 224.0
        self.Kxkur = 292.0
        self.MCELL = 2.
        self.MgADP = 0.05
        self.MgATP = 9.8
        self.Nai_init = 7.0
        self.ORIGINAL = 0.
        self.ORdINa = 1.
        self.PCab = 2.5e-8
        self.PKNa = 0.01833
        self.PNab = 3.75e-10
        self.R = 8314.0
        self.T = 310.0
        self.TT2INa = 0.
        self.V_init = -87.
        self.k2p = 687.2
        self.a2 = self.k2p
        self.k4p = 639.0
        self.a4 = (((self.k4p*self.MgATP)/self.Kmgatp)/(1.0+(self.MgATP/self.Kmgatp)))
        self.aCaMK = 0.05
        self.a_init = 0.0
        self.bt = 4.75
        self.a_rel = (0.5*self.bt)
        self.btp = (1.25*self.bt)
        self.a_relp = (0.5*self.btp)
        self.ap_init = 0.0
        self.k1m = 182.4
        self.b1 = (self.k1m*self.MgADP)
        self.bCaMK = 0.00068
        self.cansr_init = 1.2
        self.csqnmax = 10.0
        self.d_init = 0.0
        self.delta = -0.1550
        self.eP = 4.2
        self.fcaf_init = 1.0
        self.fcafp_init = 1.0
        self.fcas_init = 1.0
        self.ff_init = 1.0
        self.ffp_init = 1.0
        self.fs_init = 1.0
        self.hL_init = 1.0
        self.hLp_init = 1.0
        self.hf_init = 1.0
        self.hs_init = 1.0
        self.hsp_init = 1.0
        self.iF_init = 1.0
        self.iFp_init = 1.0
        self.iS_init = 1.0
        self.iSp_init = 1.0
        self.j_init = 1.0
        self.jca_init = 1.0
        self.jp_init = 1.0
        self.k1p = 949.5
        self.kCaoff = 5.0e3
        self.k2_i = self.kCaoff
        self.k2_ss = self.kCaoff
        self.k2m = 39.4
        self.k2n = 1000.0
        self.k3m = 79300.0
        self.k3p = 1899.0
        self.k4m = 40.0
        self.k5_i = self.kCaoff
        self.k5_ss = self.kCaoff
        self.kCaon = 1.5e6
        self.kasymm = 12.5
        self.kmcmdn = 0.00238
        self.kmcsqn = 0.8
        self.kmtrpn = 0.0005
        self.kna1 = 15.0
        self.kna2 = 5.0
        self.kna3 = 88.12
        self.mL_init = 0.0
        self.m_init = 0.0
        self.nca_init = 0.0
        self.qca = 0.1670
        self.qna = 0.5224
        self.scale_GNa = 0.37
        self.tau_hL = 200.0
        self.tau_hLp = (3.0*self.tau_hL)
        self.tau_jca = 75.0
        self.trpnmax = 0.07
        self.vcell = ((((1000.*3.14)*self.rad)*self.rad)*self.L)
        self.vjsr = (0.0048*self.vcell)
        self.vmyo = (0.68*self.vcell)
        self.vnsr = (0.0552*self.vcell)
        self.vss = (0.02*self.vcell)
        self.wca = 6.0e4
        self.wna = 6.0e4
        self.wnaca = 5.0e3
        self.xk1_init = 1.0
        self.xrf_init = 0.0
        self.xrs_init = 0.0
        self.xs1_init = 0.0
        self.xs2_init = 0.0
        self.zca = 2.0
        self.zk = 1.0
        self.zna = 1.0

        # Parameters
        self.Cao = 1.8
        self.GpCa = 0.0005
        self.INa_Type = self.TT2INa if ina == "TT2INa" else self.ORdINa
        self.Ko = 5.4
        self.Nao = 140.0
        self.celltype = 1. if cell_type == "EPI" else 2. if cell_type == "MCELL" else 0.
        self.factorICaK = 1.0
        self.factorICaL = 1.0
        self.factorICaNa = 1.0
        self.factorINaCa = 1.
        self.factorINaCass = 1.0
        self.factorINaK = 1.
        self.factorIbCa = 1.
        self.factorIbNa = 1.
        self.jrel_stiff_const = 0.005
        self.modelformulation = self.CLERX if formulation == "CLERX" else self.ORIGINAL
        self.scale_tau_m = 0.3
        self.shift_m_inf = -8.0
        self.vHalfXs = 11.60
        self.GK1 = ((0.1908*1.2) if (self.celltype==self.EPI) else ((0.1908*1.3) if (self.celltype==self.MCELL) else 0.1908))
        self.GKb = ((0.003*0.6) if (self.celltype==self.EPI) else 0.003)
        self.GKr = ((0.046*1.3) if (self.celltype==self.EPI) else ((0.046*0.8) if (self.celltype==self.MCELL) else 0.046))
        self.GKs = ((0.0034*1.4) if (self.celltype==self.EPI) else 0.0034)
        self.GNa = (14.838 if (self.INa_Type==self.TT2INa) else (75.*self.scale_GNa))
        self.GNaCa = ((0.0008*1.1) if (self.celltype==self.EPI) else ((0.0008*1.4) if (self.celltype==self.MCELL) else 0.0008))
        self.GNaL = ((0.0075*0.6) if (self.celltype==self.EPI) else 0.0075)
        self.Gto = ((0.02*4.0) if (self.celltype==self.EPI) else ((0.02*4.0) if (self.celltype==self.MCELL) else 0.02))

        # Initial values that initialize_sv derives rather than defines
        self.cajsr_init = self.cansr_init
        self.cass_init = self.Cai_init
        self.kss_init = self.Ki_init
        self.nass_init = self.Nai_init
        hTT2_inf = (1./((1.+(exp(((self.V_init+71.55)/7.43))))*(1.+(exp(((self.V_init+71.55)/7.43))))))
        self.hTT2_init = hTT2_inf
        mORd_inf = (1.0/(1.0+(exp(((-((self.V_init+39.57)-(self.shift_m_inf)))/9.871)))))
        self.mORd_init = mORd_inf
        mTT2_inf = (1./((1.+(exp(((-56.86-(self.V_init))/9.03))))*(1.+(exp(((-56.86-(self.V_init))/9.03))))))
        self.mTT2_init = mTT2_inf
        jTT2_inf = hTT2_inf
        self.jTT2_init = jTT2_inf

        # Cai_TableIndex
        self.Bcai_idx = 0
        self.IpCa_idx = 1
        self.Jupnp_idx = 2
        self.Jupp_idx = 3
        self.KsCa_idx = 4
        self.allo_i_idx = 5
        self.Cai_NROWS = 6

        # V_TableIndex
        self.Afcaf_idx = 0
        self.Afcas_idx = 1
        self.AiF_idx = 2
        self.AiS_idx = 3
        self.Axrf_idx = 4
        self.Axrs_idx = 5
        self.Knai_idx = 6
        self.a3_idx = 7
        self.a_rush_larsen_A_idx = 8
        self.a_rush_larsen_B_idx = 9
        self.ap_rush_larsen_A_idx = 10
        self.ap_rush_larsen_B_idx = 11
        self.b2_idx = 12
        self.d_rush_larsen_A_idx = 13
        self.d_rush_larsen_B_idx = 14
        self.exp_2vfrt_idx = 15
        self.exp_vfrt_idx = 16
        self.fcaf_rush_larsen_A_idx = 17
        self.fcaf_rush_larsen_B_idx = 18
        self.fcafp_rush_larsen_A_idx = 19
        self.fcafp_rush_larsen_B_idx = 20
        self.fcas_rush_larsen_A_idx = 21
        self.fcas_rush_larsen_B_idx = 22
        self.ff_rush_larsen_A_idx = 23
        self.ff_rush_larsen_B_idx = 24
        self.ffp_rush_larsen_A_idx = 25
        self.ffp_rush_larsen_B_idx = 26
        self.fs_rush_larsen_A_idx = 27
        self.fs_rush_larsen_B_idx = 28
        self.hL_rush_larsen_A_idx = 29
        self.hLp_rush_larsen_A_idx = 30
        self.hTT2_rush_larsen_A_idx = 31
        self.hTT2_rush_larsen_B_idx = 32
        self.hca_idx = 33
        self.hf_rush_larsen_A_idx = 34
        self.hf_rush_larsen_B_idx = 35
        self.hna_idx = 36
        self.hs_rush_larsen_A_idx = 37
        self.hs_rush_larsen_B_idx = 38
        self.hsp_rush_larsen_A_idx = 39
        self.hsp_rush_larsen_B_idx = 40
        self.iF_rush_larsen_A_idx = 41
        self.iF_rush_larsen_B_idx = 42
        self.iFp_rush_larsen_A_idx = 43
        self.iFp_rush_larsen_B_idx = 44
        self.iS_rush_larsen_A_idx = 45
        self.iS_rush_larsen_B_idx = 46
        self.iSp_rush_larsen_A_idx = 47
        self.iSp_rush_larsen_B_idx = 48
        self.jTT2_rush_larsen_A_idx = 49
        self.jTT2_rush_larsen_B_idx = 50
        self.j_rush_larsen_A_idx = 51
        self.j_rush_larsen_B_idx = 52
        self.jca_rush_larsen_A_idx = 53
        self.jp_rush_larsen_A_idx = 54
        self.jp_rush_larsen_B_idx = 55
        self.k3_i_idx = 56
        self.k3_ss_idx = 57
        self.k3pp_i_idx = 58
        self.k3pp_ss_idx = 59
        self.k8_i_idx = 60
        self.k8_ss_idx = 61
        self.mL_rush_larsen_A_idx = 62
        self.mL_rush_larsen_B_idx = 63
        self.mORd_rush_larsen_A_idx = 64
        self.mORd_rush_larsen_B_idx = 65
        self.mTT2_rush_larsen_A_idx = 66
        self.mTT2_rush_larsen_B_idx = 67
        self.rk1_idx = 68
        self.rkr_idx = 69
        self.vffrt_expm1_2vfrt_idx = 70
        self.vffrt_expm1_vfrt_idx = 71
        self.xk1_rush_larsen_A_idx = 72
        self.xk1_rush_larsen_B_idx = 73
        self.xkb_idx = 74
        self.xrf_rush_larsen_A_idx = 75
        self.xrf_rush_larsen_B_idx = 76
        self.xrs_rush_larsen_A_idx = 77
        self.xrs_rush_larsen_B_idx = 78
        self.xs1_rush_larsen_A_idx = 79
        self.xs1_rush_larsen_B_idx = 80
        self.xs2_rush_larsen_A_idx = 81
        self.xs2_rush_larsen_B_idx = 82
        self.V_NROWS = 83

        # Cai_TableParam  (float32, as LUT_alloc stores them)
        self.Cai_T_mn = 9.999999974752427e-07
        self.Cai_T_mx = 0.009999999776482582
        self.Cai_T_res = 9.999999974752427e-07
        self.Cai_T_step = 1000000.0
        self.Cai_T_mn_ind = 1
        self.Cai_T_mx_ind = 10000

        # V_TableParam  (float32, as LUT_alloc stores them)
        self.V_T_mn = -1000.0
        self.V_T_mx = 1000.0
        self.V_T_res = 0.009999999776482582
        self.V_T_step = 100.0
        self.V_T_mn_ind = -100000
        self.V_T_mx_ind = 100000

        # 2 lookup tables
        self.Cai_tab = torch.tensor([1.0])
        self.V_tab = torch.tensor([1.0])

        # 43 states variables
        self.CaMKt = torch.tensor([self.CaMKt_init])
        self.Cai = torch.tensor([self.Cai_init])
        self.Jrelnp = torch.tensor([self.Jrelnp_init], dtype=self.dtype)
        self.Jrelp = torch.tensor([self.Jrelp_init], dtype=self.dtype)
        self.Ki = torch.tensor([self.Ki_init])
        self.Nai = torch.tensor([self.Nai_init])
        self.a = torch.tensor([self.a_init], dtype=self.dtype)
        self.ap = torch.tensor([self.ap_init], dtype=self.dtype)
        self.cajsr = torch.tensor([self.cajsr_init])
        self.cansr = torch.tensor([self.cansr_init])
        self.cass = torch.tensor([self.cass_init])
        self.d = torch.tensor([self.d_init], dtype=self.dtype)
        self.fcaf = torch.tensor([self.fcaf_init], dtype=self.dtype)
        self.fcafp = torch.tensor([self.fcafp_init], dtype=self.dtype)
        self.fcas = torch.tensor([self.fcas_init], dtype=self.dtype)
        self.ff = torch.tensor([self.ff_init], dtype=self.dtype)
        self.ffp = torch.tensor([self.ffp_init], dtype=self.dtype)
        self.fs = torch.tensor([self.fs_init], dtype=self.dtype)
        self.hL = torch.tensor([self.hL_init], dtype=self.dtype)
        self.hLp = torch.tensor([self.hLp_init], dtype=self.dtype)
        self.hTT2 = torch.tensor([self.hTT2_init], dtype=self.dtype)
        self.hf = torch.tensor([self.hf_init], dtype=self.dtype)
        self.hs = torch.tensor([self.hs_init], dtype=self.dtype)
        self.hsp = torch.tensor([self.hsp_init], dtype=self.dtype)
        self.iF = torch.tensor([self.iF_init], dtype=self.dtype)
        self.iFp = torch.tensor([self.iFp_init], dtype=self.dtype)
        self.iS = torch.tensor([self.iS_init], dtype=self.dtype)
        self.iSp = torch.tensor([self.iSp_init], dtype=self.dtype)
        self.j = torch.tensor([self.j_init], dtype=self.dtype)
        self.jTT2 = torch.tensor([self.jTT2_init], dtype=self.dtype)
        self.jca = torch.tensor([self.jca_init], dtype=self.dtype)
        self.jp = torch.tensor([self.jp_init], dtype=self.dtype)
        self.kss = torch.tensor([self.kss_init])
        self.mL = torch.tensor([self.mL_init], dtype=self.dtype)
        self.mORd = torch.tensor([self.mORd_init], dtype=self.dtype)
        self.mTT2 = torch.tensor([self.mTT2_init], dtype=self.dtype)
        self.nass = torch.tensor([self.nass_init])
        self.nca = torch.tensor([self.nca_init])
        self.xk1 = torch.tensor([self.xk1_init], dtype=self.dtype)
        self.xrf = torch.tensor([self.xrf_init], dtype=self.dtype)
        self.xrs = torch.tensor([self.xrs_init], dtype=self.dtype)
        self.xs1 = torch.tensor([self.xs1_init], dtype=self.dtype)
        self.xs2 = torch.tensor([self.xs2_init], dtype=self.dtype)

        if not torch.jit.is_scripting():
            self.differentiate = torch.compile(
                self.differentiate,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )


    def interpolate(self, X, table, mn: float, mx: float, res: float, step: float, mn_ind: int, mx_ind: int):
        idx = torch.clamp((X * step).to(torch.long), mn_ind, mx_ind)
        derr = ((X - idx * res) / res).unsqueeze(1)
        lo = idx - mn_ind
        hi = torch.clamp(lo + 1, max=table.shape[0] - 1)
        interp = (1 - derr) * table[lo] + derr * table[hi]
        oob = ((X < mn) | (X > mx)).unsqueeze(1)
        return torch.where(oob, table[lo], interp)

    def construct_tables(self):
        H = (1.0e-4 if (self.modelformulation==self.CLERX) else 1.0e-7)
        PCa = ((0.0001*1.2) if (self.celltype==self.EPI) else ((0.0001*2.5) if (self.celltype==self.MCELL) else 0.0001))
        Pnak = ((30.*0.9) if (self.celltype==self.EPI) else ((30.*0.7) if (self.celltype==self.MCELL) else 30.))
        cmdnmax = ((0.05*1.3) if (self.celltype==self.EPI) else 0.05)
        h10_i = ((self.kasymm+1.0)+((self.Nao/self.kna1)*(1.0+(self.Nao/self.kna2))))
        h10_ss = ((self.kasymm+1.0)+((self.Nao/self.kna1)*(1.+(self.Nao/self.kna2))))
        hL_rush_larsen_B = (exp(((-self.dt)/self.tau_hL)))
        hL_rush_larsen_C = (expm1(((-self.dt)/self.tau_hL)))
        hLp_rush_larsen_B = (exp(((-self.dt)/self.tau_hLp)))
        hLp_rush_larsen_C = (expm1(((-self.dt)/self.tau_hLp)))
        jca_rush_larsen_B = (exp(((-self.dt)/self.tau_jca)))
        jca_rush_larsen_C = (expm1(((-self.dt)/self.tau_jca)))
        sqrt_Ko = (sqrt(self.Ko))
        sqrt_Ko_54 = (sqrt((self.Ko/5.4)))
        PCaK = (3.574e-4*PCa)
        PCaNa = (0.00125*PCa)
        PCap = (1.1*PCa)
        h11_i = ((self.Nao*self.Nao)/((h10_i*self.kna1)*self.kna2))
        h11_ss = ((self.Nao*self.Nao)/((h10_ss*self.kna1)*self.kna2))
        h12_i = (1.0/h10_i)
        h12_ss = (1.0/h10_ss)
        PCaKp = (3.574e-4*PCap)
        PCaNap = (0.00125*PCap)
        k1_i = ((h12_i*self.Cao)*self.kCaon)
        k1_ss = ((h12_ss*self.Cao)*self.kCaon)

        # construct the Cai lookup table
        Cai = torch.arange(self.Cai_T_mn_ind, self.Cai_T_mx_ind + 1, device=self.device, dtype=self.dtype) * self.Cai_T_res
        Cai_tab = torch.zeros((Cai.shape[0], self.Cai_NROWS)).to(self.device).to(self.dtype)

        Cai_tab[:, self.Bcai_idx] = (1.0/((1.0+(((cmdnmax*self.kmcmdn)/(self.kmcmdn+Cai))/(self.kmcmdn+Cai)))+(((self.trpnmax*self.kmtrpn)/(self.kmtrpn+Cai))/(self.kmtrpn+Cai))))
        Cai_tab[:, self.IpCa_idx] = ((self.GpCa*Cai)/(0.0005+Cai))
        Cai_tab[:, self.Jupnp_idx] = ((((0.004375*Cai)/(Cai+0.00092))*1.3) if (self.celltype==self.EPI) else ((0.004375*Cai)/(Cai+0.00092)))
        Cai_tab[:, self.Jupp_idx] = (((((2.75*0.004375)*Cai)/((Cai+0.00092)-(0.00017)))*1.3) if (self.celltype==self.EPI) else (((2.75*0.004375)*Cai)/((Cai+0.00092)-(0.00017))))
        Cai_tab[:, self.KsCa_idx] = (1.0+(0.6/(1.0+(torch.pow((3.8e-5/Cai),1.4)))))
        Cai_tab[:, self.allo_i_idx] = (1.0/(1.0+((self.KmCaAct_i/Cai)*(self.KmCaAct_i/Cai))))

        self.Cai_tab = Cai_tab

        # construct the V lookup table
        V = torch.arange(self.V_T_mn_ind, self.V_T_mx_ind + 1, device=self.device, dtype=self.dtype) * self.V_T_res
        V_tab = torch.zeros((V.shape[0], self.V_NROWS)).to(self.device).to(self.dtype)

        V_tab[:, self.Afcaf_idx] = (0.3+(0.6/(1.0+(torch.exp(((V-(10.0))/10.0))))))
        V_tab[:, self.AiF_idx] = (1.0/(1.0+(torch.exp(((V-(213.6))/151.2)))))
        V_tab[:, self.Axrf_idx] = (1.0/(1.0+(torch.exp(((V+54.81)/38.21)))))
        KNao = (self.KNao0*(torch.exp(((((1.0-(self.delta))*V)*self.F)/((3.0*self.R)*self.T)))))
        V_tab[:, self.Knai_idx] = (self.Knai0*(torch.exp((((self.delta*V)*self.F)/((3.0*self.R)*self.T)))))
        a_inf = (1.0/(1.0+(torch.exp(((-(V-(14.34)))/14.82)))))
        aa_hTT2 = (torch.where((V>=-40.), 0., (0.057*(torch.exp(((-(V+80.))/6.8))))))
        aa_jTT2 = (torch.where((V>=-40.), 0., ((((-2.5428e4*(torch.exp((0.2444*V))))-((6.948e-6*(torch.exp((-0.04391*V))))))*(V+37.78))/(1.+(torch.exp((0.311*(V+79.23))))))))
        aa_mTT2 = (1./(1.+(torch.exp(((-60.-(V))/5.)))))
        ap_inf = (1.0/(1.0+(torch.exp(((-(V-(24.34)))/14.82)))))
        bb_hTT2 = (torch.where((V>=-40.), (0.77/(0.13*(1.+(torch.exp(((-(V+10.66))/11.1)))))), ((2.7*(torch.exp((0.079*V))))+(3.1e5*(torch.exp((0.3485*V)))))))
        bb_jTT2 = (torch.where((V>=-40.), ((0.6*(torch.exp((0.057*V))))/(1.+(torch.exp((-0.1*(V+32.)))))), ((0.02424*(torch.exp((-0.01052*V))))/(1.+(torch.exp((-0.1378*(V+40.14))))))))
        bb_mTT2 = ((0.1/(1.+(torch.exp(((V+35.)/5.)))))+(0.10/(1.+(torch.exp(((V-(50.))/200.))))))
        d_inf = (1.0/(1.0+(torch.exp(((-(V+3.940))/4.230)))))
        delta_epi = ((1.0-((0.95/(1.0+(torch.exp(((V+70.0)/5.0))))))) if (self.celltype==self.EPI) else torch.full_like(V, 1.0))
        dti_develop = (1.354+(1.0e-4/((torch.exp(((V-(167.4))/15.89)))+(torch.exp(((-(V-(12.23)))/0.2154))))))
        dti_recover = (1.0-((0.5/(1.0+(torch.exp(((V+70.0)/20.0)))))))
        f_inf = (1.0/(1.0+(torch.exp(((V+19.58)/3.696)))))
        hL_inf = (1.0/(1.0+(torch.exp(((V+87.61)/7.488)))))
        hLp_inf = (1.0/(1.0+(torch.exp(((V+93.81)/7.488)))))
        hTT2_inf = (1./((1.+(torch.exp(((V+71.55)/7.43))))*(1.+(torch.exp(((V+71.55)/7.43))))))
        h_inf = (1.0/(1.+(torch.exp(((V+82.90)/6.086)))))
        V_tab[:, self.hca_idx] = (torch.exp((((self.qca*V)*self.F)/(self.R*self.T))))
        V_tab[:, self.hna_idx] = (torch.exp((((self.qna*V)*self.F)/(self.R*self.T))))
        hsp_inf = (1.0/(1.+(torch.exp(((V+89.1)/6.086)))))
        i_inf = (1.0/(1.0+(torch.exp(((V+43.94)/5.711)))))
        mL_inf = (1.0/(1.0+(torch.exp(((-(V+42.85))/5.264)))))
        mORd_inf = (1.0/(1.0+(torch.exp(((-((V+39.57)-(self.shift_m_inf)))/9.871)))))
        mTT2_inf = (1./((1.+(torch.exp(((-56.86-(V))/9.03))))*(1.+(torch.exp(((-56.86-(V))/9.03))))))
        V_tab[:, self.rk1_idx] = (1.0/(1.0+(torch.exp((((V+105.8)-((2.6*self.Ko)))/9.493)))))
        V_tab[:, self.rkr_idx] = ((1.0/(1.0+(torch.exp(((V+55.0)/75.0)))))/(1.0+(torch.exp(((V-(10.0))/30.0)))))
        tau_a = (1.0515/((1.0/(1.2089*(1.0+(torch.exp(((-(V-(18.4099)))/29.3814))))))+(3.5/(1.0+(torch.exp(((V+100.0)/29.3814)))))))
        tau_d = (0.6+(1.0/((torch.exp((-0.05*(V+6.0))))+(torch.exp((0.09*(V+14.0)))))))
        tau_fcaf = (7.0+(1.0/((0.04*(torch.exp(((-(V-(4.0)))/7.0))))+(0.04*(torch.exp(((V-(4.0))/7.0)))))))
        tau_fcas = (100.0+(1.0/((0.00012*(torch.exp(((-V)/3.0))))+(0.00012*(torch.exp((V/7.0)))))))
        tau_ff = (7.0+(1.0/((0.0045*(torch.exp(((-(V+20.0))/10.0))))+(0.0045*(torch.exp(((V+20.0)/10.0)))))))
        tau_fs = (1000.0+(1.0/((0.000035*(torch.exp(((-(V+5.0))/4.0))))+(0.000035*(torch.exp(((V+5.0)/6.0)))))))
        tau_hf = (1.0/((1.432e-5*(torch.exp(((-(V+1.196))/6.285))))+(6.149*(torch.exp(((V+0.5096)/20.27))))))
        tau_hs = (1.0/((0.009794*(torch.exp(((-(V+17.95))/28.05))))+(0.3343*(torch.exp(((V+5.730)/56.66))))))
        tau_j = (2.038+(1.0/((0.02136*(torch.exp(((-(V+100.6))/8.281))))+(0.3052*(torch.exp(((V+0.9941)/38.45)))))))
        tau_mORd = (self.scale_tau_m/((6.765*(torch.exp(((V+11.64)/34.77))))+(8.552*(torch.exp(((-(V+77.42))/5.955))))))
        tau_xk1 = (122.2/((torch.exp(((-(V+127.2))/20.36)))+(torch.exp(((V+236.8)/69.33)))))
        tau_xrf = (12.98+(1.0/((0.3652*(torch.exp(((V-(31.66))/3.869))))+(4.123e-5*(torch.exp(((-(V-(47.78)))/20.38)))))))
        tau_xrs = (1.865+(1.0/((0.06629*(torch.exp(((V-(34.70))/7.355))))+(1.128e-5*(torch.exp(((-(V-(29.74)))/25.94)))))))
        tau_xs1 = (817.3+(1.0/((2.326e-4*(torch.exp(((V+48.28)/17.80))))+(0.001292*(torch.exp(((-(V+210.0))/230.0)))))))
        tau_xs2 = (1.0/((0.01*(torch.exp(((V-(50.0))/20.0))))+(0.0193*(torch.exp(((-(V+66.54))/31.0))))))
        vffrt = (((V*self.F)*self.F)/(self.R*self.T))
        vfrt = ((V*self.F)/(self.R*self.T))
        xk1_inf = (1.0/(1.0+(torch.exp(((-((V+(2.5538*self.Ko))+144.59))/((1.5692*self.Ko)+3.8115))))))
        V_tab[:, self.xkb_idx] = (1.0/(1.0+(torch.exp(((-(V-(14.48)))/18.34)))))
        xr_inf = (1.0/(1.0+(torch.exp(((-(V+8.337))/6.789)))))
        xs1_inf = (1.0/(1.0+(torch.exp(((-(V+self.vHalfXs))/8.932)))))
        V_tab[:, self.Afcas_idx] = (1.0-(V_tab[:, self.Afcaf_idx]))
        V_tab[:, self.AiS_idx] = (1.0-(V_tab[:, self.AiF_idx]))
        V_tab[:, self.Axrs_idx] = (1.0-(V_tab[:, self.Axrf_idx]))
        Nao_KNao_3 = (((self.Nao/KNao)*(self.Nao/KNao))*(self.Nao/KNao))
        Nao_KNao_p1_3 = (((1.0+(self.Nao/KNao))*(1.0+(self.Nao/KNao)))*(1.0+(self.Nao/KNao)))
        V_tab[:, self.a_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_a)))
        a_rush_larsen_C = (torch.expm1(((-self.dt)/tau_a)))
        V_tab[:, self.d_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_d)))
        d_rush_larsen_C = (torch.expm1(((-self.dt)/tau_d)))
        V_tab[:, self.exp_2vfrt_idx] = (torch.exp((2.0*vfrt)))
        V_tab[:, self.exp_vfrt_idx] = (torch.exp(vfrt))
        fca_inf = f_inf
        V_tab[:, self.fcaf_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_fcaf)))
        fcaf_rush_larsen_C = (torch.expm1(((-self.dt)/tau_fcaf)))
        V_tab[:, self.fcas_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_fcas)))
        fcas_rush_larsen_C = (torch.expm1(((-self.dt)/tau_fcas)))
        ff_inf = f_inf
        V_tab[:, self.ff_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_ff)))
        ff_rush_larsen_C = (torch.expm1(((-self.dt)/tau_ff)))
        ffp_inf = f_inf
        fs_inf = f_inf
        V_tab[:, self.fs_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_fs)))
        fs_rush_larsen_C = (torch.expm1(((-self.dt)/tau_fs)))
        h7_i = (1.0+((self.Nao/self.kna3)*(1.0+(1.0/V_tab[:, self.hna_idx]))))
        h7_ss = (1.0+((self.Nao/self.kna3)*(1.0+(1.0/V_tab[:, self.hna_idx]))))
        V_tab[:, self.hL_rush_larsen_A_idx] = ((-hL_inf)*hL_rush_larsen_C)
        V_tab[:, self.hLp_rush_larsen_A_idx] = ((-hLp_inf)*hLp_rush_larsen_C)
        hf_inf = h_inf
        V_tab[:, self.hf_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_hf)))
        hf_rush_larsen_C = (torch.expm1(((-self.dt)/tau_hf)))
        hs_inf = h_inf
        V_tab[:, self.hs_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_hs)))
        hs_rush_larsen_C = (torch.expm1(((-self.dt)/tau_hs)))
        iF_inf = i_inf
        iFp_inf = i_inf
        iS_inf = i_inf
        iSp_inf = i_inf
        jTT2_inf = hTT2_inf
        j_inf = h_inf
        V_tab[:, self.j_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_j)))
        j_rush_larsen_C = (torch.expm1(((-self.dt)/tau_j)))
        V_tab[:, self.mORd_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_mORd)))
        mORd_rush_larsen_C = (torch.expm1(((-self.dt)/tau_mORd)))
        tau_ap = tau_a
        tau_fcafp = (2.5*tau_fcaf)
        tau_ffp = (2.5*tau_ff)
        tau_hTT2 = (1.0/(aa_hTT2+bb_hTT2))
        tau_hsp = (3.0*tau_hs)
        tau_iF = ((4.562+(1./((0.3933*(torch.exp(((-(V+100.0))/100.0))))+(0.08004*(torch.exp(((V+50.0)/16.59)))))))*delta_epi)
        tau_iS = ((23.62+(1./((0.001416*(torch.exp(((-(V+96.52))/59.05))))+(1.780e-8*(torch.exp(((V+114.1)/8.079)))))))*delta_epi)
        tau_jTT2 = (1.0/(aa_jTT2+bb_jTT2))
        tau_jp = (1.46*tau_j)
        tau_mTT2 = (aa_mTT2*bb_mTT2)
        V_tab[:, self.vffrt_expm1_2vfrt_idx] = (torch.where((V==0.), (self.F/2.), (vffrt/(torch.expm1((2.0*vfrt))))))
        V_tab[:, self.vffrt_expm1_vfrt_idx] = (torch.where((V==0.), self.F, (vffrt/(torch.expm1(vfrt)))))
        V_tab[:, self.xk1_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_xk1)))
        xk1_rush_larsen_C = (torch.expm1(((-self.dt)/tau_xk1)))
        xrf_inf = xr_inf
        V_tab[:, self.xrf_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_xrf)))
        xrf_rush_larsen_C = (torch.expm1(((-self.dt)/tau_xrf)))
        xrs_inf = xr_inf
        V_tab[:, self.xrs_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_xrs)))
        xrs_rush_larsen_C = (torch.expm1(((-self.dt)/tau_xrs)))
        V_tab[:, self.xs1_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_xs1)))
        xs1_rush_larsen_C = (torch.expm1(((-self.dt)/tau_xs1)))
        xs2_inf = xs1_inf
        V_tab[:, self.xs2_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_xs2)))
        xs2_rush_larsen_C = (torch.expm1(((-self.dt)/tau_xs2)))
        V_tab[:, self.a3_idx] = (((self.k3p*(self.Ko/self.KKo))*(self.Ko/self.KKo))/((Nao_KNao_p1_3+((1.0+(self.Ko/self.KKo))*(1.0+(self.Ko/self.KKo))))-(1.0)))
        V_tab[:, self.a_rush_larsen_A_idx] = ((-a_inf)*a_rush_larsen_C)
        V_tab[:, self.ap_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_ap)))
        ap_rush_larsen_C = (torch.expm1(((-self.dt)/tau_ap)))
        V_tab[:, self.b2_idx] = ((self.k2m*Nao_KNao_3)/((Nao_KNao_p1_3+((1.0+(self.Ko/self.KKo))*(1.0+(self.Ko/self.KKo))))-(1.0)))
        V_tab[:, self.d_rush_larsen_A_idx] = ((-d_inf)*d_rush_larsen_C)
        fcaf_inf = fca_inf
        fcafp_inf = fca_inf
        V_tab[:, self.fcafp_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_fcafp)))
        fcafp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_fcafp)))
        fcas_inf = fca_inf
        V_tab[:, self.ff_rush_larsen_A_idx] = ((-ff_inf)*ff_rush_larsen_C)
        V_tab[:, self.ffp_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_ffp)))
        ffp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_ffp)))
        V_tab[:, self.fs_rush_larsen_A_idx] = ((-fs_inf)*fs_rush_larsen_C)
        h8_i = (self.Nao/((self.kna3*V_tab[:, self.hna_idx])*h7_i))
        h8_ss = (self.Nao/((self.kna3*V_tab[:, self.hna_idx])*h7_ss))
        h9_i = (1.0/h7_i)
        h9_ss = (1.0/h7_ss)
        V_tab[:, self.hTT2_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_hTT2)))
        hTT2_rush_larsen_C = (torch.expm1(((-self.dt)/tau_hTT2)))
        V_tab[:, self.hf_rush_larsen_A_idx] = ((-hf_inf)*hf_rush_larsen_C)
        V_tab[:, self.hs_rush_larsen_A_idx] = ((-hs_inf)*hs_rush_larsen_C)
        V_tab[:, self.hsp_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_hsp)))
        hsp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_hsp)))
        V_tab[:, self.iF_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_iF)))
        iF_rush_larsen_C = (torch.expm1(((-self.dt)/tau_iF)))
        V_tab[:, self.iS_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_iS)))
        iS_rush_larsen_C = (torch.expm1(((-self.dt)/tau_iS)))
        V_tab[:, self.jTT2_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_jTT2)))
        jTT2_rush_larsen_C = (torch.expm1(((-self.dt)/tau_jTT2)))
        V_tab[:, self.j_rush_larsen_A_idx] = ((-j_inf)*j_rush_larsen_C)
        jca_inf = fca_inf
        jp_inf = j_inf
        V_tab[:, self.jp_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_jp)))
        jp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_jp)))
        V_tab[:, self.mORd_rush_larsen_A_idx] = ((-mORd_inf)*mORd_rush_larsen_C)
        V_tab[:, self.mTT2_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_mTT2)))
        mTT2_rush_larsen_C = (torch.expm1(((-self.dt)/tau_mTT2)))
        tau_iFp = ((dti_develop*dti_recover)*tau_iF)
        tau_iSp = ((dti_develop*dti_recover)*tau_iS)
        tau_m = (tau_mTT2 if (self.INa_Type==self.TT2INa) else tau_mORd)
        V_tab[:, self.xk1_rush_larsen_A_idx] = ((-xk1_inf)*xk1_rush_larsen_C)
        V_tab[:, self.xrf_rush_larsen_A_idx] = ((-xrf_inf)*xrf_rush_larsen_C)
        V_tab[:, self.xrs_rush_larsen_A_idx] = ((-xrs_inf)*xrs_rush_larsen_C)
        V_tab[:, self.xs1_rush_larsen_A_idx] = ((-xs1_inf)*xs1_rush_larsen_C)
        V_tab[:, self.xs2_rush_larsen_A_idx] = ((-xs2_inf)*xs2_rush_larsen_C)
        V_tab[:, self.ap_rush_larsen_A_idx] = ((-ap_inf)*ap_rush_larsen_C)
        V_tab[:, self.fcaf_rush_larsen_A_idx] = ((-fcaf_inf)*fcaf_rush_larsen_C)
        V_tab[:, self.fcafp_rush_larsen_A_idx] = ((-fcafp_inf)*fcafp_rush_larsen_C)
        V_tab[:, self.fcas_rush_larsen_A_idx] = ((-fcas_inf)*fcas_rush_larsen_C)
        V_tab[:, self.ffp_rush_larsen_A_idx] = ((-ffp_inf)*ffp_rush_larsen_C)
        V_tab[:, self.hTT2_rush_larsen_A_idx] = ((-hTT2_inf)*hTT2_rush_larsen_C)
        V_tab[:, self.hsp_rush_larsen_A_idx] = ((-hsp_inf)*hsp_rush_larsen_C)
        V_tab[:, self.iF_rush_larsen_A_idx] = ((-iF_inf)*iF_rush_larsen_C)
        V_tab[:, self.iFp_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_iFp)))
        iFp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_iFp)))
        V_tab[:, self.iS_rush_larsen_A_idx] = ((-iS_inf)*iS_rush_larsen_C)
        V_tab[:, self.iSp_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_iSp)))
        iSp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_iSp)))
        V_tab[:, self.jTT2_rush_larsen_A_idx] = ((-jTT2_inf)*jTT2_rush_larsen_C)
        V_tab[:, self.jca_rush_larsen_A_idx] = ((-jca_inf)*jca_rush_larsen_C)
        V_tab[:, self.jp_rush_larsen_A_idx] = ((-jp_inf)*jp_rush_larsen_C)
        k3p_i = (h9_i*self.wca)
        k3p_ss = (h9_ss*self.wca)
        V_tab[:, self.k3pp_i_idx] = (h8_i*self.wnaca)
        V_tab[:, self.k3pp_ss_idx] = (h8_ss*self.wnaca)
        V_tab[:, self.k8_i_idx] = ((h8_i*h11_i)*self.wna)
        V_tab[:, self.k8_ss_idx] = ((h8_ss*h11_ss)*self.wna)
        V_tab[:, self.mTT2_rush_larsen_A_idx] = ((-mTT2_inf)*mTT2_rush_larsen_C)
        tau_mL = tau_m
        V_tab[:, self.iFp_rush_larsen_A_idx] = ((-iFp_inf)*iFp_rush_larsen_C)
        V_tab[:, self.iSp_rush_larsen_A_idx] = ((-iSp_inf)*iSp_rush_larsen_C)
        V_tab[:, self.k3_i_idx] = (k3p_i+V_tab[:, self.k3pp_i_idx])
        V_tab[:, self.k3_ss_idx] = (k3p_ss+V_tab[:, self.k3pp_ss_idx])
        V_tab[:, self.mL_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_mL)))
        mL_rush_larsen_C = (torch.expm1(((-self.dt)/tau_mL)))
        V_tab[:, self.mL_rush_larsen_A_idx] = ((-mL_inf)*mL_rush_larsen_C)

        self.V_tab = V_tab

    def initialize(self, n_nodes: int):
        self.construct_tables()
        
        V = torch.full((n_nodes,), self.V_init, device=self.device, dtype=self.dtype)

        self.CaMKt = torch.full((n_nodes,), self.CaMKt_init, device=self.device, dtype=self.dtype)
        self.Cai = torch.full((n_nodes,), self.Cai_init, device=self.device, dtype=self.dtype)
        self.Cai = self.Cai * 1e3
        self.Jrelnp = torch.full((n_nodes,), self.Jrelnp_init, device=self.device, dtype=self.dtype)
        self.Jrelp = torch.full((n_nodes,), self.Jrelp_init, device=self.device, dtype=self.dtype)
        self.Ki = torch.full((n_nodes,), self.Ki_init, device=self.device, dtype=self.dtype)
        self.Nai = torch.full((n_nodes,), self.Nai_init, device=self.device, dtype=self.dtype)
        self.a = torch.full((n_nodes,), self.a_init, device=self.device, dtype=self.dtype)
        self.ap = torch.full((n_nodes,), self.ap_init, device=self.device, dtype=self.dtype)
        self.cajsr = torch.full((n_nodes,), self.cajsr_init, device=self.device, dtype=self.dtype)
        self.cansr = torch.full((n_nodes,), self.cansr_init, device=self.device, dtype=self.dtype)
        self.cass = torch.full((n_nodes,), self.cass_init, device=self.device, dtype=self.dtype)
        self.d = torch.full((n_nodes,), self.d_init, device=self.device, dtype=self.dtype)
        self.fcaf = torch.full((n_nodes,), self.fcaf_init, device=self.device, dtype=self.dtype)
        self.fcafp = torch.full((n_nodes,), self.fcafp_init, device=self.device, dtype=self.dtype)
        self.fcas = torch.full((n_nodes,), self.fcas_init, device=self.device, dtype=self.dtype)
        self.ff = torch.full((n_nodes,), self.ff_init, device=self.device, dtype=self.dtype)
        self.ffp = torch.full((n_nodes,), self.ffp_init, device=self.device, dtype=self.dtype)
        self.fs = torch.full((n_nodes,), self.fs_init, device=self.device, dtype=self.dtype)
        self.hL = torch.full((n_nodes,), self.hL_init, device=self.device, dtype=self.dtype)
        self.hLp = torch.full((n_nodes,), self.hLp_init, device=self.device, dtype=self.dtype)
        self.hTT2 = torch.full((n_nodes,), self.hTT2_init, device=self.device, dtype=self.dtype)
        self.hf = torch.full((n_nodes,), self.hf_init, device=self.device, dtype=self.dtype)
        self.hs = torch.full((n_nodes,), self.hs_init, device=self.device, dtype=self.dtype)
        self.hsp = torch.full((n_nodes,), self.hsp_init, device=self.device, dtype=self.dtype)
        self.iF = torch.full((n_nodes,), self.iF_init, device=self.device, dtype=self.dtype)
        self.iFp = torch.full((n_nodes,), self.iFp_init, device=self.device, dtype=self.dtype)
        self.iS = torch.full((n_nodes,), self.iS_init, device=self.device, dtype=self.dtype)
        self.iSp = torch.full((n_nodes,), self.iSp_init, device=self.device, dtype=self.dtype)
        self.j = torch.full((n_nodes,), self.j_init, device=self.device, dtype=self.dtype)
        self.jTT2 = torch.full((n_nodes,), self.jTT2_init, device=self.device, dtype=self.dtype)
        self.jca = torch.full((n_nodes,), self.jca_init, device=self.device, dtype=self.dtype)
        self.jp = torch.full((n_nodes,), self.jp_init, device=self.device, dtype=self.dtype)
        self.kss = torch.full((n_nodes,), self.kss_init, device=self.device, dtype=self.dtype)
        self.mL = torch.full((n_nodes,), self.mL_init, device=self.device, dtype=self.dtype)
        self.mORd = torch.full((n_nodes,), self.mORd_init, device=self.device, dtype=self.dtype)
        self.mTT2 = torch.full((n_nodes,), self.mTT2_init, device=self.device, dtype=self.dtype)
        self.nass = torch.full((n_nodes,), self.nass_init, device=self.device, dtype=self.dtype)
        self.nca = torch.full((n_nodes,), self.nca_init, device=self.device, dtype=self.dtype)
        self.xk1 = torch.full((n_nodes,), self.xk1_init, device=self.device, dtype=self.dtype)
        self.xrf = torch.full((n_nodes,), self.xrf_init, device=self.device, dtype=self.dtype)
        self.xrs = torch.full((n_nodes,), self.xrs_init, device=self.device, dtype=self.dtype)
        self.xs1 = torch.full((n_nodes,), self.xs1_init, device=self.device, dtype=self.dtype)
        self.xs2 = torch.full((n_nodes,), self.xs2_init, device=self.device, dtype=self.dtype)

        return V

    def differentiate(self, V):
        self.Cai = self.Cai * 1e-3

        # Define the constants that depend on the parameters.
        H = (1.0e-4 if (self.modelformulation==self.CLERX) else 1.0e-7)
        PCa = ((0.0001*1.2) if (self.celltype==self.EPI) else ((0.0001*2.5) if (self.celltype==self.MCELL) else 0.0001))
        Pnak = ((30.*0.9) if (self.celltype==self.EPI) else ((30.*0.7) if (self.celltype==self.MCELL) else 30.))
        cmdnmax = ((0.05*1.3) if (self.celltype==self.EPI) else 0.05)
        h10_i = ((self.kasymm+1.0)+((self.Nao/self.kna1)*(1.0+(self.Nao/self.kna2))))
        h10_ss = ((self.kasymm+1.0)+((self.Nao/self.kna1)*(1.+(self.Nao/self.kna2))))
        hL_rush_larsen_B = (exp(((-self.dt)/self.tau_hL)))
        hL_rush_larsen_C = (expm1(((-self.dt)/self.tau_hL)))
        hLp_rush_larsen_B = (exp(((-self.dt)/self.tau_hLp)))
        hLp_rush_larsen_C = (expm1(((-self.dt)/self.tau_hLp)))
        jca_rush_larsen_B = (exp(((-self.dt)/self.tau_jca)))
        jca_rush_larsen_C = (expm1(((-self.dt)/self.tau_jca)))
        sqrt_Ko = (sqrt(self.Ko))
        sqrt_Ko_54 = (sqrt((self.Ko/5.4)))
        PCaK = (3.574e-4*PCa)
        PCaNa = (0.00125*PCa)
        PCap = (1.1*PCa)
        h11_i = ((self.Nao*self.Nao)/((h10_i*self.kna1)*self.kna2))
        h11_ss = ((self.Nao*self.Nao)/((h10_ss*self.kna1)*self.kna2))
        h12_i = (1.0/h10_i)
        h12_ss = (1.0/h10_ss)
        PCaKp = (3.574e-4*PCap)
        PCaNap = (0.00125*PCap)
        k1_i = ((h12_i*self.Cao)*self.kCaon)
        k1_ss = ((h12_ss*self.Cao)*self.kCaon)

        Cai_row = self.interpolate(self.Cai, self.Cai_tab, self.Cai_T_mn, self.Cai_T_mx, self.Cai_T_res, self.Cai_T_step, self.Cai_T_mn_ind, self.Cai_T_mx_ind)
        V_row = self.interpolate(V, self.V_tab, self.V_T_mn, self.V_T_mx, self.V_T_res, self.V_T_step, self.V_T_mn_ind, self.V_T_mx_ind)

        # Compute storevars and external modvars
        CaMKb = ((self.CaMKo*(1.0-(self.CaMKt)))/(1.0+(self.KmCaM/self.cass)))
        EK = (((self.R*self.T)/self.F)*(torch.log((self.Ko/self.Ki))))
        EKs = (((self.R*self.T)/self.F)*(torch.log(((self.Ko+(self.PKNa*self.Nao))/(self.Ki+(self.PKNa*self.Nai))))))
        ENa = (((self.R*self.T)/self.F)*(torch.log((self.Nao/self.Nai))))
        P = (self.eP/(((1.0+(H/self.Khp))+(self.Nai/self.Knap))+(self.Ki/self.Kxkur)))
        allo_ss = (1.0/(1.0+((self.KmCaAct_ss/self.cass)*(self.KmCaAct_ss/self.cass))))
        f = ((self.Aff*self.ff)+(self.Afs*self.fs))
        fp = ((self.Aff*self.ffp)+(self.Afs*self.fs))
        h = ((self.Ahf*self.hf)+(self.Ahs*self.hs))
        h4_i = (1.0+((self.Nai/self.kna1)*(1.+(self.Nai/self.kna2))))
        h4_ss = (1.0+((self.nass/self.kna1)*(1.+(self.nass/self.kna2))))
        hp = ((self.Ahf*self.hf)+(self.Ahs*self.hsp))
        CaMKa = (CaMKb+self.CaMKt)
        IK1 = ((((self.GK1*sqrt_Ko)*V_row[:, self.rk1_idx])*self.xk1)*(V-(EK)))
        IKb = ((self.GKb*V_row[:, self.xkb_idx])*(V-(EK)))
        IKs = ((((self.GKs*Cai_row[:, self.KsCa_idx])*self.xs1)*self.xs2)*(V-(EKs)))
        b3 = (((self.k3m*P)*H)/(1.0+(self.MgATP/self.Kmgatp)))
        h1_i = (1.+((self.Nai/self.kna3)*(1.+V_row[:, self.hna_idx])))
        h1_ss = (1.+((self.nass/self.kna3)*(1.+V_row[:, self.hna_idx])))
        h5_i = ((self.Nai*self.Nai)/((h4_i*self.kna1)*self.kna2))
        h5_ss = ((self.nass*self.nass)/((h4_ss*self.kna1)*self.kna2))
        h6_i = (1.0/h4_i)
        h6_ss = (1.0/h4_ss)
        nai_Knai_3 = (((self.Nai/V_row[:, self.Knai_idx])*(self.Nai/V_row[:, self.Knai_idx]))*(self.Nai/V_row[:, self.Knai_idx]))
        nai_Knai_p1_3 = (((1.0+(self.Nai/V_row[:, self.Knai_idx]))*(1.0+(self.Nai/V_row[:, self.Knai_idx])))*(1.0+(self.Nai/V_row[:, self.Knai_idx])))
        IbCa = ((((self.factorIbCa*self.PCab)*4.0)*V_row[:, self.vffrt_expm1_2vfrt_idx])*((self.Cai*V_row[:, self.exp_2vfrt_idx])-((0.341*self.Cao))))
        IbNa = (((self.factorIbNa*self.PNab)*V_row[:, self.vffrt_expm1_vfrt_idx])*((self.Nai*V_row[:, self.exp_vfrt_idx])-(self.Nao)))
        PhiCaK = (V_row[:, self.vffrt_expm1_vfrt_idx]*(((0.75*self.kss)*V_row[:, self.exp_vfrt_idx])-((0.75*self.Ko))))
        PhiCaL = ((4.0*V_row[:, self.vffrt_expm1_2vfrt_idx])*((self.cass*V_row[:, self.exp_2vfrt_idx])-((0.341*self.Cao))))
        PhiCaNa = (V_row[:, self.vffrt_expm1_vfrt_idx]*(((0.75*self.nass)*V_row[:, self.exp_vfrt_idx])-((0.75*self.Nao))))
        a1 = ((self.k1p*nai_Knai_3)/((nai_Knai_p1_3+((1.0+(self.Ki/self.Kki))*(1.0+(self.Ki/self.Kki))))-(1.0)))
        b4 = (((self.k4m*(self.Ki/self.Kki))*(self.Ki/self.Kki))/((nai_Knai_p1_3+((1.0+(self.Ki/self.Kki))*(1.0+(self.Ki/self.Kki))))-(1.0)))
        fICaLp = (1.0/(1.0+(self.KmCaMK/CaMKa)))
        fINaLp = (1.0/(1.0+(self.KmCaMK/CaMKa)))
        fINap = (1.0/(1.0+(self.KmCaMK/CaMKa)))
        fItop = (1.0/(1.0+(self.KmCaMK/CaMKa)))
        fca = ((V_row[:, self.Afcaf_idx]*self.fcaf)+(V_row[:, self.Afcas_idx]*self.fcas))
        fcap = ((V_row[:, self.Afcaf_idx]*self.fcafp)+(V_row[:, self.Afcas_idx]*self.fcas))
        h2_i = ((self.Nai*V_row[:, self.hna_idx])/(self.kna3*h1_i))
        h2_ss = ((self.nass*V_row[:, self.hna_idx])/(self.kna3*h1_ss))
        h3_i = (1.0/h1_i)
        h3_ss = (1.0/h1_ss)
        i = ((V_row[:, self.AiF_idx]*self.iF)+(V_row[:, self.AiS_idx]*self.iS))
        ip = ((V_row[:, self.AiF_idx]*self.iFp)+(V_row[:, self.AiS_idx]*self.iSp))
        k6_i = ((h6_i*self.Cai)*self.kCaon)
        k6_ss = ((h6_ss*self.cass)*self.kCaon)
        xr = ((V_row[:, self.Axrf_idx]*self.xrf)+(V_row[:, self.Axrs_idx]*self.xrs))
        ICaK = ((((((self.factorICaK*(1.0-(fICaLp)))*PCaK)*PhiCaK)*self.d)*((f*(1.0-(self.nca)))+((self.jca*fca)*self.nca)))+((((fICaLp*PCaKp)*PhiCaK)*self.d)*((fp*(1.0-(self.nca)))+((self.jca*fcap)*self.nca))))
        ICaL = ((((((self.factorICaL*(1.0-(fICaLp)))*PCa)*PhiCaL)*self.d)*((f*(1.0-(self.nca)))+((self.jca*fca)*self.nca)))+((((fICaLp*PCap)*PhiCaL)*self.d)*((fp*(1.0-(self.nca)))+((self.jca*fcap)*self.nca))))
        ICaNa = ((((((self.factorICaNa*(1.0-(fICaLp)))*PCaNa)*PhiCaNa)*self.d)*((f*(1.0-(self.nca)))+((self.jca*fca)*self.nca)))+((((fICaLp*PCaNap)*PhiCaNa)*self.d)*((fp*(1.0-(self.nca)))+((self.jca*fcap)*self.nca))))
        IKr = ((((self.GKr*sqrt_Ko_54)*xr)*V_row[:, self.rkr_idx])*(V-(EK)))
        INa = (((((((self.GNa*self.mTT2)*self.mTT2)*self.mTT2)*self.hTT2)*self.jTT2)*(V-(ENa))) if (self.INa_Type==self.TT2INa) else (((((self.GNa*(V-(ENa)))*self.mORd)*self.mORd)*self.mORd)*((((1.0-(fINap))*h)*self.j)+((fINap*hp)*self.jp))))
        INaL = (((self.GNaL*(V-(ENa)))*self.mL)*(((1.0-(fINaLp))*self.hL)+(fINaLp*self.hLp)))
        Ito = ((self.Gto*(V-(EK)))*((((1.0-(fItop))*self.a)*i)+((fItop*self.ap)*ip)))
        k4p_i = ((h3_i*self.wca)/V_row[:, self.hca_idx])
        k4p_ss = ((h3_ss*self.wca)/V_row[:, self.hca_idx])
        k4pp_i = (h2_i*self.wnaca)
        k4pp_ss = (h2_ss*self.wnaca)
        k7_i = ((h5_i*h2_i)*self.wna)
        k7_ss = ((h5_ss*h2_ss)*self.wna)
        x1 = ((((((self.a4*a1)*self.a2)+((self.b1*b4)*b3))+((self.a2*b4)*b3))+((b3*a1)*self.a2)) if (self.modelformulation==self.CLERX) else (((((self.a4*a1)*self.a2)+((V_row[:, self.b2_idx]*b4)*b3))+((self.a2*b4)*b3))+((b3*a1)*self.a2)))
        x2 = (((((V_row[:, self.b2_idx]*self.b1)*b4)+((a1*self.a2)*V_row[:, self.a3_idx]))+((V_row[:, self.a3_idx]*self.b1)*b4))+((self.a2*V_row[:, self.a3_idx])*b4))
        x3 = (((((self.a2*V_row[:, self.a3_idx])*self.a4)+((b3*V_row[:, self.b2_idx])*self.b1))+((V_row[:, self.b2_idx]*self.b1)*self.a4))+((V_row[:, self.a3_idx]*self.a4)*self.b1))
        x4 = (((((b4*b3)*V_row[:, self.b2_idx])+((V_row[:, self.a3_idx]*self.a4)*a1))+((V_row[:, self.b2_idx]*self.a4)*a1))+((b3*V_row[:, self.b2_idx])*a1))
        E1 = (x1/(((x1+x2)+x3)+x4))
        E2 = (x2/(((x1+x2)+x3)+x4))
        E3 = (x3/(((x1+x2)+x3)+x4))
        E4 = (x4/(((x1+x2)+x3)+x4))
        k4_i = (k4p_i+k4pp_i)
        k4_ss = (k4p_ss+k4pp_ss)
        r = (((((a1*self.a2)*V_row[:, self.a3_idx])*self.a4)-((((self.b1*V_row[:, self.b2_idx])*b3)*b4)))/(((x1+x2)+x3)+x4))
        JNaKK = ((-2.*r) if (self.modelformulation==self.CLERX) else (2.*((E4*self.b1)-((E3*a1)))))
        JNaKNa = ((3.*r) if (self.modelformulation==self.CLERX) else (3.*((E1*V_row[:, self.a3_idx])-((E2*b3)))))
        x1_i = (((self.k2_i*k4_i)*(k7_i+k6_i))+((self.k5_i*k7_i)*(self.k2_i+V_row[:, self.k3_i_idx])))
        x1_ss = (((self.k2_ss*k4_ss)*(k7_ss+k6_ss))+((self.k5_ss*k7_ss)*(self.k2_ss+V_row[:, self.k3_ss_idx])))
        x2_i = (((k1_i*k7_i)*(k4_i+self.k5_i))+((k4_i*k6_i)*(k1_i+V_row[:, self.k8_i_idx])))
        x2_ss = (((k1_ss*k7_ss)*(k4_ss+self.k5_ss))+((k4_ss*k6_ss)*(k1_ss+V_row[:, self.k8_ss_idx])))
        x3_i = (((k1_i*V_row[:, self.k3_i_idx])*(k7_i+k6_i))+((V_row[:, self.k8_i_idx]*k6_i)*(self.k2_i+V_row[:, self.k3_i_idx])))
        x3_ss = (((k1_ss*V_row[:, self.k3_ss_idx])*(k7_ss+k6_ss))+((V_row[:, self.k8_ss_idx]*k6_ss)*(self.k2_ss+V_row[:, self.k3_ss_idx])))
        x4_i = (((self.k2_i*V_row[:, self.k8_i_idx])*(k4_i+self.k5_i))+((V_row[:, self.k3_i_idx]*self.k5_i)*(k1_i+V_row[:, self.k8_i_idx])))
        x4_ss = (((self.k2_ss*V_row[:, self.k8_ss_idx])*(k4_ss+self.k5_ss))+((V_row[:, self.k3_ss_idx]*self.k5_ss)*(k1_ss+V_row[:, self.k8_ss_idx])))
        E1_i = (x1_i/(((x1_i+x2_i)+x3_i)+x4_i))
        E1_ss = (x1_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss))
        E2_i = (x2_i/(((x1_i+x2_i)+x3_i)+x4_i))
        E2_ss = (x2_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss))
        E3_i = (x3_i/(((x1_i+x2_i)+x3_i)+x4_i))
        E3_ss = (x3_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss))
        E4_i = (x4_i/(((x1_i+x2_i)+x3_i)+x4_i))
        E4_ss = (x4_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss))
        INaK = ((self.factorINaK*Pnak)*((self.zna*JNaKNa)+(self.zk*JNaKK)))
        JncxCa_ss = ((E2_ss*self.k2_ss)-((E1_ss*k1_ss)))
        JncxCai = ((E2_i*self.k2_i)-((E1_i*k1_i)))
        JncxNa_ss = (((3.0*((E4_ss*k7_ss)-((E1_ss*V_row[:, self.k8_ss_idx]))))+(E3_ss*k4pp_ss))-((E2_ss*V_row[:, self.k3pp_ss_idx])))
        JncxNai = (((3.0*((E4_i*k7_i)-((E1_i*V_row[:, self.k8_i_idx]))))+(E3_i*k4pp_i))-((E2_i*V_row[:, self.k3pp_i_idx])))
        INaCa = ((((self.factorINaCa*0.8)*self.GNaCa)*Cai_row[:, self.allo_i_idx])*((self.zna*JncxNai)+(self.zca*JncxCai)))
        INaCa_ss = ((((self.factorINaCass*0.2)*self.GNaCa)*allo_ss)*((self.zna*JncxNa_ss)+(self.zca*JncxCa_ss)))
        Iion = (((((((((((((((INa+INaL)+Ito)+ICaL)+ICaNa)+ICaK)+IKr)+IKs)+IK1)+INaCa)+INaCa_ss)+INaK)+IbNa)+IKb)+Cai_row[:, self.IpCa_idx])+IbCa)

        # Complete Forward Euler Update
        Bcajsr = (1.0/(1.0+(((self.csqnmax*self.kmcsqn)/(self.kmcsqn+self.cajsr))/(self.kmcsqn+self.cajsr))))
        Bcass = (1.0/((1.0+(((self.BSRmax*self.KmBSR)/(self.KmBSR+self.cass))/(self.KmBSR+self.cass)))+(((self.BSLmax*self.KmBSL)/(self.KmBSL+self.cass))/(self.KmBSL+self.cass))))
        Jdiff = ((self.cass-(self.Cai))/0.2)
        JdiffK = ((self.kss-(self.Ki))/2.0)
        JdiffNa = ((self.nass-(self.Nai))/2.0)
        Jleak = ((0.0039375*self.cansr)/15.0)
        Jtr = ((self.cansr-(self.cajsr))/100.0)
        diff_CaMKt = (((self.aCaMK*CaMKb)*(CaMKb+self.CaMKt))-((self.bCaMK*self.CaMKt)))
        fJrelp = (1.0/(1.0+(self.KmCaMK/CaMKa)))
        fJupp = (1.0/(1.0+(self.KmCaMK/CaMKa)))
        km2n = self.jca
        kmn_cass_4 = ((((1.0+(self.Kmn/self.cass))*(1.0+(self.Kmn/self.cass)))*(1.0+(self.Kmn/self.cass)))*(1.0+(self.Kmn/self.cass)))
        Jrel_canidate = (((1.0-(fJrelp))*self.Jrelnp)+(fJrelp*self.Jrelp))
        Jup = ((((1.0-(fJupp))*Cai_row[:, self.Jupnp_idx])+(fJupp*Cai_row[:, self.Jupp_idx]))-(Jleak))
        anca = (1.0/((self.k2n/km2n)+kmn_cass_4))
        diff_Ki = ((((-(((((Ito+IKr)+IKs)+IK1)+IKb)-((2.0*INaK))))*self.Acap)/(self.F*self.vmyo))+((JdiffK*self.vss)/self.vmyo))
        diff_Nai = ((((-((((INa+INaL)+(3.0*INaCa))+(3.0*INaK))+IbNa))*self.Acap)/(self.F*self.vmyo))+((JdiffNa*self.vss)/self.vmyo))
        diff_kss = ((((-ICaK)*self.Acap)/(self.F*self.vss))-(JdiffK))
        diff_nass = ((((-(ICaNa+(3.0*INaCa_ss)))*self.Acap)/(self.F*self.vss))-(JdiffNa))
        Jrel = (torch.where(((Jrel_canidate*self.jrel_stiff_const)>self.cajsr), (self.cajsr/self.jrel_stiff_const), Jrel_canidate))
        diff_Cai = (Cai_row[:, self.Bcai_idx]*(((((-((Cai_row[:, self.IpCa_idx]+IbCa)-((2.0*INaCa))))*self.Acap)/((2.0*self.F)*self.vmyo))-(((Jup*self.vnsr)/self.vmyo)))+((Jdiff*self.vss)/self.vmyo)))
        diff_cansr = (Jup-(((Jtr*self.vjsr)/self.vnsr)))
        diff_nca = ((anca*self.k2n)-((self.nca*km2n)))
        diff_cajsr = (Bcajsr*(Jtr-(Jrel)))
        diff_cass = (Bcass*(((((-(ICaL-((2.0*INaCa_ss))))*self.Acap)/((2.0*self.F)*self.vss))+((Jrel*self.vjsr)/self.vss))-(Jdiff)))
        CaMKt_new = self.CaMKt+diff_CaMKt*self.dt
        Cai_new = self.Cai+diff_Cai*self.dt
        Ki_new = self.Ki+diff_Ki*self.dt
        Nai_new = self.Nai+diff_Nai*self.dt
        cajsr_new = self.cajsr+diff_cajsr*self.dt
        cansr_new = self.cansr+diff_cansr*self.dt
        cass_new = self.cass+diff_cass*self.dt
        kss_new = self.kss+diff_kss*self.dt
        nass_new = self.nass+diff_nass*self.dt
        nca_new = self.nca+diff_nca*self.dt

        # Complete Rush Larsen Update
        jrel_cajsr_term = ((((((((1.5/self.cajsr)*(1.5/self.cajsr))*(1.5/self.cajsr))*(1.5/self.cajsr))*(1.5/self.cajsr))*(1.5/self.cajsr))*(1.5/self.cajsr))*(1.5/self.cajsr))
        tau_rel_canidate = (self.bt/(1.0+(0.0123/self.cajsr)))
        tau_relp_canidate = (self.btp/(1.0+(0.0123/self.cajsr)))
        a_rush_larsen_B = V_row[:, self.a_rush_larsen_B_idx]
        d_rush_larsen_B = V_row[:, self.d_rush_larsen_B_idx]
        fcaf_rush_larsen_B = V_row[:, self.fcaf_rush_larsen_B_idx]
        fcas_rush_larsen_B = V_row[:, self.fcas_rush_larsen_B_idx]
        ff_rush_larsen_B = V_row[:, self.ff_rush_larsen_B_idx]
        fs_rush_larsen_B = V_row[:, self.fs_rush_larsen_B_idx]
        hL_rush_larsen_A = V_row[:, self.hL_rush_larsen_A_idx]
        hLp_rush_larsen_A = V_row[:, self.hLp_rush_larsen_A_idx]
        hf_rush_larsen_B = V_row[:, self.hf_rush_larsen_B_idx]
        hs_rush_larsen_B = V_row[:, self.hs_rush_larsen_B_idx]
        j_rush_larsen_B = V_row[:, self.j_rush_larsen_B_idx]
        jrel_ical_term = ((-ICaL)/(1.0+jrel_cajsr_term))
        mORd_rush_larsen_B = V_row[:, self.mORd_rush_larsen_B_idx]
        tau_rel = (torch.where((tau_rel_canidate<0.001), 0.001, tau_rel_canidate))
        tau_relp = (torch.where((tau_relp_canidate<0.001), 0.001, tau_relp_canidate))
        xk1_rush_larsen_B = V_row[:, self.xk1_rush_larsen_B_idx]
        xrf_rush_larsen_B = V_row[:, self.xrf_rush_larsen_B_idx]
        xrs_rush_larsen_B = V_row[:, self.xrs_rush_larsen_B_idx]
        xs1_rush_larsen_B = V_row[:, self.xs1_rush_larsen_B_idx]
        xs2_rush_larsen_B = V_row[:, self.xs2_rush_larsen_B_idx]
        Jrel_inf = (((self.a_rel*jrel_ical_term)*1.7) if (self.celltype==self.MCELL) else (self.a_rel*jrel_ical_term))
        Jrel_infp = (((self.a_relp*jrel_ical_term)*1.7) if (self.celltype==self.MCELL) else (self.a_relp*jrel_ical_term))
        a_rush_larsen_A = V_row[:, self.a_rush_larsen_A_idx]
        ap_rush_larsen_B = V_row[:, self.ap_rush_larsen_B_idx]
        d_rush_larsen_A = V_row[:, self.d_rush_larsen_A_idx]
        fcafp_rush_larsen_B = V_row[:, self.fcafp_rush_larsen_B_idx]
        ff_rush_larsen_A = V_row[:, self.ff_rush_larsen_A_idx]
        ffp_rush_larsen_B = V_row[:, self.ffp_rush_larsen_B_idx]
        fs_rush_larsen_A = V_row[:, self.fs_rush_larsen_A_idx]
        hTT2_rush_larsen_B = V_row[:, self.hTT2_rush_larsen_B_idx]
        hf_rush_larsen_A = V_row[:, self.hf_rush_larsen_A_idx]
        hs_rush_larsen_A = V_row[:, self.hs_rush_larsen_A_idx]
        hsp_rush_larsen_B = V_row[:, self.hsp_rush_larsen_B_idx]
        iF_rush_larsen_B = V_row[:, self.iF_rush_larsen_B_idx]
        iS_rush_larsen_B = V_row[:, self.iS_rush_larsen_B_idx]
        jTT2_rush_larsen_B = V_row[:, self.jTT2_rush_larsen_B_idx]
        j_rush_larsen_A = V_row[:, self.j_rush_larsen_A_idx]
        jp_rush_larsen_B = V_row[:, self.jp_rush_larsen_B_idx]
        mORd_rush_larsen_A = V_row[:, self.mORd_rush_larsen_A_idx]
        mTT2_rush_larsen_B = V_row[:, self.mTT2_rush_larsen_B_idx]
        tau_Jrelnp = tau_rel
        tau_Jrelp = tau_relp
        xk1_rush_larsen_A = V_row[:, self.xk1_rush_larsen_A_idx]
        xrf_rush_larsen_A = V_row[:, self.xrf_rush_larsen_A_idx]
        xrs_rush_larsen_A = V_row[:, self.xrs_rush_larsen_A_idx]
        xs1_rush_larsen_A = V_row[:, self.xs1_rush_larsen_A_idx]
        xs2_rush_larsen_A = V_row[:, self.xs2_rush_larsen_A_idx]
        Jrelnp_inf = Jrel_inf
        Jrelnp_rush_larsen_B = (torch.exp(((-self.dt)/tau_Jrelnp)))
        Jrelnp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_Jrelnp)))
        Jrelp_inf = Jrel_infp
        Jrelp_rush_larsen_B = (torch.exp(((-self.dt)/tau_Jrelp)))
        Jrelp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_Jrelp)))
        ap_rush_larsen_A = V_row[:, self.ap_rush_larsen_A_idx]
        fcaf_rush_larsen_A = V_row[:, self.fcaf_rush_larsen_A_idx]
        fcafp_rush_larsen_A = V_row[:, self.fcafp_rush_larsen_A_idx]
        fcas_rush_larsen_A = V_row[:, self.fcas_rush_larsen_A_idx]
        ffp_rush_larsen_A = V_row[:, self.ffp_rush_larsen_A_idx]
        hTT2_rush_larsen_A = V_row[:, self.hTT2_rush_larsen_A_idx]
        hsp_rush_larsen_A = V_row[:, self.hsp_rush_larsen_A_idx]
        iF_rush_larsen_A = V_row[:, self.iF_rush_larsen_A_idx]
        iFp_rush_larsen_B = V_row[:, self.iFp_rush_larsen_B_idx]
        iS_rush_larsen_A = V_row[:, self.iS_rush_larsen_A_idx]
        iSp_rush_larsen_B = V_row[:, self.iSp_rush_larsen_B_idx]
        jTT2_rush_larsen_A = V_row[:, self.jTT2_rush_larsen_A_idx]
        jca_rush_larsen_A = V_row[:, self.jca_rush_larsen_A_idx]
        jp_rush_larsen_A = V_row[:, self.jp_rush_larsen_A_idx]
        mTT2_rush_larsen_A = V_row[:, self.mTT2_rush_larsen_A_idx]
        Jrelnp_rush_larsen_A = ((-Jrelnp_inf)*Jrelnp_rush_larsen_C)
        Jrelp_rush_larsen_A = ((-Jrelp_inf)*Jrelp_rush_larsen_C)
        iFp_rush_larsen_A = V_row[:, self.iFp_rush_larsen_A_idx]
        iSp_rush_larsen_A = V_row[:, self.iSp_rush_larsen_A_idx]
        mL_rush_larsen_B = V_row[:, self.mL_rush_larsen_B_idx]
        mL_rush_larsen_A = V_row[:, self.mL_rush_larsen_A_idx]
        Jrelnp_new = Jrelnp_rush_larsen_A+Jrelnp_rush_larsen_B*self.Jrelnp
        Jrelp_new = Jrelp_rush_larsen_A+Jrelp_rush_larsen_B*self.Jrelp
        a_new = a_rush_larsen_A+a_rush_larsen_B*self.a
        ap_new = ap_rush_larsen_A+ap_rush_larsen_B*self.ap
        d_new = d_rush_larsen_A+d_rush_larsen_B*self.d
        fcaf_new = fcaf_rush_larsen_A+fcaf_rush_larsen_B*self.fcaf
        fcafp_new = fcafp_rush_larsen_A+fcafp_rush_larsen_B*self.fcafp
        fcas_new = fcas_rush_larsen_A+fcas_rush_larsen_B*self.fcas
        ff_new = ff_rush_larsen_A+ff_rush_larsen_B*self.ff
        ffp_new = ffp_rush_larsen_A+ffp_rush_larsen_B*self.ffp
        fs_new = fs_rush_larsen_A+fs_rush_larsen_B*self.fs
        hL_new = hL_rush_larsen_A+hL_rush_larsen_B*self.hL
        hLp_new = hLp_rush_larsen_A+hLp_rush_larsen_B*self.hLp
        hTT2_new = hTT2_rush_larsen_A+hTT2_rush_larsen_B*self.hTT2
        hf_new = hf_rush_larsen_A+hf_rush_larsen_B*self.hf
        hs_new = hs_rush_larsen_A+hs_rush_larsen_B*self.hs
        hsp_new = hsp_rush_larsen_A+hsp_rush_larsen_B*self.hsp
        iF_new = iF_rush_larsen_A+iF_rush_larsen_B*self.iF
        iFp_new = iFp_rush_larsen_A+iFp_rush_larsen_B*self.iFp
        iS_new = iS_rush_larsen_A+iS_rush_larsen_B*self.iS
        iSp_new = iSp_rush_larsen_A+iSp_rush_larsen_B*self.iSp
        j_new = j_rush_larsen_A+j_rush_larsen_B*self.j
        jTT2_new = jTT2_rush_larsen_A+jTT2_rush_larsen_B*self.jTT2
        jca_new = jca_rush_larsen_A+jca_rush_larsen_B*self.jca
        jp_new = jp_rush_larsen_A+jp_rush_larsen_B*self.jp
        mL_new = mL_rush_larsen_A+mL_rush_larsen_B*self.mL
        mORd_new = mORd_rush_larsen_A+mORd_rush_larsen_B*self.mORd
        mTT2_new = mTT2_rush_larsen_A+mTT2_rush_larsen_B*self.mTT2
        xk1_new = xk1_rush_larsen_A+xk1_rush_larsen_B*self.xk1
        xrf_new = xrf_rush_larsen_A+xrf_rush_larsen_B*self.xrf
        xrs_new = xrs_rush_larsen_A+xrs_rush_larsen_B*self.xrs
        xs1_new = xs1_rush_larsen_A+xs1_rush_larsen_B*self.xs1
        xs2_new = xs2_rush_larsen_A+xs2_rush_larsen_B*self.xs2

        # Finish the update
        self.CaMKt = torch.clamp(CaMKt_new, min=1e-9)
        self.Cai = torch.clamp(Cai_new, min=1e-9)
        self.Jrelnp = Jrelnp_new
        self.Jrelp = Jrelp_new
        self.Ki = torch.clamp(Ki_new, min=1e-9)
        self.Nai = torch.clamp(Nai_new, min=1e-9)
        self.a = torch.clamp(a_new, 0, 1)
        self.ap = torch.clamp(ap_new, 0, 1)
        self.cajsr = torch.clamp(cajsr_new, min=1e-9)
        self.cansr = torch.clamp(cansr_new, min=1e-9)
        self.cass = torch.clamp(cass_new, min=1e-9)
        self.d = torch.clamp(d_new, 0, 1)
        self.fcaf = torch.clamp(fcaf_new, 0, 1)
        self.fcafp = torch.clamp(fcafp_new, 0, 1)
        self.fcas = torch.clamp(fcas_new, 0, 1)
        self.ff = torch.clamp(ff_new, 0, 1)
        self.ffp = torch.clamp(ffp_new, 0, 1)
        self.fs = torch.clamp(fs_new, 0, 1)
        self.hL = torch.clamp(hL_new, 0, 1)
        self.hLp = torch.clamp(hLp_new, 0, 1)
        self.hTT2 = hTT2_new
        self.hf = torch.clamp(hf_new, 0, 1)
        self.hs = torch.clamp(hs_new, 0, 1)
        self.hsp = torch.clamp(hsp_new, 0, 1)
        self.iF = torch.clamp(iF_new, 0, 1)
        self.iFp = torch.clamp(iFp_new, 0, 1)
        self.iS = torch.clamp(iS_new, 0, 1)
        self.iSp = torch.clamp(iSp_new, 0, 1)
        self.j = torch.clamp(j_new, 0, 1)
        self.jTT2 = jTT2_new
        self.jca = torch.clamp(jca_new, 0, 1)
        self.jp = torch.clamp(jp_new, 0, 1)
        self.kss = torch.clamp(kss_new, min=1e-9)
        self.mL = torch.clamp(mL_new, 0, 1)
        self.mORd = mORd_new
        self.mTT2 = mTT2_new
        self.nass = torch.clamp(nass_new, min=1e-9)
        self.nca = torch.clamp(nca_new, min=1e-9)
        self.xk1 = torch.clamp(xk1_new, 0, 1)
        self.xrf = torch.clamp(xrf_new, 0, 1)
        self.xrs = torch.clamp(xrs_new, 0, 1)
        self.xs1 = torch.clamp(xs1_new, 0, 1)
        self.xs2 = torch.clamp(xs2_new, 0, 1)

        self.Cai = self.Cai * 1e3

        return -Iion


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    dt = 0.01
    dt_imp = float(np.float32(dt))   # limpet keeps the IMP time step in a float
    stimulus = 20
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    ionic = OHaraRudy(cell_type="ENDO", 
                     dt=dt_imp, 
                     device=device, 
                     dtype=torch.float64)
    V = ionic.initialize(n_nodes=1)

    V_list = []
    Cai_list = []

    ctime = 0.0
    for _ in range(int(1000/dt)):
        V_list.append([ctime, V.item()])
        Cai_list.append([ctime, ionic.Cai.item()])

        if ctime >= 0 and ctime < (0+2.0): 
            V = V + dt * stimulus
        dV = ionic.differentiate(V)
        V = V + dt * dV
        ctime += dt

    plt.figure()
    V_list = np.array(V_list)    
    plt.plot(V_list[:, 0], V_list[:, 1])
    plt.savefig("V_OHaraRudy.png")

    plt.figure()
    Cai_list = np.array(Cai_list)    
    plt.plot(Cai_list[:, 0], Cai_list[:, 1])
    plt.savefig("Cai.png")
