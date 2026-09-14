import torch
import torchcor as tc
import math
from math import exp, expm1, log, sqrt
from typing import Optional, List


@torch.jit.script
class Tomek20:
    def __init__(self, 
                 dt: float, 
                 region_ids: Optional[List[int]] = None, 
                 cell_type: str = "ENDO", 
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float64):
        
        self.name = "Tomek20"
        self.dt = dt
        self.region_ids = region_ids
        self.node_indices = torch.tensor([0])
        self.device = device
        # the DynSarc Newton step takes a central difference with ha = sqrt(2.2e-16),
        # which underflows in single precision: keep dtype at float64
        self.dtype = dtype

        self.cell_type = "ENDO" if cell_type is None else cell_type

        # Constants
        self.C1_init = (9.984733e-01 if cell_type == "EPI" else (9.983451e-01 if cell_type == "M" else 9.982511e-01))
        self.C2_init = (7.393045e-04 if cell_type == "EPI" else (7.086460e-04 if cell_type == "M" else 7.936020e-04))
        self.C3_init = (6.029079e-04 if cell_type == "EPI" else (5.910047e-04 if cell_type == "M" else 6.532143e-04))
        self.CD_init = 0.
        self.CaMKt_init = (1.273541e-02 if cell_type == "EPI" else (2.018820e-02 if cell_type == "M" else 1.095026e-02))
        self.Cai_mM_init = (6.621816e-05 if cell_type == "EPI" else (8.297576e-05 if cell_type == "M" else 7.453481e-05))
        self.I_init = (5.678255e-06 if cell_type == "EPI" else (1.064230e-05 if cell_type == "M" else 9.804083e-06))
        self.Jrel_np_init = (6.778827e-25 if cell_type == "EPI" else (-1.300486e-21 if cell_type == "M" else 1.808248e-22))
        self.Jrel_p_init = (-1.581941e-23 if cell_type == "EPI" else (-7.610714e-20 if cell_type == "M" else 4.358608e-21))
        self.Ki_init = (1.523639e+02 if cell_type == "EPI" else (1.567131e+02 if cell_type == "M" else 1.477115e+02))
        self.Nai_init = (1.340062e+01 if cell_type == "EPI" else (1.594867e+01 if cell_type == "M" else 1.239736e+01))
        self.O_init = (1.787783e-04 if cell_type == "EPI" else (3.445733e-04 if cell_type == "M" else 2.922449e-04))
        self.TRPN_init = 0.
        self.TmBlocked_init = 0.
        self.V_init = (-9.074563e+01 if cell_type == "EPI" else (-9.133918e+01 if cell_type == "M" else -8.974808e+01))
        self.XS_init = 0.
        self.XW_init = 0.
        self.ZETAS_init = 0.
        self.ZETAW_init = 0.
        self.aa_init = (8.320408e-04 if cell_type == "EPI" else (7.994042e-04 if cell_type == "M" else 8.899259e-04))
        self.ap_init = (4.239121e-04 if cell_type == "EPI" else (4.072777e-04 if cell_type == "M" else 4.534165e-04))
        self.cajsr_init = (1.805047e+00 if cell_type == "EPI" else (2.016415e+00 if cell_type == "M" else 1.525693e+00))
        self.cansr_init = (1.806794e+00 if cell_type == "EPI" else (2.012225e+00 if cell_type == "M" else 1.528001e+00))
        self.cass_init = (5.749921e-05 if cell_type == "EPI" else (6.642187e-05 if cell_type == "M" else 6.497341e-05))
        self.cli_init = (3.431721e+01 if cell_type == "EPI" else (4.891277e+01 if cell_type == "M" else 2.920698e+01))
        self.clss_init = (3.431719e+01 if cell_type == "EPI" else (4.891274e+01 if cell_type == "M" else 2.920696e+01))
        self.d_init = (-2.486527e-36 if cell_type == "EPI" else (-8.334604e-30 if cell_type == "M" else 1.588841e-31))
        self.delta_sl_init = 0.
        self.fcaf_init = 1.
        self.fcafp_init = 1.
        self.fcas_init = (9.999377e-01 if cell_type == "EPI" else (9.997540e-01 if cell_type == "M" else 9.999014e-01))
        self.ff_init = 1.
        self.ffp_init = 1.
        self.fs_init = (9.510602e-01 if cell_type == "EPI" else (9.183587e-01 if cell_type == "M" else 9.401791e-01))
        self.hL_init = (5.916536e-01 if cell_type == "EPI" else (5.986118e-01 if cell_type == "M" else 5.566017e-01))
        self.hLp_init = (3.476812e-01 if cell_type == "EPI" else (3.339899e-01 if cell_type == "M" else 3.115491e-01))
        self.h_init = (8.645148e-01 if cell_type == "EPI" else (8.739077e-01 if cell_type == "M" else 8.473267e-01))
        self.hp_init = (7.313656e-01 if cell_type == "EPI" else (7.478972e-01 if cell_type == "M" else 7.018454e-01))
        self.iF_init = (9.997242e-01 if cell_type == "EPI" else (9.997514e-01 if cell_type == "M" else 9.996716e-01))
        self.iFp_init = (9.997242e-01 if cell_type == "EPI" else (9.997514e-01 if cell_type == "M" else 9.996716e-01))
        self.iS_init = (9.997235e-01 if cell_type == "EPI" else (5.702538e-01 if cell_type == "M" else 5.988908e-01))
        self.iSp_init = (9.997241e-01 if cell_type == "EPI" else (6.351927e-01 if cell_type == "M" else 6.620692e-01))
        self.j_init = (8.644571e-01 if cell_type == "EPI" else (8.737841e-01 if cell_type == "M" else 8.471657e-01))
        self.jca_init = (9.999886e-01 if cell_type == "EPI" else (9.999743e-01 if cell_type == "M" else 9.999846e-01))
        self.jp_init = (8.643527e-01 if cell_type == "EPI" else (8.735375e-01 if cell_type == "M" else 8.469014e-01))
        self.kss_init = (1.523638e+02 if cell_type == "EPI" else (1.567130e+02 if cell_type == "M" else 1.477114e+02))
        self.length_init = 1.
        self.mL_init = (1.117969e-04 if cell_type == "EPI" else (9.987709e-05 if cell_type == "M" else 1.351203e-04))
        self.m_init = (5.253231e-04 if cell_type == "EPI" else (4.619565e-04 if cell_type == "M" else 6.517154e-04))
        self.nass_init = (1.340094e+01 if cell_type == "EPI" else (1.594922e+01 if cell_type == "M" else 1.239770e+01))
        self.nca_i_init = (5.272668e-04 if cell_type == "EPI" else (1.257861e-03 if cell_type == "M" else 8.326009e-04))
        self.nca_ss_init = (3.049523e-04 if cell_type == "EPI" else (5.336520e-04 if cell_type == "M" else 4.899378e-04))
        self.stretch_init = 1.
        self.xs1_init = (2.233584e-01 if cell_type == "EPI" else (2.642293e-01 if cell_type == "M" else 2.439590e-01))
        self.xs2_init = (1.418247e-04 if cell_type == "EPI" else (1.327348e-04 if cell_type == "M" else 1.586167e-04))
        self.Cai_init = (self.Cai_mM_init*1000.)

        # Parameters
        self.A_atp = 2.
        self.Aff = 0.6
        self.BSLmax = 1.124
        self.BSRmax = 0.047
        self.CaMKo = 0.05
        self.Cae = 1.8
        self.EKshift = 0.
        self.F = 96485.
        self.Fjunc = 1.
        self.GClCa = 0.2843
        self.GClb = 1.98e-3
        self.GK1_b = 0.6992
        self.GKb_b = 0.0189
        self.GKr_b = 0.0321
        self.GKs_b = 0.0011
        self.GNa = 11.7802
        self.GNaL_b = 0.0279
        self.Gncx_b = 0.0034
        self.GpCa = 5e-04
        self.Gto_b = 0.16
        self.H = 1e-7
        self.ICaL_fractionSS = 0.8
        self.INaCa_fractionSS = 0.35
        self.Jrel_b = 1.5378
        self.Jup_b = 1.0
        self.KKe = 0.3582
        self.KKi = 0.5
        self.KNae0 = 27.78
        self.KNai0 = 9.073
        self.K_atp = 0.25
        self.K_o_n = 5.
        self.KdClCa = 0.1
        self.Ke = 5.0
        self.Khp = 1.698e-7
        self.KmBSL = 0.0087
        self.KmBSR = 0.00087
        self.KmCaAct = 150e-6
        self.KmCaM = 0.0015
        self.KmCaMK = 0.15
        self.KmCap = 0.0005
        self.Kmgatp = 1.698e-7
        self.Kmn = 0.002
        self.Knap = 224.
        self.Kxkur = 292.
        self.L = 0.01
        self.MgADP = 0.05
        self.MgATP = 9.8
        self.Nae = 145.0
        self.PCa_b = 8.3757e-05
        self.PCab = 5.9194e-08
        self.PKNa = 0.01833
        self.PNab = 1.9239e-09
        self.Pnak_b = 15.4509
        self.R = 8314.
        self.SL0 = 1.79
        self.T = 310.
        self.TOT_A = 25.
        self.TRPN_n = 2.68927e+00
        self.Tref = 120.
        self.a = 2.1
        self.aCaMK = 0.05
        self.alpha_1 = 0.154375
        self.b = 9.1
        self.bCaMK = 0.00068
        self.beta_0 = 2.3
        self.beta_1 = -2.4
        self.beta_1_ = 0.1911
        self.bt = 4.75
        self.ca50 = 1.03119e+00
        self.cajsr_half = 1.7
        self.clo = 154.6
        self.cmdnmax_b = 0.05
        self.csqnmax = 10.
        self.delta = -0.155
        self.dielConstant = 74.
        self.dr = 1.90417e-01
        self.eP = 4.2
        self.eta_l = 200.
        self.eta_s = 20.
        self.fkatp = 0.0
        self.gamma = 0.0085
        self.gamma_wu = 0.615
        self.gkatp = 4.3195
        self.k1m = 182.4
        self.k1p = 949.5
        self.k2m = 39.4
        self.k2n = 500.
        self.k2p = 687.2
        self.k3m = 79300.
        self.k3p = 1899.
        self.k4m = 40.
        self.k4p = 639.
        self.kCaeff = 5e3
        self.kCaen = 1.5e6
        self.kasymm = 12.5
        self.kmcmdn = 0.00238
        self.kmcsqn = 0.8
        self.kmtrpn = 0.0005
        self.kna1 = 15.
        self.kna2 = 5.
        self.kna3 = 88.12
        self.koff = 0.1
        self.ktm_unblock = 1.0
        self.mu = 3.
        self.nperm = 7.42611e+00
        self.nu = 7.
        self.offset = 0.
        self.par_k = 7.
        self.perm50 = 3.40654e-01
        self.phi = 2.23
        self.qca = 0.167
        self.qna = 0.5224
        self.rad = 0.0011
        self.tauCa = 0.2
        self.tauCl = 2.0
        self.tauK = 2.0
        self.tauNa = 2.0
        self.tauTr = 60.0
        self.tau_hL = 200.
        self.tau_jca = 72.5
        self.trpnmax = 0.07
        self.vShift = 0.
        self.wca = 6e4
        self.wfrac = 0.5
        self.wna = 6e4
        self.wnaca = 5e3
        self.zca = 2.
        self.zcl = -1.
        self.zk = 1.
        self.zna = 1.
        self.celltype = 1. if cell_type == "EPI" else 2. if cell_type == "M" else 0.

        # 0 lookup tables

        # 53 states variables
        self.C1 = torch.tensor([self.C1_init])
        self.C2 = torch.tensor([self.C2_init])
        self.C3 = torch.tensor([self.C3_init])
        self.CD = torch.tensor([self.CD_init])
        self.CaMKt = torch.tensor([self.CaMKt_init])
        self.Cai = torch.tensor([self.Cai_init])
        self.Cai_mM = torch.tensor([self.Cai_mM_init])
        self.I = torch.tensor([self.I_init])
        self.Jrel_np = torch.tensor([self.Jrel_np_init])
        self.Jrel_p = torch.tensor([self.Jrel_p_init])
        self.Ki = torch.tensor([self.Ki_init])
        self.Nai = torch.tensor([self.Nai_init])
        self.O = torch.tensor([self.O_init])
        self.TRPN = torch.tensor([self.TRPN_init])
        self.TmBlocked = torch.tensor([self.TmBlocked_init])
        self.XS = torch.tensor([self.XS_init])
        self.XW = torch.tensor([self.XW_init])
        self.ZETAS = torch.tensor([self.ZETAS_init])
        self.ZETAW = torch.tensor([self.ZETAW_init])
        self.aa = torch.tensor([self.aa_init])
        self.ap = torch.tensor([self.ap_init])
        self.cajsr = torch.tensor([self.cajsr_init])
        self.cansr = torch.tensor([self.cansr_init])
        self.cass = torch.tensor([self.cass_init])
        self.cli = torch.tensor([self.cli_init])
        self.clss = torch.tensor([self.clss_init])
        self.d = torch.tensor([self.d_init])
        self.fcaf = torch.tensor([self.fcaf_init])
        self.fcafp = torch.tensor([self.fcafp_init])
        self.fcas = torch.tensor([self.fcas_init])
        self.ff = torch.tensor([self.ff_init])
        self.ffp = torch.tensor([self.ffp_init])
        self.fs = torch.tensor([self.fs_init])
        self.h = torch.tensor([self.h_init])
        self.hL = torch.tensor([self.hL_init])
        self.hLp = torch.tensor([self.hLp_init])
        self.hp = torch.tensor([self.hp_init])
        self.iF = torch.tensor([self.iF_init])
        self.iFp = torch.tensor([self.iFp_init])
        self.iS = torch.tensor([self.iS_init])
        self.iSp = torch.tensor([self.iSp_init])
        self.j = torch.tensor([self.j_init])
        self.jca = torch.tensor([self.jca_init])
        self.jp = torch.tensor([self.jp_init])
        self.kss = torch.tensor([self.kss_init])
        self.m = torch.tensor([self.m_init])
        self.mL = torch.tensor([self.mL_init])
        self.nass = torch.tensor([self.nass_init])
        self.nca_i = torch.tensor([self.nca_i_init])
        self.nca_ss = torch.tensor([self.nca_ss_init])
        self.stretch = torch.tensor([self.stretch_init])
        self.xs1 = torch.tensor([self.xs1_init])
        self.xs2 = torch.tensor([self.xs2_init])

        # Lambda is an input, delLambda and Tension are outputs (nodal externals)
        self.length = torch.tensor([self.length_init])
        self.delta_sl = torch.tensor([self.delta_sl_init])
        self.Tension = torch.tensor([0.])

        if not torch.jit.is_scripting():
            self.differentiate = torch.compile(
                self.differentiate,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )


    def interpolate(self, X, table, mn: float, mx: float, res: float, step: float, mx_idx: int):
        X = torch.clamp(X, mn, mx)
        idx = ((X - mn) * step).to(torch.long)
        lower_idx = torch.clamp(idx, 0, mx_idx - 1)
        higher_idx = lower_idx + 1
        lower_pos = lower_idx * res + mn
        w = ((X - lower_pos) / res).unsqueeze(1)
        return (1 - w) * table[lower_idx] + w * table[higher_idx]

    def construct_tables(self):
        # Tomek20Land17_DynSarc declares no lookup tables, every rate is evaluated on the fly
        pass

    def initialize(self, n_nodes: int):
        self.construct_tables()
        
        V = torch.full((n_nodes,), self.V_init, device=self.device, dtype=self.dtype)

        self.C1 = torch.full((n_nodes,), self.C1_init, device=self.device, dtype=self.dtype)
        self.C2 = torch.full((n_nodes,), self.C2_init, device=self.device, dtype=self.dtype)
        self.C3 = torch.full((n_nodes,), self.C3_init, device=self.device, dtype=self.dtype)
        self.CD = torch.full((n_nodes,), self.CD_init, device=self.device, dtype=self.dtype)
        self.CaMKt = torch.full((n_nodes,), self.CaMKt_init, device=self.device, dtype=self.dtype)
        self.Cai = torch.full((n_nodes,), self.Cai_init, device=self.device, dtype=self.dtype)
        self.Cai_mM = torch.full((n_nodes,), self.Cai_mM_init, device=self.device, dtype=self.dtype)
        self.I = torch.full((n_nodes,), self.I_init, device=self.device, dtype=self.dtype)
        self.Jrel_np = torch.full((n_nodes,), self.Jrel_np_init, device=self.device, dtype=self.dtype)
        self.Jrel_p = torch.full((n_nodes,), self.Jrel_p_init, device=self.device, dtype=self.dtype)
        self.Ki = torch.full((n_nodes,), self.Ki_init, device=self.device, dtype=self.dtype)
        self.Nai = torch.full((n_nodes,), self.Nai_init, device=self.device, dtype=self.dtype)
        self.O = torch.full((n_nodes,), self.O_init, device=self.device, dtype=self.dtype)
        self.TRPN = torch.full((n_nodes,), self.TRPN_init, device=self.device, dtype=self.dtype)
        self.TmBlocked = torch.full((n_nodes,), self.TmBlocked_init, device=self.device, dtype=self.dtype)
        self.XS = torch.full((n_nodes,), self.XS_init, device=self.device, dtype=self.dtype)
        self.XW = torch.full((n_nodes,), self.XW_init, device=self.device, dtype=self.dtype)
        self.ZETAS = torch.full((n_nodes,), self.ZETAS_init, device=self.device, dtype=self.dtype)
        self.ZETAW = torch.full((n_nodes,), self.ZETAW_init, device=self.device, dtype=self.dtype)
        self.aa = torch.full((n_nodes,), self.aa_init, device=self.device, dtype=self.dtype)
        self.ap = torch.full((n_nodes,), self.ap_init, device=self.device, dtype=self.dtype)
        self.cajsr = torch.full((n_nodes,), self.cajsr_init, device=self.device, dtype=self.dtype)
        self.cansr = torch.full((n_nodes,), self.cansr_init, device=self.device, dtype=self.dtype)
        self.cass = torch.full((n_nodes,), self.cass_init, device=self.device, dtype=self.dtype)
        self.cli = torch.full((n_nodes,), self.cli_init, device=self.device, dtype=self.dtype)
        self.clss = torch.full((n_nodes,), self.clss_init, device=self.device, dtype=self.dtype)
        self.d = torch.full((n_nodes,), self.d_init, device=self.device, dtype=self.dtype)
        self.fcaf = torch.full((n_nodes,), self.fcaf_init, device=self.device, dtype=self.dtype)
        self.fcafp = torch.full((n_nodes,), self.fcafp_init, device=self.device, dtype=self.dtype)
        self.fcas = torch.full((n_nodes,), self.fcas_init, device=self.device, dtype=self.dtype)
        self.ff = torch.full((n_nodes,), self.ff_init, device=self.device, dtype=self.dtype)
        self.ffp = torch.full((n_nodes,), self.ffp_init, device=self.device, dtype=self.dtype)
        self.fs = torch.full((n_nodes,), self.fs_init, device=self.device, dtype=self.dtype)
        self.h = torch.full((n_nodes,), self.h_init, device=self.device, dtype=self.dtype)
        self.hL = torch.full((n_nodes,), self.hL_init, device=self.device, dtype=self.dtype)
        self.hLp = torch.full((n_nodes,), self.hLp_init, device=self.device, dtype=self.dtype)
        self.hp = torch.full((n_nodes,), self.hp_init, device=self.device, dtype=self.dtype)
        self.iF = torch.full((n_nodes,), self.iF_init, device=self.device, dtype=self.dtype)
        self.iFp = torch.full((n_nodes,), self.iFp_init, device=self.device, dtype=self.dtype)
        self.iS = torch.full((n_nodes,), self.iS_init, device=self.device, dtype=self.dtype)
        self.iSp = torch.full((n_nodes,), self.iSp_init, device=self.device, dtype=self.dtype)
        self.j = torch.full((n_nodes,), self.j_init, device=self.device, dtype=self.dtype)
        self.jca = torch.full((n_nodes,), self.jca_init, device=self.device, dtype=self.dtype)
        self.jp = torch.full((n_nodes,), self.jp_init, device=self.device, dtype=self.dtype)
        self.kss = torch.full((n_nodes,), self.kss_init, device=self.device, dtype=self.dtype)
        self.m = torch.full((n_nodes,), self.m_init, device=self.device, dtype=self.dtype)
        self.mL = torch.full((n_nodes,), self.mL_init, device=self.device, dtype=self.dtype)
        self.nass = torch.full((n_nodes,), self.nass_init, device=self.device, dtype=self.dtype)
        self.nca_i = torch.full((n_nodes,), self.nca_i_init, device=self.device, dtype=self.dtype)
        self.nca_ss = torch.full((n_nodes,), self.nca_ss_init, device=self.device, dtype=self.dtype)
        self.stretch = torch.full((n_nodes,), self.stretch_init, device=self.device, dtype=self.dtype)
        self.xs1 = torch.full((n_nodes,), self.xs1_init, device=self.device, dtype=self.dtype)
        self.xs2 = torch.full((n_nodes,), self.xs2_init, device=self.device, dtype=self.dtype)

        self.length = torch.full((n_nodes,), self.length_init, device=self.device, dtype=self.dtype)
        self.delta_sl = torch.full((n_nodes,), self.delta_sl_init, device=self.device, dtype=self.dtype)
        self.Tension = torch.zeros(n_nodes, device=self.device, dtype=self.dtype)

        return V

    def differentiate(self, V):
        # Define the constants that depend on the parameters.
        ha = 1.4832397e-8
        delta_epi = 1.
        A = (((0.25*self.TOT_A)/(((1.-(self.dr))*self.wfrac)+self.dr))*(self.dr/0.25))
        Afs = (1.-(self.Aff))
        Ageo = ((((2.*3.14)*self.rad)*self.rad)+(((2.*3.14)*self.rad)*self.L))
        GK1 = ((self.GK1_b*1.2) if (self.celltype==1.) else (((self.GK1_b*1.3) if (self.celltype==2.) else self.GK1_b)))
        GKb = ((self.GKb_b*0.6) if (self.celltype==1.) else self.GKb_b)
        GKr = ((self.GKr_b*1.3) if (self.celltype==1.) else (((self.GKr_b*0.8) if (self.celltype==2.) else self.GKr_b)))
        GKs = ((self.GKs_b*1.4) if (self.celltype==1.) else self.GKs_b)
        GNaL = ((self.GNaL_b*0.6) if (self.celltype==1.) else self.GNaL_b)
        Gncx = ((self.Gncx_b*1.1) if (self.celltype==1.) else (((self.Gncx_b*1.4) if (self.celltype==2.) else self.Gncx_b)))
        Gto = ((self.Gto_b*2.) if (self.celltype==1.) else (((self.Gto_b*2.) if (self.celltype==2.) else self.Gto_b)))
        Io = ((0.5*(((self.Nae+self.Ke)+self.clo)+(4.*self.Cae)))/1000.)
        PCa = ((self.PCa_b*1.2) if (self.celltype==1.) else (((self.PCa_b*2.0) if (self.celltype==2.) else self.PCa_b)))
        Pnak = ((self.Pnak_b*0.9) if (self.celltype==1.) else (((self.Pnak_b*0.7) if (self.celltype==2.) else self.Pnak_b)))
        XSSS = (self.dr*0.5)
        XWSS = (((1.-(self.dr))*self.wfrac)*0.5)
        a2 = self.k2p
        a4 = (((self.k4p*self.MgATP)/self.Kmgatp)/(1.+(self.MgATP/self.Kmgatp)))
        a_rel = (0.5*self.bt)
        akik = (pow((self.Ke/self.K_o_n),0.24))
        b1 = (self.k1m*self.MgADP)
        bkik = (1./(1.+((self.A_atp/self.K_atp)*(self.A_atp/self.K_atp))))
        btp = (1.25*self.bt)
        cmdnmax = ((self.cmdnmax_b*1.3) if (self.celltype==1.) else self.cmdnmax_b)
        constA = (1.82e6*(pow((self.dielConstant*self.T),-1.5)))
        h10_i = ((self.kasymm+1.)+((self.Nae/self.kna1)*(1.+(self.Nae/self.kna2))))
        h10_ss = ((self.kasymm+1.)+((self.Nae/self.kna1)*(1.+(self.Nae/self.kna2))))
        hL_rush_larsen_B = (exp(((-self.dt)/self.tau_hL)))
        hL_rush_larsen_C = (expm1(((-self.dt)/self.tau_hL)))
        jca_rush_larsen_B = (exp(((-self.dt)/self.tau_jca)))
        jca_rush_larsen_C = (expm1(((-self.dt)/self.tau_jca)))
        k2_i = self.kCaeff
        k2_ss = self.kCaeff
        k5_i = self.kCaeff
        k5_ss = self.kCaeff
        k_uw = (0.026*self.nu)
        k_ws = (0.004*self.mu)
        tau_hLp = (3.*self.tau_hL)
        upScale = (1.3 if (self.celltype==1.) else 1.)
        vcell = ((((1000.*3.14)*self.rad)*self.rad)*self.L)
        Acap = (2.*Ageo)
        PCaK = (3.574e-4*PCa)
        PCaNa = (0.00125*PCa)
        PCap = (1.1*PCa)
        a_relp = (0.5*btp)
        cds = ((((self.phi*k_ws)*(1.-(self.dr)))*self.wfrac)/self.dr)
        cdw = ((((self.phi*k_uw)*(1.-(self.dr)))*(1.-(self.wfrac)))/((1.-(self.dr))*self.wfrac))
        gamma_Cae = (exp((((-constA)*4.)*(((sqrt(Io))/(1.+(sqrt(Io))))-((0.3*Io))))))
        gamma_Ke = (exp(((-constA)*(((sqrt(Io))/(1.+(sqrt(Io))))-((0.3*Io))))))
        gamma_Nae = (exp(((-constA)*(((sqrt(Io))/(1.+(sqrt(Io))))-((0.3*Io))))))
        h11_i = ((self.Nae*self.Nae)/((h10_i*self.kna1)*self.kna2))
        h11_ss = ((self.Nae*self.Nae)/((h10_ss*self.kna1)*self.kna2))
        h12_i = (1./h10_i)
        h12_ss = (1./h10_ss)
        hLp_rush_larsen_B = (exp(((-self.dt)/tau_hLp)))
        hLp_rush_larsen_C = (expm1(((-self.dt)/tau_hLp)))
        k_su = ((k_ws*((1./self.dr)-(1.)))*self.wfrac)
        k_wu = ((k_uw*((1./self.wfrac)-(1.)))-(k_ws))
        ktm_block = (((self.ktm_unblock*(pow(self.perm50,self.nperm)))*0.5)/((0.5-(XSSS))-(XWSS)))
        vjsr = (0.0048*vcell)
        vmyo = (0.68*vcell)
        vnsr = (0.0552*vcell)
        vss = (0.02*vcell)
        PCaKp = (3.574e-4*PCap)
        PCaNap = (0.00125*PCap)
        k1_i = ((h12_i*self.Cae)*self.kCaen)
        k1_ss = ((h12_ss*self.Cae)*self.kCaen)


        # Compute storevars and external modvars
        Afcaf = (0.3+(0.6/(1.+(torch.exp(((V-(10.))/10.))))))
        AiF = (1./(1.+(torch.exp((((V+self.EKshift)-(213.6))/151.2)))))
        CaMKb = ((self.CaMKo*(1.-(self.CaMKt)))/(1.+(self.KmCaM/self.cass)))
        Cai_new = (self.Cai_mM*1000.)
        ECl = (((self.R*self.T)/(self.zcl*self.F))*(torch.log((self.clo/self.cli))))
        EClss = (((self.R*self.T)/(self.zcl*self.F))*(torch.log((self.clo/self.clss))))
        EK = (((self.R*self.T)/(self.zk*self.F))*(torch.log((self.Ke/self.Ki))))
        EKs = (((self.R*self.T)/(self.zk*self.F))*(torch.log(((self.Ke+(self.PKNa*self.Nae))/(self.Ki+(self.PKNa*self.Nai))))))
        ENa = (((self.R*self.T)/(self.zna*self.F))*(torch.log((self.Nae/self.Nai))))
        Ii = ((0.5*(((self.Nai+self.Ki)+self.cli)+(4.*self.Cai_mM)))/1000.)
        IpCa = ((self.GpCa*self.Cai_mM)/(self.KmCap+self.Cai_mM))
        Iss = ((0.5*(((self.nass+self.kss)+self.clss)+(4.*self.cass)))/1000.)
        KsCa = (1.+(0.6/(1.+(torch.pow((3.8e-5/self.Cai_mM),1.4)))))
        P = (self.eP/(((1.+(self.H/self.Khp))+(self.Nai/self.Knap))+(self.Ki/self.Kxkur)))
        allo_i = (1./(1.+((self.KmCaAct/self.Cai_mM)*(self.KmCaAct/self.Cai_mM))))
        allo_ss = (1./(1.+((self.KmCaAct/self.cass)*(self.KmCaAct/self.cass))))
        f = ((self.Aff*self.ff)+(Afs*self.fs))
        fp = ((self.Aff*self.ffp)+(Afs*self.fs))
        h4_i = (1.+((self.Nai/self.kna1)*(1.+(self.Nai/self.kna2))))
        h4_ss = (1.+((self.nass/self.kna1)*(1.+(self.nass/self.kna2))))
        lambda_m = (torch.where((self.stretch>1.2), 1.2, self.stretch))
        vffrt = (((V*self.F)*self.F)/(self.R*self.T))
        vfrt = ((V*self.F)/(self.R*self.T))
        # the GHK terms below are 0/0 at V=0 (and underflow below ~1e-20); nudge off exact zero, keeping vffrt = F*vfrt
        _at_zero = (torch.abs(vfrt) < 1e-20)
        vfrt = torch.where(_at_zero, torch.full_like(vfrt, 1e-12), vfrt)
        vffrt = torch.where(_at_zero, torch.full_like(vffrt, 1e-12 * self.F), vffrt)
        xkb = (1./(1.+(torch.exp(((-(V-(10.8968)))/23.9871)))))
        Afcas = (1.-(Afcaf))
        AiS = (1.-(AiF))
        C = (lambda_m-(self.length))
        CaMKa = (CaMKb+self.CaMKt)
        IClCa_junc = (((self.Fjunc*self.GClCa)/(1.+(self.KdClCa/self.cass)))*(V-(EClss)))
        IClCa_sl = ((((1.-(self.Fjunc))*self.GClCa)/(1.+(self.KdClCa/self.Cai_mM)))*(V-(ECl)))
        IClb = (self.GClb*(V-(ECl)))
        IKb = ((GKb*xkb)*(V-(EK)))
        IKr = (((GKr*(sqrt((self.Ke/5.))))*self.O)*(V-(EK)))
        IKs = ((((GKs*KsCa)*self.xs1)*self.xs2)*(V-(EKs)))
        INab = (((self.PNab*vffrt)*((self.Nai*(torch.exp(vfrt)))-(self.Nae)))/((torch.expm1(vfrt))))
        Ikatp = ((((self.fkatp*self.gkatp)*akik)*bkik)*(V-(EK)))
        KNae = (self.KNae0*(torch.exp((((1.-(self.delta))*vfrt)/3.))))
        KNai = (self.KNai0*(torch.exp(((self.delta*vfrt)/3.))))
        aK1 = (4.094/(1.+(torch.exp((0.1217*((V-(EK))-(49.934)))))))
        b3 = (((self.k3m*P)*self.H)/(1.+(self.MgATP/self.Kmgatp)))
        bK1 = (((15.72*(torch.exp((0.0674*((V-(EK))-(3.257))))))+(torch.exp((0.0618*((V-(EK))-(594.31))))))/(1.+(torch.exp((-0.1629*((V-(EK))+14.207))))))
        gamma_Ki = (torch.exp(((-constA)*(((torch.sqrt(Ii))/(1.+(torch.sqrt(Ii))))-((0.3*Ii))))))
        gamma_Nai = (torch.exp(((-constA)*(((torch.sqrt(Ii))/(1.+(torch.sqrt(Ii))))-((0.3*Ii))))))
        gamma_cai = (torch.exp((((-constA)*4.)*(((torch.sqrt(Ii))/(1.+(torch.sqrt(Ii))))-((0.3*Ii))))))
        gamma_cass = (torch.exp((((-constA)*4.)*(((torch.sqrt(Iss))/(1.+(torch.sqrt(Iss))))-((0.3*Iss))))))
        gamma_kss = (torch.exp(((-constA)*(((torch.sqrt(Iss))/(1.+(torch.sqrt(Iss))))-((0.3*Iss))))))
        gamma_nass = (torch.exp(((-constA)*(((torch.sqrt(Iss))/(1.+(torch.sqrt(Iss))))-((0.3*Iss))))))
        h5_i = ((self.Nai*self.Nai)/((h4_i*self.kna1)*self.kna2))
        h5_ss = ((self.nass*self.nass)/((h4_ss*self.kna1)*self.kna2))
        h6_i = (1./h4_i)
        h6_ss = (1./h4_ss)
        hca = (torch.exp((self.qca*vfrt)))
        hna = (torch.exp((self.qna*vfrt)))
        lambda_0 = lambda_m
        lambda_s = (torch.where((lambda_m>0.87), 0.87, lambda_m))
        Cme1 = ((lambda_0-(ha))-(self.length))
        Cpe1 = ((lambda_0+ha)-(self.length))
        F1 = ((torch.expm1((self.b*C))))
        ICab = ((((self.PCab*4.)*vffrt)*(((gamma_cai*self.Cai_mM)*(torch.exp((2.*vfrt))))-((gamma_Cae*self.Cae))))/((torch.expm1((2.*vfrt)))))
        IClCa = (IClCa_junc+IClCa_sl)
        K1ss = (aK1/(aK1+bK1))
        PhiCaK_i = ((vffrt*(((gamma_Ki*self.Ki)*(torch.exp(vfrt)))-((gamma_Ke*self.Ke))))/((torch.expm1(vfrt))))
        PhiCaK_ss = ((vffrt*(((gamma_kss*self.kss)*(torch.exp(vfrt)))-((gamma_Ke*self.Ke))))/((torch.expm1(vfrt))))
        PhiCaL_i = (((4.*vffrt)*(((gamma_cai*self.Cai_mM)*(torch.exp((2.*vfrt))))-((gamma_Cae*self.Cae))))/((torch.expm1((2.*vfrt)))))
        PhiCaL_ss = (((4.*vffrt)*(((gamma_cass*self.cass)*(torch.exp((2.*vfrt))))-((gamma_Cae*self.Cae))))/((torch.expm1((2.*vfrt)))))
        PhiCaNa_i = ((vffrt*(((gamma_Nai*self.Nai)*(torch.exp(vfrt)))-((gamma_Nae*self.Nae))))/((torch.expm1(vfrt))))
        PhiCaNa_ss = ((vffrt*(((gamma_nass*self.nass)*(torch.exp(vfrt)))-((gamma_Nae*self.Nae))))/((torch.expm1(vfrt))))
        a1 = ((self.k1p*(((self.Nai/KNai)*(self.Nai/KNai))*(self.Nai/KNai)))/(((((1.+(self.Nai/KNai))*(1.+(self.Nai/KNai)))*(1.+(self.Nai/KNai)))+((1.+(self.Ki/self.KKi))*(1.+(self.Ki/self.KKi))))-(1.)))
        a3 = ((self.k3p*((self.Ke/self.KKe)*(self.Ke/self.KKe)))/(((((1.+(self.Nae/KNae))*(1.+(self.Nae/KNae)))*(1.+(self.Nae/KNae)))+((1.+(self.Ke/self.KKe))*(1.+(self.Ke/self.KKe))))-(1.)))
        b2 = ((self.k2m*(((self.Nae/KNae)*(self.Nae/KNae))*(self.Nae/KNae)))/(((((1.+(self.Nae/KNae))*(1.+(self.Nae/KNae)))*(1.+(self.Nae/KNae)))+((1.+(self.Ke/self.KKe))*(1.+(self.Ke/self.KKe))))-(1.)))
        b4 = ((self.k4m*((self.Ki/self.KKi)*(self.Ki/self.KKi)))/(((((1.+(self.Nai/KNai))*(1.+(self.Nai/KNai)))*(1.+(self.Nai/KNai)))+((1.+(self.Ki/self.KKi))*(1.+(self.Ki/self.KKi))))-(1.)))
        eta_p = (torch.where(((C-(self.CD))<0.), self.eta_s, self.eta_l))
        fICaLp = (1./(1.+(self.KmCaMK/CaMKa)))
        fINaLp = (1./(1.+(self.KmCaMK/CaMKa)))
        fINap = (1./(1.+(self.KmCaMK/CaMKa)))
        fItop = (1./(1.+(self.KmCaMK/CaMKa)))
        fca = ((Afcaf*self.fcaf)+(Afcas*self.fcas))
        fcap = ((Afcaf*self.fcafp)+(Afcas*self.fcas))
        h1_i = (1.+((self.Nai/self.kna3)*(1.+hna)))
        h1_ss = (1.+((self.nass/self.kna3)*(1.+hna)))
        h7_i = (1.+((self.Nae/self.kna3)*(1.+(1./hna))))
        h7_ss = (1.+((self.Nae/self.kna3)*(1.+(1./hna))))
        iii = ((AiF*self.iF)+(AiS*self.iS))
        ip = ((AiF*self.iFp)+(AiS*self.iSp))
        k6_i = ((h6_i*self.Cai_mM)*self.kCaen)
        k6_ss = ((h6_ss*self.cass)*self.kCaen)
        lambda_s_me1 = (torch.where(((lambda_0-(ha))>0.87), 0.87, (lambda_0-(ha))))
        lambda_s_pe1 = (torch.where(((lambda_0+ha)>0.87), 0.87, (lambda_0+ha)))
        overlap = (1.+(self.beta_0*((lambda_m+lambda_s)-(1.87))))
        ICaK_i = ((1.-(self.ICaL_fractionSS))*((((((1.-(fICaLp))*PCaK)*PhiCaK_i)*self.d)*((f*(1.-(self.nca_i)))+((self.jca*fca)*self.nca_i)))+((((fICaLp*PCaKp)*PhiCaK_i)*self.d)*((fp*(1.-(self.nca_i)))+((self.jca*fcap)*self.nca_i)))))
        ICaK_ss = (self.ICaL_fractionSS*((((((1.-(fICaLp))*PCaK)*PhiCaK_ss)*self.d)*((f*(1.-(self.nca_ss)))+((self.jca*fca)*self.nca_ss)))+((((fICaLp*PCaKp)*PhiCaK_ss)*self.d)*((fp*(1.-(self.nca_ss)))+((self.jca*fcap)*self.nca_ss)))))
        ICaL_i = ((1.-(self.ICaL_fractionSS))*((((((1.-(fICaLp))*PCa)*PhiCaL_i)*self.d)*((f*(1.-(self.nca_i)))+((self.jca*fca)*self.nca_i)))+((((fICaLp*PCap)*PhiCaL_i)*self.d)*((fp*(1.-(self.nca_i)))+((self.jca*fcap)*self.nca_i)))))
        ICaL_ss = (self.ICaL_fractionSS*((((((1.-(fICaLp))*PCa)*PhiCaL_ss)*self.d)*((f*(1.-(self.nca_ss)))+((self.jca*fca)*self.nca_ss)))+((((fICaLp*PCap)*PhiCaL_ss)*self.d)*((fp*(1.-(self.nca_ss)))+((self.jca*fcap)*self.nca_ss)))))
        ICaNa_i = ((1.-(self.ICaL_fractionSS))*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_i)*self.d)*((f*(1.-(self.nca_i)))+((self.jca*fca)*self.nca_i)))+((((fICaLp*PCaNap)*PhiCaNa_i)*self.d)*((fp*(1.-(self.nca_i)))+((self.jca*fcap)*self.nca_i)))))
        ICaNa_ss = (self.ICaL_fractionSS*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_ss)*self.d)*((f*(1.-(self.nca_ss)))+((self.jca*fca)*self.nca_ss)))+((((fICaLp*PCaNap)*PhiCaNa_ss)*self.d)*((fp*(1.-(self.nca_ss)))+((self.jca*fcap)*self.nca_ss)))))
        IK1 = (((GK1*(sqrt((self.Ke/5.))))*K1ss)*(V-(EK)))
        INa = (((self.GNa*(V-(ENa)))*((self.m*self.m)*self.m))*((((1.-(fINap))*self.h)*self.j)+((fINap*self.hp)*self.jp)))
        INaL = (((GNaL*(V-(ENa)))*self.mL)*(((1.-(fINaLp))*self.hL)+(fINaLp*self.hLp)))
        Ito = ((Gto*(V-(EK)))*((((1.-(fItop))*self.aa)*iii)+((fItop*self.ap)*ip)))
        Tp_me1 = (self.a*(((torch.expm1((self.b*Cme1))))+(self.par_k*(Cme1-(self.CD)))))
        Tp_pe1 = (self.a*(((torch.expm1((self.b*Cpe1))))+(self.par_k*(Cpe1-(self.CD)))))
        dCd_dt = ((self.par_k*(C-(self.CD)))/eta_p)
        h2_i = ((self.Nai*hna)/(self.kna3*h1_i))
        h2_ss = ((self.nass*hna)/(self.kna3*h1_ss))
        h3_i = (1./h1_i)
        h3_ss = (1./h1_ss)
        h8_i = (self.Nae/((self.kna3*hna)*h7_i))
        h8_ss = (self.Nae/((self.kna3*hna)*h7_ss))
        h9_i = (1./h7_i)
        h9_ss = (1./h7_ss)
        h_lambda = (torch.where((overlap>0.), overlap, 0.))
        overlap_me1 = (1.+(self.beta_0*(((lambda_0-(ha))+lambda_s_me1)-(1.87))))
        overlap_pe1 = (1.+(self.beta_0*(((lambda_0+ha)+lambda_s_pe1)-(1.87))))
        x1 = (((((a4*a1)*a2)+((b2*b4)*b3))+((a2*b4)*b3))+((b3*a1)*a2))
        x2 = (((((b2*b1)*b4)+((a1*a2)*a3))+((a3*b1)*b4))+((a2*a3)*b4))
        x3 = (((((a2*a3)*a4)+((b3*b2)*b1))+((b2*b1)*a4))+((a3*a4)*b1))
        x4 = (((((b4*b3)*b2)+((a3*a4)*a1))+((b2*a4)*a1))+((b3*b2)*a1))
        E1 = (x1/(((x1+x2)+x3)+x4))
        E2 = (x2/(((x1+x2)+x3)+x4))
        E3 = (x3/(((x1+x2)+x3)+x4))
        E4 = (x4/(((x1+x2)+x3)+x4))
        Fd = (eta_p*dCd_dt)
        ICaK = (ICaK_ss+ICaK_i)
        ICaL = (ICaL_ss+ICaL_i)
        ICaNa = (ICaNa_ss+ICaNa_i)
        Ka = ((h_lambda*(self.Tref/self.dr))*((A*self.XS)+(A*self.XW)))
        Ta = ((h_lambda*(self.Tref/self.dr))*(((self.ZETAS+1.)*self.XS)+(self.ZETAW*self.XW)))
        h_lambda_me1 = (torch.where((overlap_me1>0.), overlap_me1, 0.))
        h_lambda_pe1 = (torch.where((overlap_pe1>0.), overlap_pe1, 0.))
        k3p_i = (h9_i*self.wca)
        k3p_ss = (h9_ss*self.wca)
        k3pp_i = (h8_i*self.wnaca)
        k3pp_ss = (h8_ss*self.wnaca)
        k4p_i = ((h3_i*self.wca)/hca)
        k4p_ss = ((h3_ss*self.wca)/hca)
        k4pp_i = (h2_i*self.wnaca)
        k4pp_ss = (h2_ss*self.wnaca)
        k7_i = ((h5_i*h2_i)*self.wna)
        k7_ss = ((h5_ss*h2_ss)*self.wna)
        k8_i = ((h8_i*h11_i)*self.wna)
        k8_ss = ((h8_ss*h11_ss)*self.wna)
        JnakK = (2.*((E4*b1)-((E3*a1))))
        JnakNa = (3.*((E1*a3)-((E2*b3))))
        Ka_me1 = ((h_lambda_me1*(self.Tref/self.dr))*((A*self.XS)+(A*self.XW)))
        Ka_pe1 = ((h_lambda_pe1*(self.Tref/self.dr))*((A*self.XS)+(A*self.XW)))
        Ta_me1 = ((h_lambda_me1*(self.Tref/self.dr))*(((self.ZETAS+1.)*self.XS)+(self.ZETAW*self.XW)))
        Ta_pe1 = ((h_lambda_pe1*(self.Tref/self.dr))*(((self.ZETAS+1.)*self.XS)+(self.ZETAW*self.XW)))
        Tp = (self.a*(F1+Fd))
        k3_i = (k3p_i+k3pp_i)
        k3_ss = (k3p_ss+k3pp_ss)
        k4_i = (k4p_i+k4pp_i)
        k4_ss = (k4p_ss+k4pp_ss)
        INaK = (Pnak*((self.zna*JnakNa)+(self.zk*JnakK)))
        derivative_1 = ((((((Tp_pe1+Ta_pe1)+(Ka_pe1*((lambda_0+ha)-(lambda_m))))-(Tp_me1))-(Ta_me1))-((Ka_me1*((lambda_0-(ha))-(lambda_m)))))/(2.*ha))
        func_1 = ((Tp+Ta)+(Ka*(lambda_0-(lambda_m))))
        x1_i = (((k2_i*k4_i)*(k7_i+k6_i))+((k5_i*k7_i)*(k2_i+k3_i)))
        x1_ss = (((k2_ss*k4_ss)*(k7_ss+k6_ss))+((k5_ss*k7_ss)*(k2_ss+k3_ss)))
        x2_i = (((k1_i*k7_i)*(k4_i+k5_i))+((k4_i*k6_i)*(k1_i+k8_i)))
        x2_ss = (((k1_ss*k7_ss)*(k4_ss+k5_ss))+((k4_ss*k6_ss)*(k1_ss+k8_ss)))
        x3_i = (((k1_i*k3_i)*(k7_i+k6_i))+((k8_i*k6_i)*(k2_i+k3_i)))
        x3_ss = (((k1_ss*k3_ss)*(k7_ss+k6_ss))+((k8_ss*k6_ss)*(k2_ss+k3_ss)))
        x4_i = (((k2_i*k8_i)*(k4_i+k5_i))+((k3_i*k5_i)*(k1_i+k8_i)))
        x4_ss = (((k2_ss*k8_ss)*(k4_ss+k5_ss))+((k3_ss*k5_ss)*(k1_ss+k8_ss)))
        E1_i = (x1_i/(((x1_i+x2_i)+x3_i)+x4_i))
        E1_ss = (x1_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss))
        E2_i = (x2_i/(((x1_i+x2_i)+x3_i)+x4_i))
        E2_ss = (x2_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss))
        E3_i = (x3_i/(((x1_i+x2_i)+x3_i)+x4_i))
        E3_ss = (x3_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss))
        E4_i = (x4_i/(((x1_i+x2_i)+x3_i)+x4_i))
        E4_ss = (x4_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss))
        lambda_1 = (lambda_0-((func_1/derivative_1)))
        C_2 = (lambda_1-(self.length))
        Cme2 = ((lambda_1-(ha))-(self.length))
        Cpe2 = ((lambda_1+ha)-(self.length))
        JncxCa_i = ((E2_i*k2_i)-((E1_i*k1_i)))
        JncxCa_ss = ((E2_ss*k2_ss)-((E1_ss*k1_ss)))
        JncxNa_i = (((3.*((E4_i*k7_i)-((E1_i*k8_i))))+(E3_i*k4pp_i))-((E2_i*k3pp_i)))
        JncxNa_ss = (((3.*((E4_ss*k7_ss)-((E1_ss*k8_ss))))+(E3_ss*k4pp_ss))-((E2_ss*k3pp_ss)))
        lambda_s_2 = (torch.where((lambda_1>0.87), 0.87, lambda_1))
        lambda_s_me2 = (torch.where(((lambda_1-(ha))>0.87), 0.87, (lambda_1-(ha))))
        lambda_s_pe2 = (torch.where(((lambda_1+ha)>0.87), 0.87, (lambda_1+ha)))
        INaCa_i = ((((1.-(self.INaCa_fractionSS))*Gncx)*allo_i)*((self.zna*JncxNa_i)+(self.zca*JncxCa_i)))
        INaCa_ss = (((self.INaCa_fractionSS*Gncx)*allo_ss)*((self.zna*JncxNa_ss)+(self.zca*JncxCa_ss)))
        Tp_2 = (self.a*(((torch.expm1((self.b*C_2))))+(self.par_k*(C_2-(self.CD)))))
        Tp_me2 = (self.a*(((torch.expm1((self.b*Cme2))))+(self.par_k*(Cme2-(self.CD)))))
        Tp_pe2 = (self.a*(((torch.expm1((self.b*Cpe2))))+(self.par_k*(Cpe2-(self.CD)))))
        overlap_2 = (1.+(self.beta_0*((lambda_1+lambda_s_2)-(1.87))))
        overlap_me2 = (1.+(self.beta_0*(((lambda_1-(ha))+lambda_s_me2)-(1.87))))
        overlap_pe2 = (1.+(self.beta_0*(((lambda_1+ha)+lambda_s_pe2)-(1.87))))
        Iion = ((((((((((((((((((INa+INaL)+Ito)+ICaL)+ICaNa)+ICaK)+IKr)+IKs)+IK1)+INaCa_i)+INaCa_ss)+INaK)+INab)+IKb)+IpCa)+ICab)+IClCa)+IClb)+Ikatp)
        h_lambda_2 = (torch.where((overlap_2>0.), overlap_2, 0.))
        h_lambda_me2 = (torch.where((overlap_me2>0.), overlap_me2, 0.))
        h_lambda_pe2 = (torch.where((overlap_pe2>0.), overlap_pe2, 0.))
        Ka_2 = ((h_lambda_2*(self.Tref/self.dr))*((A*self.XS)+(A*self.XW)))
        Ka_me2 = ((h_lambda_me2*(self.Tref/self.dr))*((A*self.XS)+(A*self.XW)))
        Ka_pe2 = ((h_lambda_pe2*(self.Tref/self.dr))*((A*self.XS)+(A*self.XW)))
        Ta_2 = ((h_lambda_2*(self.Tref/self.dr))*(((self.ZETAS+1.)*self.XS)+(self.ZETAW*self.XW)))
        Ta_me2 = ((h_lambda_me2*(self.Tref/self.dr))*(((self.ZETAS+1.)*self.XS)+(self.ZETAW*self.XW)))
        Ta_pe2 = ((h_lambda_pe2*(self.Tref/self.dr))*(((self.ZETAS+1.)*self.XS)+(self.ZETAW*self.XW)))
        derivative_2 = ((((((Tp_pe2+Ta_pe2)+(Ka_pe2*((lambda_1+ha)-(lambda_m))))-(Tp_me2))-(Ta_me2))-((Ka_me2*((lambda_1-(ha))-(lambda_m)))))/(2.*ha))
        func_2 = ((Tp_2+Ta_2)+(Ka_2*(lambda_1-(lambda_m))))
        lambda_2 = (lambda_1-((func_2/derivative_2)))
        C_3 = (lambda_2-(self.length))
        Cme3 = ((lambda_2-(ha))-(self.length))
        Cpe3 = ((lambda_2+ha)-(self.length))
        lambda_s_3 = (torch.where((lambda_2>0.87), 0.87, lambda_2))
        lambda_s_me3 = (torch.where(((lambda_2-(ha))>0.87), 0.87, (lambda_2-(ha))))
        lambda_s_pe3 = (torch.where(((lambda_2+ha)>0.87), 0.87, (lambda_2+ha)))
        Tp_3 = (self.a*(((torch.expm1((self.b*C_3))))+(self.par_k*(C_3-(self.CD)))))
        Tp_me3 = (self.a*(((torch.expm1((self.b*Cme3))))+(self.par_k*(Cme3-(self.CD)))))
        Tp_pe3 = (self.a*(((torch.expm1((self.b*Cpe3))))+(self.par_k*(Cpe3-(self.CD)))))
        overlap_3 = (1.+(self.beta_0*((lambda_2+lambda_s_3)-(1.87))))
        overlap_me3 = (1.+(self.beta_0*(((lambda_2-(ha))+lambda_s_me3)-(1.87))))
        overlap_pe3 = (1.+(self.beta_0*(((lambda_2+ha)+lambda_s_pe3)-(1.87))))
        h_lambda_3 = (torch.where((overlap_3>0.), overlap_3, 0.))
        h_lambda_me3 = (torch.where((overlap_me3>0.), overlap_me3, 0.))
        h_lambda_pe3 = (torch.where((overlap_pe3>0.), overlap_pe3, 0.))
        Ka_3 = ((h_lambda_3*(self.Tref/self.dr))*((A*self.XS)+(A*self.XW)))
        Ka_me3 = ((h_lambda_me3*(self.Tref/self.dr))*((A*self.XS)+(A*self.XW)))
        Ka_pe3 = ((h_lambda_pe3*(self.Tref/self.dr))*((A*self.XS)+(A*self.XW)))
        Ta_3 = ((h_lambda_3*(self.Tref/self.dr))*(((self.ZETAS+1.)*self.XS)+(self.ZETAW*self.XW)))
        Ta_me3 = ((h_lambda_me3*(self.Tref/self.dr))*(((self.ZETAS+1.)*self.XS)+(self.ZETAW*self.XW)))
        Ta_pe3 = ((h_lambda_pe3*(self.Tref/self.dr))*(((self.ZETAS+1.)*self.XS)+(self.ZETAW*self.XW)))
        derivative_3 = ((((((Tp_pe3+Ta_pe3)+(Ka_pe3*((lambda_2+ha)-(lambda_m))))-(Tp_me3))-(Ta_me3))-((Ka_me3*((lambda_2-(ha))-(lambda_m)))))/(2.*ha))
        func_3 = ((Tp_3+Ta_3)+(Ka_3*(lambda_2-(lambda_m))))
        lambda_3 = (lambda_2-((func_3/derivative_3)))
        dlambdadt = ((lambda_3-(lambda_m))/self.dt)
        lambda_s_4 = (torch.where((lambda_3>0.87), 0.87, lambda_3))
        delta_sl = dlambdadt
        overlap_4 = (1.+(self.beta_0*((lambda_3+lambda_s_4)-(1.87))))
        h_lambda_4 = (torch.where((overlap_4>0.), overlap_4, 0.))
        Ta_4 = ((h_lambda_4*(self.Tref/self.dr))*(((self.ZETAS+1.)*self.XS)+(self.ZETAW*self.XW)))
        Tension = Ta_4

        # Complete Forward Euler Update
        Bcai = (1./(1.+((cmdnmax*self.kmcmdn)/((self.kmcmdn+self.Cai_mM)*(self.kmcmdn+self.Cai_mM)))))
        Bcajsr = (1./(1.+((self.csqnmax*self.kmcsqn)/((self.kmcsqn+self.cajsr)*(self.kmcsqn+self.cajsr)))))
        Bcass = (1./((1.+((self.BSRmax*self.KmBSR)/((self.KmBSR+self.cass)*(self.KmBSR+self.cass))))+((self.BSLmax*self.KmBSL)/((self.KmBSL+self.cass)*(self.KmBSL+self.cass)))))
        Jdiff = ((self.cass-(self.Cai_mM))/self.tauCa)
        JdiffCl = ((self.clss-(self.cli))/self.tauCl)
        JdiffK = ((self.kss-(self.Ki))/self.tauK)
        JdiffNa = ((self.nass-(self.Nai))/self.tauNa)
        Jleak = ((0.0048825*self.cansr)/15.)
        Jtr = ((self.cansr-(self.cajsr))/self.tauTr)
        Jupnp = (((upScale*0.005425)*self.Cai_mM)/(self.Cai_mM+0.00092))
        Jupp = ((((upScale*2.75)*0.005425)*self.Cai_mM)/((self.Cai_mM+0.00092)-(0.00017)))
        XU = (((1.-(self.TmBlocked))-(self.XW))-(self.XS))
        alpha = (0.1161*(torch.exp((0.2990*vfrt))))
        alpha_2 = (0.0578*(torch.exp((0.9710*vfrt))))
        alpha_C2ToI = (0.52e-4*(torch.exp((1.525*vfrt))))
        alpha_i = (0.2533*(torch.exp((0.5953*vfrt))))
        beta = (0.2442*(torch.exp((-1.604*vfrt))))
        beta_2 = (0.349e-3*(torch.exp((-1.062*vfrt))))
        beta_i = (0.06525*(torch.exp((-0.8209*vfrt))))
        ca50_ = (self.ca50+(self.beta_1*(lambda_m-(1.))))
        diff_CD = dCd_dt
        diff_CaMKt = (((self.aCaMK*CaMKb)*(CaMKb+self.CaMKt))-((self.bCaMK*self.CaMKt)))
        diff_ZETAS = ((A*dlambdadt)-((cds*self.ZETAS)))
        diff_ZETAW = ((A*dlambdadt)-((cdw*self.ZETAW)))
        diff_stretch = dlambdadt
        fJrelp = (1./(1.+(self.KmCaMK/CaMKa)))
        fJupp = (1./(1.+(self.KmCaMK/CaMKa)))
        gr_w_ = (torch.where((self.ZETAW<0.), (-self.ZETAW), self.ZETAW))
        km2n = self.jca
        trpn_np_ = (torch.pow(self.TRPN,((-self.nperm)/2.)))
        xb_su = (k_su*self.XS)
        xb_ws = (k_ws*self.XW)
        xb_wu = (k_wu*self.XW)
        zs_neg = (torch.where((self.ZETAS<-1.), ((-self.ZETAS)-(1.)), 0.))
        zs_pos = (torch.where((self.ZETAS>0.), self.ZETAS, 0.))
        Jrel = (self.Jrel_b*(((1.-(fJrelp))*self.Jrel_np)+(fJrelp*self.Jrel_p)))
        Jup = (self.Jup_b*((((1.-(fJupp))*Jupnp)+(fJupp*Jupp))-(Jleak)))
        anca_i = (1./((self.k2n/km2n)+((((1.+(self.Kmn/self.Cai_mM))*(1.+(self.Kmn/self.Cai_mM)))*(1.+(self.Kmn/self.Cai_mM)))*(1.+(self.Kmn/self.Cai_mM)))))
        anca_ss = (1./((self.k2n/km2n)+((((1.+(self.Kmn/self.cass))*(1.+(self.Kmn/self.cass)))*(1.+(self.Kmn/self.cass)))*(1.+(self.Kmn/self.cass)))))
        beta_ItoC2 = (((beta_2*beta_i)*alpha_C2ToI)/(alpha_2*alpha_i))
        diff_C2 = (((alpha*self.C3)+(self.beta_1_*self.C1))-(((beta+self.alpha_1)*self.C2)))
        diff_C3 = ((beta*self.C2)-((alpha*self.C3)))
        diff_Ki = ((((-(((((((Ito+IKr)+IKs)+IK1)+IKb)+Ikatp)-((2.*INaK)))+ICaK_i))*Acap)/(self.F*vmyo))+((JdiffK*vss)/vmyo))
        diff_Nai = ((((-(((((INa+INaL)+(3.*INaCa_i))+ICaNa_i)+(3.*INaK))+INab))*Acap)/(self.F*vmyo))+((JdiffNa*vss)/vmyo))
        diff_O = (((alpha_2*self.C1)+(beta_i*self.I))-(((beta_2+alpha_i)*self.O)))
        diff_TRPN = (self.koff*(((torch.pow((self.Cai/ca50_),self.TRPN_n))*(1.-(self.TRPN)))-(self.TRPN)))
        diff_cli = ((((IClb+IClCa_sl)*Acap)/(self.F*vmyo))+((JdiffCl*vss)/vmyo))
        diff_clss = (((IClCa_junc*Acap)/(self.F*vss))-(JdiffCl))
        diff_kss = ((((-ICaK_ss)*Acap)/(self.F*vss))-(JdiffK))
        diff_nass = ((((-(ICaNa_ss+(3.*INaCa_ss)))*Acap)/(self.F*vss))-(JdiffNa))
        gamma_rate_w = (self.gamma_wu*gr_w_)
        trpn_np = (torch.where((trpn_np_>100.), 100., trpn_np_))
        xb_uw = (k_uw*XU)
        zs_ = (torch.where((zs_pos>zs_neg), zs_pos, zs_neg))
        diff_C1 = ((((self.alpha_1*self.C2)+(beta_2*self.O))+(beta_ItoC2*self.I))-((((self.beta_1_+alpha_2)+alpha_C2ToI)*self.C1)))
        diff_Cai_mM = (Bcai*((((((-(((ICaL_i+IpCa)+ICab)-((2.*INaCa_i))))*Acap)/((2.*self.F)*vmyo))-(((Jup*vnsr)/vmyo)))+((Jdiff*vss)/vmyo))-((self.trpnmax*diff_TRPN))))
        diff_I = (((alpha_C2ToI*self.C1)+(alpha_i*self.O))-(((beta_ItoC2+beta_i)*self.I)))
        diff_TmBlocked = (((ktm_block*trpn_np)*XU)-(((self.ktm_unblock*(torch.pow(self.TRPN,(self.nperm/2.))))*self.TmBlocked)))
        diff_cajsr = (Bcajsr*(Jtr-(Jrel)))
        diff_cansr = (Jup-(((Jtr*vjsr)/vnsr)))
        diff_cass = (Bcass*(((((-(ICaL_ss-((2.*INaCa_ss))))*Acap)/((2.*self.F)*vss))+((Jrel*vjsr)/vss))-(Jdiff)))
        diff_nca_i = ((anca_i*self.k2n)-((self.nca_i*km2n)))
        diff_nca_ss = ((anca_ss*self.k2n)-((self.nca_ss*km2n)))
        gamma_rate = (self.gamma*zs_)
        xb_wu_gamma = (gamma_rate_w*self.XW)
        dXW = (((xb_uw-(xb_wu))-(xb_ws))-(xb_wu_gamma))
        xb_su_gamma = (gamma_rate*self.XS)
        dXS = ((xb_ws-(xb_su))-(xb_su_gamma))
        diff_XW = dXW
        diff_XS = dXS
        C1_new = self.C1+diff_C1*self.dt
        C2_new = self.C2+diff_C2*self.dt
        C3_new = self.C3+diff_C3*self.dt
        CD_new = self.CD+diff_CD*self.dt
        CaMKt_new = self.CaMKt+diff_CaMKt*self.dt
        Cai_mM_new = self.Cai_mM+diff_Cai_mM*self.dt
        I_new = self.I+diff_I*self.dt
        Ki_new = self.Ki+diff_Ki*self.dt
        Nai_new = self.Nai+diff_Nai*self.dt
        O_new = self.O+diff_O*self.dt
        TRPN_new = self.TRPN+diff_TRPN*self.dt
        TmBlocked_new = self.TmBlocked+diff_TmBlocked*self.dt
        XS_new = self.XS+diff_XS*self.dt
        XW_new = self.XW+diff_XW*self.dt
        ZETAS_new = self.ZETAS+diff_ZETAS*self.dt
        ZETAW_new = self.ZETAW+diff_ZETAW*self.dt
        cajsr_new = self.cajsr+diff_cajsr*self.dt
        cansr_new = self.cansr+diff_cansr*self.dt
        cass_new = self.cass+diff_cass*self.dt
        cli_new = self.cli+diff_cli*self.dt
        clss_new = self.clss+diff_clss*self.dt
        kss_new = self.kss+diff_kss*self.dt
        nass_new = self.nass+diff_nass*self.dt
        nca_i_new = self.nca_i+diff_nca_i*self.dt
        nca_ss_new = self.nca_ss+diff_nca_ss*self.dt
        stretch_new = self.stretch+diff_stretch*self.dt

        # Complete Rush Larsen Update
        Jrel_inf_b = (((-a_rel)*ICaL_ss)/(1.+((((((((self.cajsr_half/self.cajsr)*(self.cajsr_half/self.cajsr))*(self.cajsr_half/self.cajsr))*(self.cajsr_half/self.cajsr))*(self.cajsr_half/self.cajsr))*(self.cajsr_half/self.cajsr))*(self.cajsr_half/self.cajsr))*(self.cajsr_half/self.cajsr))))
        Jrel_infp_b = (((-a_relp)*ICaL_ss)/(1.+((((((((self.cajsr_half/self.cajsr)*(self.cajsr_half/self.cajsr))*(self.cajsr_half/self.cajsr))*(self.cajsr_half/self.cajsr))*(self.cajsr_half/self.cajsr))*(self.cajsr_half/self.cajsr))*(self.cajsr_half/self.cajsr))*(self.cajsr_half/self.cajsr))))
        aa_inf = (1./(1.+(torch.exp(((-((V+self.EKshift)-(14.34)))/14.82)))))
        ah = (torch.where((V>=-40.), 0., (0.057*(torch.exp(((-(V+80.))/6.8))))))
        aj = (torch.where((V>=-40.), 0., ((((-2.5428e4*(torch.exp((0.2444*V))))-((6.948e-6*(torch.exp((-0.04391*V))))))*(V+37.78))/(1.+(torch.exp((0.311*(V+79.23))))))))
        ap_inf = (1./(1.+(torch.exp(((-((V+self.EKshift)-(24.34)))/14.82)))))
        bh = (torch.where((V>=-40.), (0.77/(0.13*(1.+(torch.exp(((-(V+10.66))/11.1)))))), ((2.7*(torch.exp((0.079*V))))+(3.1e5*(torch.exp((0.3485*V)))))))
        bj = (torch.where((V>=-40.), ((0.6*(torch.exp((0.057*V))))/(1.+(torch.exp((-0.1*(V+32.)))))), ((0.02424*(torch.exp((-0.01052*V))))/(1.+(torch.exp((-0.1378*(V+40.14))))))))
        d_inf = (torch.where((V>=31.4978), 1., (1.0763*(torch.exp((-1.0070*(torch.exp((-0.0829*V)))))))))
        dti_develop = (1.354+(1.e-4/((torch.exp((((V+self.EKshift)-(167.4))/15.89)))+(torch.exp(((-((V+self.EKshift)-(12.23)))/0.2154))))))
        dti_recover = (1.-((0.5/(1.+(torch.exp((((V+self.EKshift)+70.0)/20.0)))))))
        f_inf = (1./(1.+(torch.exp(((V+19.58)/3.696)))))
        hL_inf = (1./(1.+(torch.exp(((V+87.61)/7.488)))))
        hLp_inf = (1./(1.+(torch.exp(((V+93.81)/7.488)))))
        h_inf = (1./((1.+(torch.exp(((V+71.55)/7.43))))*(1.+(torch.exp(((V+71.55)/7.43))))))
        hp_inf = (1./((1.+(torch.exp(((V+77.55)/7.43))))*(1.+(torch.exp(((V+77.55)/7.43))))))
        i_inf = (1./(1.+(torch.exp((((V+self.EKshift)+43.94)/5.711)))))
        jca_inf = (1.0/(1.0+(torch.exp(((V+18.08)/2.7916)))))
        mL_inf = (1./(1.+(torch.exp(((-(V+42.85))/5.264)))))
        m_inf = (1./((1.+(torch.exp(((-(V+56.86))/9.03))))*(1.+(torch.exp(((-(V+56.86))/9.03))))))
        tau_aa = (1.0515/((1./(1.2089*(1.+(torch.exp(((-((V+self.EKshift)-(18.4099)))/29.3814))))))+(3.5/(1.+(torch.exp((((V+self.EKshift)+100.)/29.3814)))))))
        tau_d = ((self.offset+0.6)+(1./((torch.exp((-0.05*((V+self.vShift)+6.))))+(torch.exp((0.09*((V+self.vShift)+14.)))))))
        tau_fcaf = (7.+(1./((0.04*(torch.exp(((-(V-(4.)))/7.))))+(0.04*(torch.exp(((V-(4.))/7.)))))))
        tau_fcas = (100.+(1./((0.00012*(torch.exp(((-V)/3.))))+(0.00012*(torch.exp((V/7.)))))))
        tau_ff = (7.+(1./((0.0045*(torch.exp(((-(V+20.))/10.))))+(0.0045*(torch.exp(((V+20.)/10.)))))))
        tau_fs = (1000.+(1./((0.000035*(torch.exp(((-(V+5.))/4.))))+(0.000035*(torch.exp(((V+5.)/6.)))))))
        tau_iF_b = (4.562+(1./((0.3933*(torch.exp(((-((V+self.EKshift)+100.))/100.))))+(0.08004*(torch.exp((((V+self.EKshift)+50.)/16.59)))))))
        tau_iS_b = (23.62+(1./((0.001416*(torch.exp(((-((V+self.EKshift)+96.52))/59.05))))+(1.78e-8*(torch.exp((((V+self.EKshift)+114.1)/8.079)))))))
        tau_m = ((0.1292*(torch.exp((-(((V+45.79)/15.54)*((V+45.79)/15.54))))))+(0.06487*(torch.exp((-(((V-(4.823))/51.12)*((V-(4.823))/51.12)))))))
        tau_mL = ((0.1292*(torch.exp((-(((V+45.79)/15.54)*((V+45.79)/15.54))))))+(0.06487*(torch.exp((-(((V-(4.823))/51.12)*((V-(4.823))/51.12)))))))
        tau_rel_b = (self.bt/(1.+(0.0123/self.cajsr)))
        tau_relp_b = (btp/(1.+(0.0123/self.cajsr)))
        tau_xs1 = (817.3+(1./((2.326e-4*(torch.exp(((V+48.28)/17.8))))+(0.001292*(torch.exp(((-(V+210.))/230.)))))))
        tau_xs2 = (1./((0.01*(torch.exp(((V-(50.))/20.))))+(0.0193*(torch.exp(((-(V+66.54))/31.))))))
        xs1_inf = (1./(1.+(torch.exp(((-(V+11.6))/8.932)))))
        Jrel_np_inf = ((Jrel_inf_b*1.7) if (self.celltype==2.) else Jrel_inf_b)
        Jrel_p_inf = ((Jrel_infp_b*1.7) if (self.celltype==2.) else Jrel_infp_b)
        aa_rush_larsen_B = (torch.exp(((-self.dt)/tau_aa)))
        aa_rush_larsen_C = (torch.expm1(((-self.dt)/tau_aa)))
        d_rush_larsen_B = (torch.exp(((-self.dt)/tau_d)))
        d_rush_larsen_C = (torch.expm1(((-self.dt)/tau_d)))
        fca_inf = f_inf
        fcaf_rush_larsen_B = (torch.exp(((-self.dt)/tau_fcaf)))
        fcaf_rush_larsen_C = (torch.expm1(((-self.dt)/tau_fcaf)))
        fcas_rush_larsen_B = (torch.exp(((-self.dt)/tau_fcas)))
        fcas_rush_larsen_C = (torch.expm1(((-self.dt)/tau_fcas)))
        ff_inf = f_inf
        ff_rush_larsen_B = (torch.exp(((-self.dt)/tau_ff)))
        ff_rush_larsen_C = (torch.expm1(((-self.dt)/tau_ff)))
        ffp_inf = f_inf
        fs_inf = f_inf
        fs_rush_larsen_B = (torch.exp(((-self.dt)/tau_fs)))
        fs_rush_larsen_C = (torch.expm1(((-self.dt)/tau_fs)))
        hL_rush_larsen_A = ((-hL_inf)*hL_rush_larsen_C)
        hLp_rush_larsen_A = ((-hLp_inf)*hLp_rush_larsen_C)
        iF_inf = i_inf
        iFp_inf = i_inf
        iS_inf = i_inf
        iSp_inf = i_inf
        j_inf = h_inf
        jca_rush_larsen_A = ((-jca_inf)*jca_rush_larsen_C)
        mL_rush_larsen_B = (torch.exp(((-self.dt)/tau_mL)))
        mL_rush_larsen_C = (torch.expm1(((-self.dt)/tau_mL)))
        m_rush_larsen_B = (torch.exp(((-self.dt)/tau_m)))
        m_rush_larsen_C = (torch.expm1(((-self.dt)/tau_m)))
        tau_Jrel_np = (torch.where((tau_rel_b<0.001), 0.001, tau_rel_b))
        tau_Jrel_p = (torch.where((tau_relp_b<0.001), 0.001, tau_relp_b))
        tau_ap = tau_aa
        tau_fcafp = (2.5*tau_fcaf)
        tau_ffp = (2.5*tau_ff)
        tau_h = (1./(ah+bh))
        tau_iF = (tau_iF_b*delta_epi)
        tau_iS = (tau_iS_b*delta_epi)
        tau_j = (1./(aj+bj))
        xs1_rush_larsen_B = (torch.exp(((-self.dt)/tau_xs1)))
        xs1_rush_larsen_C = (torch.expm1(((-self.dt)/tau_xs1)))
        xs2_inf = xs1_inf
        xs2_rush_larsen_B = (torch.exp(((-self.dt)/tau_xs2)))
        xs2_rush_larsen_C = (torch.expm1(((-self.dt)/tau_xs2)))
        Jrel_np_rush_larsen_B = (torch.exp(((-self.dt)/tau_Jrel_np)))
        Jrel_np_rush_larsen_C = (torch.expm1(((-self.dt)/tau_Jrel_np)))
        Jrel_p_rush_larsen_B = (torch.exp(((-self.dt)/tau_Jrel_p)))
        Jrel_p_rush_larsen_C = (torch.expm1(((-self.dt)/tau_Jrel_p)))
        aa_rush_larsen_A = ((-aa_inf)*aa_rush_larsen_C)
        ap_rush_larsen_B = (torch.exp(((-self.dt)/tau_ap)))
        ap_rush_larsen_C = (torch.expm1(((-self.dt)/tau_ap)))
        d_rush_larsen_A = ((-d_inf)*d_rush_larsen_C)
        fcaf_inf = fca_inf
        fcafp_inf = fca_inf
        fcafp_rush_larsen_B = (torch.exp(((-self.dt)/tau_fcafp)))
        fcafp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_fcafp)))
        fcas_inf = fca_inf
        ff_rush_larsen_A = ((-ff_inf)*ff_rush_larsen_C)
        ffp_rush_larsen_B = (torch.exp(((-self.dt)/tau_ffp)))
        ffp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_ffp)))
        fs_rush_larsen_A = ((-fs_inf)*fs_rush_larsen_C)
        h_rush_larsen_B = (torch.exp(((-self.dt)/tau_h)))
        h_rush_larsen_C = (torch.expm1(((-self.dt)/tau_h)))
        iF_rush_larsen_B = (torch.exp(((-self.dt)/tau_iF)))
        iF_rush_larsen_C = (torch.expm1(((-self.dt)/tau_iF)))
        iS_rush_larsen_B = (torch.exp(((-self.dt)/tau_iS)))
        iS_rush_larsen_C = (torch.expm1(((-self.dt)/tau_iS)))
        j_rush_larsen_B = (torch.exp(((-self.dt)/tau_j)))
        j_rush_larsen_C = (torch.expm1(((-self.dt)/tau_j)))
        jp_inf = j_inf
        mL_rush_larsen_A = ((-mL_inf)*mL_rush_larsen_C)
        m_rush_larsen_A = ((-m_inf)*m_rush_larsen_C)
        tau_hp = tau_h
        tau_iFp = ((dti_develop*dti_recover)*tau_iF)
        tau_iSp = ((dti_develop*dti_recover)*tau_iS)
        tau_jp = (1.46*tau_j)
        xs1_rush_larsen_A = ((-xs1_inf)*xs1_rush_larsen_C)
        xs2_rush_larsen_A = ((-xs2_inf)*xs2_rush_larsen_C)
        Jrel_np_rush_larsen_A = ((-Jrel_np_inf)*Jrel_np_rush_larsen_C)
        Jrel_p_rush_larsen_A = ((-Jrel_p_inf)*Jrel_p_rush_larsen_C)
        ap_rush_larsen_A = ((-ap_inf)*ap_rush_larsen_C)
        fcaf_rush_larsen_A = ((-fcaf_inf)*fcaf_rush_larsen_C)
        fcafp_rush_larsen_A = ((-fcafp_inf)*fcafp_rush_larsen_C)
        fcas_rush_larsen_A = ((-fcas_inf)*fcas_rush_larsen_C)
        ffp_rush_larsen_A = ((-ffp_inf)*ffp_rush_larsen_C)
        h_rush_larsen_A = ((-h_inf)*h_rush_larsen_C)
        hp_rush_larsen_B = (torch.exp(((-self.dt)/tau_hp)))
        hp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_hp)))
        iF_rush_larsen_A = ((-iF_inf)*iF_rush_larsen_C)
        iFp_rush_larsen_B = (torch.exp(((-self.dt)/tau_iFp)))
        iFp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_iFp)))
        iS_rush_larsen_A = ((-iS_inf)*iS_rush_larsen_C)
        iSp_rush_larsen_B = (torch.exp(((-self.dt)/tau_iSp)))
        iSp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_iSp)))
        j_rush_larsen_A = ((-j_inf)*j_rush_larsen_C)
        jp_rush_larsen_B = (torch.exp(((-self.dt)/tau_jp)))
        jp_rush_larsen_C = (torch.expm1(((-self.dt)/tau_jp)))
        hp_rush_larsen_A = ((-hp_inf)*hp_rush_larsen_C)
        iFp_rush_larsen_A = ((-iFp_inf)*iFp_rush_larsen_C)
        iSp_rush_larsen_A = ((-iSp_inf)*iSp_rush_larsen_C)
        jp_rush_larsen_A = ((-jp_inf)*jp_rush_larsen_C)
        Jrel_np_new = Jrel_np_rush_larsen_A+Jrel_np_rush_larsen_B*self.Jrel_np
        Jrel_p_new = Jrel_p_rush_larsen_A+Jrel_p_rush_larsen_B*self.Jrel_p
        aa_new = aa_rush_larsen_A+aa_rush_larsen_B*self.aa
        ap_new = ap_rush_larsen_A+ap_rush_larsen_B*self.ap
        d_new = d_rush_larsen_A+d_rush_larsen_B*self.d
        fcaf_new = fcaf_rush_larsen_A+fcaf_rush_larsen_B*self.fcaf
        fcafp_new = fcafp_rush_larsen_A+fcafp_rush_larsen_B*self.fcafp
        fcas_new = fcas_rush_larsen_A+fcas_rush_larsen_B*self.fcas
        ff_new = ff_rush_larsen_A+ff_rush_larsen_B*self.ff
        ffp_new = ffp_rush_larsen_A+ffp_rush_larsen_B*self.ffp
        fs_new = fs_rush_larsen_A+fs_rush_larsen_B*self.fs
        h_new = h_rush_larsen_A+h_rush_larsen_B*self.h
        hL_new = hL_rush_larsen_A+hL_rush_larsen_B*self.hL
        hLp_new = hLp_rush_larsen_A+hLp_rush_larsen_B*self.hLp
        hp_new = hp_rush_larsen_A+hp_rush_larsen_B*self.hp
        iF_new = iF_rush_larsen_A+iF_rush_larsen_B*self.iF
        iFp_new = iFp_rush_larsen_A+iFp_rush_larsen_B*self.iFp
        iS_new = iS_rush_larsen_A+iS_rush_larsen_B*self.iS
        iSp_new = iSp_rush_larsen_A+iSp_rush_larsen_B*self.iSp
        j_new = j_rush_larsen_A+j_rush_larsen_B*self.j
        jca_new = jca_rush_larsen_A+jca_rush_larsen_B*self.jca
        jp_new = jp_rush_larsen_A+jp_rush_larsen_B*self.jp
        m_new = m_rush_larsen_A+m_rush_larsen_B*self.m
        mL_new = mL_rush_larsen_A+mL_rush_larsen_B*self.mL
        xs1_new = xs1_rush_larsen_A+xs1_rush_larsen_B*self.xs1
        xs2_new = xs2_rush_larsen_A+xs2_rush_larsen_B*self.xs2

        # Finish the update
        self.C1 = C1_new
        self.C2 = C2_new
        self.C3 = C3_new
        self.CD = CD_new
        self.CaMKt = CaMKt_new
        self.Cai = Cai_new
        self.Cai_mM = Cai_mM_new
        self.I = I_new
        self.Jrel_np = Jrel_np_new
        self.Jrel_p = Jrel_p_new
        self.Ki = Ki_new
        self.Nai = Nai_new
        self.O = O_new
        self.TRPN = TRPN_new
        self.TmBlocked = TmBlocked_new
        self.XS = XS_new
        self.XW = XW_new
        self.ZETAS = ZETAS_new
        self.ZETAW = ZETAW_new
        self.aa = aa_new
        self.ap = ap_new
        self.cajsr = cajsr_new
        self.cansr = cansr_new
        self.cass = cass_new
        self.cli = cli_new
        self.clss = clss_new
        self.d = d_new
        self.fcaf = fcaf_new
        self.fcafp = fcafp_new
        self.fcas = fcas_new
        self.ff = ff_new
        self.ffp = ffp_new
        self.fs = fs_new
        self.h = h_new
        self.hL = hL_new
        self.hLp = hLp_new
        self.hp = hp_new
        self.iF = iF_new
        self.iFp = iFp_new
        self.iS = iS_new
        self.iSp = iSp_new
        self.j = j_new
        self.jca = jca_new
        self.jp = jp_new
        self.kss = kss_new
        self.m = m_new
        self.mL = mL_new
        self.nass = nass_new
        self.nca_i = nca_i_new
        self.nca_ss = nca_ss_new
        self.stretch = stretch_new
        self.xs1 = xs1_new
        self.xs2 = xs2_new
        self.Tension = Tension
        self.delta_sl = delta_sl

        return -Iion


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    dt = 0.01
    dt_imp = float(np.float32(dt))   # limpet keeps the IMP time step in a float
    stimulus = 20
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    ionic = Tomek20Land17_DynSarc(cell_type="ENDO", 
                                  dt=dt_imp, 
                                  device=device, 
                                  dtype=torch.float64)
    V = ionic.initialize(n_nodes=1)

    V_list = []
    Cai_list = []
    Tension_list = []

    ctime = 0.0
    for _ in range(int(1000/dt)):
        V_list.append([ctime, V.item()])
        Cai_list.append([ctime, ionic.Cai.item()])
        Tension_list.append([ctime, ionic.Tension.item()])

        if ctime >= 0 and ctime < (0+2.0): 
            V = V + dt * stimulus
        dV = ionic.differentiate(V)
        V = V + dt * dV
        ctime += dt

    plt.figure()
    V_list = np.array(V_list)    
    plt.plot(V_list[:, 0], V_list[:, 1])
    plt.savefig("V_Tomek20Land17_DynSarc.png")

    plt.figure()
    Cai_list = np.array(Cai_list)    
    plt.plot(Cai_list[:, 0], Cai_list[:, 1])
    plt.savefig("Cai.png")

    plt.figure()
    Tension_list = np.array(Tension_list)    
    plt.plot(Tension_list[:, 0], Tension_list[:, 1])
    plt.savefig("Tension.png")
