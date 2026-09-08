import torch
import torchcor as tc
import math
from math import exp, expm1, log, sqrt
from typing import Optional, List


@torch.jit.script
class TWorld:
    def __init__(self, 
                 dt: float, 
                 region_ids: Optional[List[int]] = None, 
                 cell_type: str = "ENDO", 
                 sex: str = "MIX", 
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float64):
        
        self.name = "TWorld"
        self.dt = dt  # the IKr / RyR / Csqn states are stepped with RK4: keep dt below ~0.011 ms
        self.region_ids = region_ids
        self.node_indices = torch.tensor([0])
        self.device = device
        self.dtype = dtype

        self.cell_type = "ENDO" if cell_type is None else cell_type
        self.sex_type = "MIX" if sex is None else sex

        # Constants
        self.C0_init = 0.997558791218494
        self.C1_init = 0.000960122333286174
        self.C2_init = 0.000804204051120587
        self.Cai_init = 9.66105402693763e-5
        self.Cli_init = 15.5745509675962
        self.I_init = 0.0000355901681542091
        self.Ki_init = 134.213097948672
        self.Nai_init = 8.88365804124162
        self.O_init = 0.00064129227809794
        self.V_init = -87.0655895407835
        self.buffers_CaM_init = 0.000326761159683871
        self.buffers_Csqn_init = 1.32358809148372
        self.buffers_Myosin_ca_init = 0.00435725142843641
        self.buffers_Myosin_mg_init = 0.134587917639655
        self.buffers_NaBj_init = 3.5588839934477
        self.buffers_NaBsl_init = 0.776467094025442
        self.buffers_SLHj_init = 0.0497605868295565
        self.buffers_SLHsl_init = 0.0960185900338078
        self.buffers_SLLj_init = 0.00415486833433121
        self.buffers_SLLsl_init = 0.00810233315864154
        self.buffers_SRB_init = 0.00247759889601234
        self.buffers_TnCHc_init = 0.124415836833344
        self.buffers_TnCHm_init = 0.00475139089328611
        self.buffers_TnClow_init = 0.0104491410305814
        self.caj_init = 9.20580586542242e-5
        self.camk_f_ICaL_init = 0.149310462149601
        self.camk_f_PLB_init = 0.0102519516890396
        self.camk_f_RyR_init = 0.0583425856770706
        self.camk_trap_init = 0.0414775695414329
        self.casig_serca_trap_init = 0.0414775695414004
        self.casl_init = 9.62428218317655e-5
        self.casr_init = 0.709622552333155
        self.contraction_Ca_TRPN_init = 0.0319778784167468
        self.contraction_TmBlocked_init = 0.992378492940848
        self.contraction_XS_init = 0.00191507884532543
        self.contraction_XW_init = 0.00285370717430549
        self.contraction_ZETAS_init = 0.
        self.contraction_ZETAW_init = 0.
        self.d_P_init = 0.0
        self.d_init = 6.00992839633468e-26
        self.fBPf_init = 0.999999987992091
        self.fcaBPf_init = 0.999999987985882
        self.fcaf_P_init = 0.999999987985882
        self.fcaf_init = 0.999999988251085
        self.fcafp_init = 0.999999987980701
        self.fcas_P_init = 0.999145398038995
        self.fcas_init = 0.991477881027689
        self.ff_P_init = 0.999999987992091
        self.ff_init = 0.999999988243592
        self.ffp_init = 0.999999987868131
        self.fs_P_init = 0.930411822666439
        self.fs_init = 0.911094950010905
        self.hL_init = 0.478572465113524
        self.hLp_init = 0.258793079229038
        self.h_P_init = 0.647375310489836
        self.h_init = 0.791660456106286
        self.hp_P_init = 0.419158183843564
        self.hp_init = 0.61240980788246
        self.ical_pureCDI_junc_init = 0.99834227586449
        self.ical_pureCDI_sl_init = 0.99826318591148
        self.j_P_init = 0.647279109941897
        self.j_init = 0.791587530878068
        self.jca_init = 0.999993204834445
        self.jp_P_init = 0.418669666153646
        self.jp_init = 0.791052149435449
        self.jrel_icaldep_act_init = -9.23949112019061e-15
        self.jrel_icaldep_f1_init = 0.999994847409611
        self.jrel_icaldep_f2_init = 0.998725931920675
        self.mL_init = 0.000224902633151066
        self.m_P_init = 0.0033402868342183
        self.m_init = 0.00115997496446638
        self.naj_init = 8.88424477947199
        self.nasl_init = 8.88341217642103
        self.nca_i_init = 0.00621547539490541
        self.nca_init = 0.00528788208273223
        self.ryr_CaRI_init = 0.0653313479138348
        self.ryr_CaRI_p_init = 0.0651828099424919
        self.ryr_I_init = 4.57362306340243e-9
        self.ryr_I_p_init = 6.8611892649539e-9
        self.ryr_O_init = 3.33929041065479e-8
        self.ryr_O_p_init = 5.00978004918837e-8
        self.ryr_R_init = 0.821525956842405
        self.ryr_R_p_init = 0.8216639452872
        self.xs_junc_init = 0.0311875010014239
        self.xs_sl_init = 0.031204810574373
        self.xtof_init = 0.00028609937449475
        self.xtof_p_init = 0.000132590043205027
        self.xtos_init = 0.000286101402392894
        self.xtos_p_init = 0.000132590983159417
        self.ytof_init = 0.99999864687518
        self.ytof_p_init = 0.99999864687975
        self.ytos_init = 0.768184915918179
        self.ytos_p_init = 0.777117031645024

        # Parameters
        self.Aff = 0.52477
        self.Bmax_Naj = 7.561
        self.Bmax_Nasl = 1.65
        self.Bmax_SR = 0.01785854
        self.Bmax_TnChigh = 0.14
        self.Bmax_TnClow = 0.07
        self.Bmax_myosin = 0.14
        self.CI_to_RI = 2.48e-3
        self.CaMK0 = 0.1
        self.Cae = 1.8
        self.Cle = 150.0
        self.Cmem = 1.3810e-10
        self.F = 96485.
        self.Fjunc = 0.11
        self.GCab_b = 5.15575e-04
        self.GClCa_b = 0.01615
        self.GClb_b = 0.00241
        self.GK1_b = 0.6992
        self.GKb_b = 0.010879
        self.GKr_b = 0.043
        self.GKs_b = 2.97002
        self.GNaL_b = 0.04229
        self.GNa_b = 22.08788
        self.GNab_b = 5.94e-04
        self.Gncx_b = 1.79e-3
        self.ICaLPCa_multiplier = 1.
        self.ICaL_fractionSS = 0.8
        self.ICab_multiplier = 1.
        self.IClCa_multiplier = 1.
        self.IClb_multiplier = 1.
        self.IK1_multiplier = 1.
        self.IKb_multiplier = 1.
        self.IKr_multiplier = 1.
        self.IKs_multiplier = 1.
        self.INaCa_fractionSS = 0.31
        self.INaCa_multiplier = 1.
        self.INaK_multiplier = 1.
        self.INaL_multiplier = 1.
        self.INa_multiplier = 1.
        self.INab_multiplier = 1.
        self.IbarNaK_b = 2.10774
        self.IbarSLCaP_b = 0.02064
        self.IpCa_multiplier = 1.
        self.Itof_multiplier = 1.
        self.Itos_multiplier = 1.
        self.J_ca_juncsl = 3.260674633581688e-13
        self.J_ca_slmyo = 1.341273673480337e-11
        self.J_na_juncsl = 1.831278232206080e-14
        self.J_na_slmyo = 1.638627922219794e-12
        self.Jrel_multiplier = 1.
        self.Jup_multiplier = 1.
        self.K_Phos_CaMK = 0.35
        self.KdClCa = 0.1
        self.Ke = 5.0
        self.KmCaAct = 150e-6
        self.KmKo = 1.5
        self.KmNaip = 11.
        self.KmNaip_PKA = 8.4615
        self.KmPCa = 5e-4
        self.Km_CaMK_Ca = 0.0075
        self.Km_SERCA_Ca = 0.4
        self.Kmf = 3.0672e-04
        self.Kmn = 0.00222
        self.Kmr = 2.31442
        self.MaxSR = 15.
        self.Max_Vmax_SERCA_Ca = 1.11142
        self.MinSR = 1.
        self.Nae = 140.0
        self.PCa_b = 1.5768e-04
        self.PNaK = 0.01833
        self.Q10SLCaP = 2.35
        self.Q10SRCaP = 2.6
        self.R = 8314.
        self.T = 310.
        self.TOT_A = 25.
        self.TRPN_n = 1.65
        self.Tref = 80.
        self.Vmax_SRCaP_b = 5.43e-3
        self.Whole_cell_PP1 = 0.13698
        self.alpha = 0.05
        self.alpha_1 = 0.154375
        self.alpha_serca = 0.05
        self.baseRateCaI = 3.0232e-04
        self.beta = 6.8e-4
        self.beta_0 = 2.3
        self.beta_1 = 0.1911
        self.beta_1_contraction = -2.4
        self.bt = 12.4767
        self.caExpFactor = 0.68655
        self.caExpFactor2 = 2.06273
        self.caTransFactor = 0.94428
        self.caTransFactor2 = 0.52967
        self.cellLength = 100.
        self.cellRadius = 10.25
        self.celltype = 1. if cell_type == "EPI" else 2. if cell_type == "MID" else 0.
        self.dielConstant = 74.
        self.directRelMidpoint = 0.95271
        self.dr = 0.25
        self.ec50SR = 0.75385
        self.ecCaI = 1e-3
        self.fICaL_PKA = 0.
        self.fIKs_PKA = 0.
        self.fINaK_PKA = 0.
        self.fINa_PKA = 0.
        self.fMyBPC_PKA = 0.
        self.fPLB_PKA = 0.
        self.fTnI_PKA = 0.
        self.fracTnIpo = 0.0031
        self.gKs_factor = 0.01
        self.gamma = 0.0085
        self.gamma_wu = 0.615
        self.hillSRCaP = 1.02809
        self.k2n = 957.85903
        self.kasymm = 19.4258
        self.kcaoff = 3.8532e3
        self.kcaon = 3.4164e6
        self.kiCa = 0.39871
        self.kim = 0.04311
        self.kna1 = 11.9712
        self.kna2 = 2.76
        self.kna3 = 88.767
        self.koCa = 23.87221
        self.koff = 0.07854
        self.koff_cam = 0.238
        self.koff_csqn = 65.
        self.koff_myoca = 4.6e-4
        self.koff_myomg = 5.7e-5
        self.koff_na = 1e-3
        self.koff_slh = 0.03
        self.koff_sll = 1.3
        self.koff_sr = 0.06
        self.koff_tnchca = 3.2e-5
        self.koff_tnchmg = 3.33e-3
        self.kom = 0.16219
        self.kon_cam = 34.
        self.kon_csqn = 100.
        self.kon_myoca = 13.8
        self.kon_myomg = 0.0157
        self.kon_na = 1e-4
        self.kon_slh = 100.
        self.kon_sll = 100.
        self.kon_sr = 100.
        self.kon_tnchca = 2.37
        self.kon_tnchmg = 3e-3
        self.ktm_unblock = 0.02626
        self.lambda_ = 1.  # "lambda" is a Python keyword
        self.lambda_max = 1.2
        self.lambda_min = 0.87
        self.lambda_rate = 0.
        self.maxCaI = 30.13294
        self.mgi = 0.5
        self.minCaI = 0.93249
        self.mu = 3.94046
        self.nperm = 2.036
        self.nu = 10.15996
        self.perm50 = 0.35
        self.phi = 2.23
        self.qca = 0.0955
        self.qna = 0.6718
        self.rateRecovery = 0.02313
        self.sex = 1. if sex == "MALE" else 2. if sex == "FEMALE" else 0.
        self.steepnessCaI = 5.93447
        self.steepnessCaSR = 5.09473
        self.tauInact = 64.11202
        self.tauInact2 = 119.48978
        self.tau_plb = 100000.
        self.tau_ryr = 10000.
        self.thL = 145.
        self.tjca = 66.0
        self.wca = 5.1756e4
        self.wfrac = 0.5
        self.wna = 3.2978e4
        self.wnaca = 2.7763e3
        self.zca = 2.
        self.zcl = -1.
        self.zk = 1.
        self.zna = 1.

        # 0 lookup tables

        # 92 states variables
        self.C0 = torch.tensor([self.C0_init])
        self.C1 = torch.tensor([self.C1_init])
        self.C2 = torch.tensor([self.C2_init])
        self.Cai = torch.tensor([self.Cai_init])
        self.Cli = torch.tensor([self.Cli_init])
        self.I = torch.tensor([self.I_init])
        self.Ki = torch.tensor([self.Ki_init])
        self.Nai = torch.tensor([self.Nai_init])
        self.O = torch.tensor([self.O_init])
        self.buffers_CaM = torch.tensor([self.buffers_CaM_init])
        self.buffers_Csqn = torch.tensor([self.buffers_Csqn_init])
        self.buffers_Myosin_ca = torch.tensor([self.buffers_Myosin_ca_init])
        self.buffers_Myosin_mg = torch.tensor([self.buffers_Myosin_mg_init])
        self.buffers_NaBj = torch.tensor([self.buffers_NaBj_init])
        self.buffers_NaBsl = torch.tensor([self.buffers_NaBsl_init])
        self.buffers_SLHj = torch.tensor([self.buffers_SLHj_init])
        self.buffers_SLHsl = torch.tensor([self.buffers_SLHsl_init])
        self.buffers_SLLj = torch.tensor([self.buffers_SLLj_init])
        self.buffers_SLLsl = torch.tensor([self.buffers_SLLsl_init])
        self.buffers_SRB = torch.tensor([self.buffers_SRB_init])
        self.buffers_TnCHc = torch.tensor([self.buffers_TnCHc_init])
        self.buffers_TnCHm = torch.tensor([self.buffers_TnCHm_init])
        self.buffers_TnClow = torch.tensor([self.buffers_TnClow_init])
        self.caj = torch.tensor([self.caj_init])
        self.camk_f_ICaL = torch.tensor([self.camk_f_ICaL_init])
        self.camk_f_PLB = torch.tensor([self.camk_f_PLB_init])
        self.camk_f_RyR = torch.tensor([self.camk_f_RyR_init])
        self.camk_trap = torch.tensor([self.camk_trap_init])
        self.casig_serca_trap = torch.tensor([self.casig_serca_trap_init])
        self.casl = torch.tensor([self.casl_init])
        self.casr = torch.tensor([self.casr_init])
        self.contraction_Ca_TRPN = torch.tensor([self.contraction_Ca_TRPN_init])
        self.contraction_TmBlocked = torch.tensor([self.contraction_TmBlocked_init])
        self.contraction_XS = torch.tensor([self.contraction_XS_init])
        self.contraction_XW = torch.tensor([self.contraction_XW_init])
        self.contraction_ZETAS = torch.tensor([self.contraction_ZETAS_init])
        self.contraction_ZETAW = torch.tensor([self.contraction_ZETAW_init])
        self.d = torch.tensor([self.d_init])
        self.d_P = torch.tensor([self.d_P_init])
        self.fBPf = torch.tensor([self.fBPf_init])
        self.fcaBPf = torch.tensor([self.fcaBPf_init])
        self.fcaf = torch.tensor([self.fcaf_init])
        self.fcaf_P = torch.tensor([self.fcaf_P_init])
        self.fcafp = torch.tensor([self.fcafp_init])
        self.fcas = torch.tensor([self.fcas_init])
        self.fcas_P = torch.tensor([self.fcas_P_init])
        self.ff = torch.tensor([self.ff_init])
        self.ff_P = torch.tensor([self.ff_P_init])
        self.ffp = torch.tensor([self.ffp_init])
        self.fs = torch.tensor([self.fs_init])
        self.fs_P = torch.tensor([self.fs_P_init])
        self.h = torch.tensor([self.h_init])
        self.hL = torch.tensor([self.hL_init])
        self.hLp = torch.tensor([self.hLp_init])
        self.h_P = torch.tensor([self.h_P_init])
        self.hp = torch.tensor([self.hp_init])
        self.hp_P = torch.tensor([self.hp_P_init])
        self.ical_pureCDI_junc = torch.tensor([self.ical_pureCDI_junc_init])
        self.ical_pureCDI_sl = torch.tensor([self.ical_pureCDI_sl_init])
        self.j = torch.tensor([self.j_init])
        self.j_P = torch.tensor([self.j_P_init])
        self.jca = torch.tensor([self.jca_init])
        self.jp = torch.tensor([self.jp_init])
        self.jp_P = torch.tensor([self.jp_P_init])
        self.jrel_icaldep_act = torch.tensor([self.jrel_icaldep_act_init])
        self.jrel_icaldep_f1 = torch.tensor([self.jrel_icaldep_f1_init])
        self.jrel_icaldep_f2 = torch.tensor([self.jrel_icaldep_f2_init])
        self.m = torch.tensor([self.m_init])
        self.mL = torch.tensor([self.mL_init])
        self.m_P = torch.tensor([self.m_P_init])
        self.naj = torch.tensor([self.naj_init])
        self.nasl = torch.tensor([self.nasl_init])
        self.nca = torch.tensor([self.nca_init])
        self.nca_i = torch.tensor([self.nca_i_init])
        self.ryr_CaRI = torch.tensor([self.ryr_CaRI_init])
        self.ryr_CaRI_p = torch.tensor([self.ryr_CaRI_p_init])
        self.ryr_I = torch.tensor([self.ryr_I_init])
        self.ryr_I_p = torch.tensor([self.ryr_I_p_init])
        self.ryr_O = torch.tensor([self.ryr_O_init])
        self.ryr_O_p = torch.tensor([self.ryr_O_p_init])
        self.ryr_R = torch.tensor([self.ryr_R_init])
        self.ryr_R_p = torch.tensor([self.ryr_R_p_init])
        self.xs_junc = torch.tensor([self.xs_junc_init])
        self.xs_sl = torch.tensor([self.xs_sl_init])
        self.xtof = torch.tensor([self.xtof_init])
        self.xtof_p = torch.tensor([self.xtof_p_init])
        self.xtos = torch.tensor([self.xtos_init])
        self.xtos_p = torch.tensor([self.xtos_p_init])
        self.ytof = torch.tensor([self.ytof_init])
        self.ytof_p = torch.tensor([self.ytof_p_init])
        self.ytos = torch.tensor([self.ytos_init])
        self.ytos_p = torch.tensor([self.ytos_p_init])

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
        # TWorld declares no lookup tables, every rate is evaluated on the fly
        pass

    def initialize(self, n_nodes: int):
        self.construct_tables()
        
        V = torch.full((n_nodes,), self.V_init, device=self.device, dtype=self.dtype)

        self.C0 = torch.full((n_nodes,), self.C0_init, device=self.device, dtype=self.dtype)
        self.C1 = torch.full((n_nodes,), self.C1_init, device=self.device, dtype=self.dtype)
        self.C2 = torch.full((n_nodes,), self.C2_init, device=self.device, dtype=self.dtype)
        self.Cai = torch.full((n_nodes,), self.Cai_init, device=self.device, dtype=self.dtype)
        self.Cai = self.Cai * 1e3
        self.Cli = torch.full((n_nodes,), self.Cli_init, device=self.device, dtype=self.dtype)
        self.I = torch.full((n_nodes,), self.I_init, device=self.device, dtype=self.dtype)
        self.Ki = torch.full((n_nodes,), self.Ki_init, device=self.device, dtype=self.dtype)
        self.Nai = torch.full((n_nodes,), self.Nai_init, device=self.device, dtype=self.dtype)
        self.O = torch.full((n_nodes,), self.O_init, device=self.device, dtype=self.dtype)
        self.buffers_CaM = torch.full((n_nodes,), self.buffers_CaM_init, device=self.device, dtype=self.dtype)
        self.buffers_Csqn = torch.full((n_nodes,), self.buffers_Csqn_init, device=self.device, dtype=self.dtype)
        self.buffers_Myosin_ca = torch.full((n_nodes,), self.buffers_Myosin_ca_init, device=self.device, dtype=self.dtype)
        self.buffers_Myosin_mg = torch.full((n_nodes,), self.buffers_Myosin_mg_init, device=self.device, dtype=self.dtype)
        self.buffers_NaBj = torch.full((n_nodes,), self.buffers_NaBj_init, device=self.device, dtype=self.dtype)
        self.buffers_NaBsl = torch.full((n_nodes,), self.buffers_NaBsl_init, device=self.device, dtype=self.dtype)
        self.buffers_SLHj = torch.full((n_nodes,), self.buffers_SLHj_init, device=self.device, dtype=self.dtype)
        self.buffers_SLHsl = torch.full((n_nodes,), self.buffers_SLHsl_init, device=self.device, dtype=self.dtype)
        self.buffers_SLLj = torch.full((n_nodes,), self.buffers_SLLj_init, device=self.device, dtype=self.dtype)
        self.buffers_SLLsl = torch.full((n_nodes,), self.buffers_SLLsl_init, device=self.device, dtype=self.dtype)
        self.buffers_SRB = torch.full((n_nodes,), self.buffers_SRB_init, device=self.device, dtype=self.dtype)
        self.buffers_TnCHc = torch.full((n_nodes,), self.buffers_TnCHc_init, device=self.device, dtype=self.dtype)
        self.buffers_TnCHm = torch.full((n_nodes,), self.buffers_TnCHm_init, device=self.device, dtype=self.dtype)
        self.buffers_TnClow = torch.full((n_nodes,), self.buffers_TnClow_init, device=self.device, dtype=self.dtype)
        self.caj = torch.full((n_nodes,), self.caj_init, device=self.device, dtype=self.dtype)
        self.camk_f_ICaL = torch.full((n_nodes,), self.camk_f_ICaL_init, device=self.device, dtype=self.dtype)
        self.camk_f_PLB = torch.full((n_nodes,), self.camk_f_PLB_init, device=self.device, dtype=self.dtype)
        self.camk_f_RyR = torch.full((n_nodes,), self.camk_f_RyR_init, device=self.device, dtype=self.dtype)
        self.camk_trap = torch.full((n_nodes,), self.camk_trap_init, device=self.device, dtype=self.dtype)
        self.casig_serca_trap = torch.full((n_nodes,), self.casig_serca_trap_init, device=self.device, dtype=self.dtype)
        self.casl = torch.full((n_nodes,), self.casl_init, device=self.device, dtype=self.dtype)
        self.casr = torch.full((n_nodes,), self.casr_init, device=self.device, dtype=self.dtype)
        self.contraction_Ca_TRPN = torch.full((n_nodes,), self.contraction_Ca_TRPN_init, device=self.device, dtype=self.dtype)
        self.contraction_TmBlocked = torch.full((n_nodes,), self.contraction_TmBlocked_init, device=self.device, dtype=self.dtype)
        self.contraction_XS = torch.full((n_nodes,), self.contraction_XS_init, device=self.device, dtype=self.dtype)
        self.contraction_XW = torch.full((n_nodes,), self.contraction_XW_init, device=self.device, dtype=self.dtype)
        self.contraction_ZETAS = torch.full((n_nodes,), self.contraction_ZETAS_init, device=self.device, dtype=self.dtype)
        self.contraction_ZETAW = torch.full((n_nodes,), self.contraction_ZETAW_init, device=self.device, dtype=self.dtype)
        self.d = torch.full((n_nodes,), self.d_init, device=self.device, dtype=self.dtype)
        self.d_P = torch.full((n_nodes,), self.d_P_init, device=self.device, dtype=self.dtype)
        self.fBPf = torch.full((n_nodes,), self.fBPf_init, device=self.device, dtype=self.dtype)
        self.fcaBPf = torch.full((n_nodes,), self.fcaBPf_init, device=self.device, dtype=self.dtype)
        self.fcaf = torch.full((n_nodes,), self.fcaf_init, device=self.device, dtype=self.dtype)
        self.fcaf_P = torch.full((n_nodes,), self.fcaf_P_init, device=self.device, dtype=self.dtype)
        self.fcafp = torch.full((n_nodes,), self.fcafp_init, device=self.device, dtype=self.dtype)
        self.fcas = torch.full((n_nodes,), self.fcas_init, device=self.device, dtype=self.dtype)
        self.fcas_P = torch.full((n_nodes,), self.fcas_P_init, device=self.device, dtype=self.dtype)
        self.ff = torch.full((n_nodes,), self.ff_init, device=self.device, dtype=self.dtype)
        self.ff_P = torch.full((n_nodes,), self.ff_P_init, device=self.device, dtype=self.dtype)
        self.ffp = torch.full((n_nodes,), self.ffp_init, device=self.device, dtype=self.dtype)
        self.fs = torch.full((n_nodes,), self.fs_init, device=self.device, dtype=self.dtype)
        self.fs_P = torch.full((n_nodes,), self.fs_P_init, device=self.device, dtype=self.dtype)
        self.h = torch.full((n_nodes,), self.h_init, device=self.device, dtype=self.dtype)
        self.hL = torch.full((n_nodes,), self.hL_init, device=self.device, dtype=self.dtype)
        self.hLp = torch.full((n_nodes,), self.hLp_init, device=self.device, dtype=self.dtype)
        self.h_P = torch.full((n_nodes,), self.h_P_init, device=self.device, dtype=self.dtype)
        self.hp = torch.full((n_nodes,), self.hp_init, device=self.device, dtype=self.dtype)
        self.hp_P = torch.full((n_nodes,), self.hp_P_init, device=self.device, dtype=self.dtype)
        self.ical_pureCDI_junc = torch.full((n_nodes,), self.ical_pureCDI_junc_init, device=self.device, dtype=self.dtype)
        self.ical_pureCDI_sl = torch.full((n_nodes,), self.ical_pureCDI_sl_init, device=self.device, dtype=self.dtype)
        self.j = torch.full((n_nodes,), self.j_init, device=self.device, dtype=self.dtype)
        self.j_P = torch.full((n_nodes,), self.j_P_init, device=self.device, dtype=self.dtype)
        self.jca = torch.full((n_nodes,), self.jca_init, device=self.device, dtype=self.dtype)
        self.jp = torch.full((n_nodes,), self.jp_init, device=self.device, dtype=self.dtype)
        self.jp_P = torch.full((n_nodes,), self.jp_P_init, device=self.device, dtype=self.dtype)
        self.jrel_icaldep_act = torch.full((n_nodes,), self.jrel_icaldep_act_init, device=self.device, dtype=self.dtype)
        self.jrel_icaldep_f1 = torch.full((n_nodes,), self.jrel_icaldep_f1_init, device=self.device, dtype=self.dtype)
        self.jrel_icaldep_f2 = torch.full((n_nodes,), self.jrel_icaldep_f2_init, device=self.device, dtype=self.dtype)
        self.m = torch.full((n_nodes,), self.m_init, device=self.device, dtype=self.dtype)
        self.mL = torch.full((n_nodes,), self.mL_init, device=self.device, dtype=self.dtype)
        self.m_P = torch.full((n_nodes,), self.m_P_init, device=self.device, dtype=self.dtype)
        self.naj = torch.full((n_nodes,), self.naj_init, device=self.device, dtype=self.dtype)
        self.nasl = torch.full((n_nodes,), self.nasl_init, device=self.device, dtype=self.dtype)
        self.nca = torch.full((n_nodes,), self.nca_init, device=self.device, dtype=self.dtype)
        self.nca_i = torch.full((n_nodes,), self.nca_i_init, device=self.device, dtype=self.dtype)
        self.ryr_CaRI = torch.full((n_nodes,), self.ryr_CaRI_init, device=self.device, dtype=self.dtype)
        self.ryr_CaRI_p = torch.full((n_nodes,), self.ryr_CaRI_p_init, device=self.device, dtype=self.dtype)
        self.ryr_I = torch.full((n_nodes,), self.ryr_I_init, device=self.device, dtype=self.dtype)
        self.ryr_I_p = torch.full((n_nodes,), self.ryr_I_p_init, device=self.device, dtype=self.dtype)
        self.ryr_O = torch.full((n_nodes,), self.ryr_O_init, device=self.device, dtype=self.dtype)
        self.ryr_O_p = torch.full((n_nodes,), self.ryr_O_p_init, device=self.device, dtype=self.dtype)
        self.ryr_R = torch.full((n_nodes,), self.ryr_R_init, device=self.device, dtype=self.dtype)
        self.ryr_R_p = torch.full((n_nodes,), self.ryr_R_p_init, device=self.device, dtype=self.dtype)
        self.xs_junc = torch.full((n_nodes,), self.xs_junc_init, device=self.device, dtype=self.dtype)
        self.xs_sl = torch.full((n_nodes,), self.xs_sl_init, device=self.device, dtype=self.dtype)
        self.xtof = torch.full((n_nodes,), self.xtof_init, device=self.device, dtype=self.dtype)
        self.xtof_p = torch.full((n_nodes,), self.xtof_p_init, device=self.device, dtype=self.dtype)
        self.xtos = torch.full((n_nodes,), self.xtos_init, device=self.device, dtype=self.dtype)
        self.xtos_p = torch.full((n_nodes,), self.xtos_p_init, device=self.device, dtype=self.dtype)
        self.ytof = torch.full((n_nodes,), self.ytof_init, device=self.device, dtype=self.dtype)
        self.ytof_p = torch.full((n_nodes,), self.ytof_p_init, device=self.device, dtype=self.dtype)
        self.ytos = torch.full((n_nodes,), self.ytos_init, device=self.device, dtype=self.dtype)
        self.ytos_p = torch.full((n_nodes,), self.ytos_p_init, device=self.device, dtype=self.dtype)

        return V

    def differentiate(self, V):
        self.Cai = self.Cai * 1e-3

        # Compute storevars and external modvars
        Afcaf = (0.3+(0.6/(1.+(torch.exp(((V-(9.24247))/27.96201))))))
        Afs = (1.-(self.Aff))
        ECaj = (((self.R*self.T)/(self.zca*self.F))*(torch.log((self.Cae/self.caj))))
        ECasl = (((self.R*self.T)/(self.zca*self.F))*(torch.log((self.Cae/self.casl))))
        ECl = (((self.R*self.T)/(self.zcl*self.F))*(torch.log((self.Cle/self.Cli))))
        EK = (((self.R*self.T)/(self.zk*self.F))*(torch.log((self.Ke/self.Ki))))
        EKs = (((self.R*self.T)/(self.zk*self.F))*(torch.log(((self.Ke+(self.PNaK*self.Nae))/(self.Ki+(self.PNaK*self.Nai))))))
        ENaj = (((self.R*self.T)/(self.zna*self.F))*(torch.log((self.Nae/self.naj))))
        ENasl = (((self.R*self.T)/(self.zna*self.F))*(torch.log((self.Nae/self.nasl))))
        Fsl = (1.0-(self.Fjunc))
        GCab = (self.ICab_multiplier*self.GCab_b)
        GClCa = (self.IClCa_multiplier*self.GClCa_b)
        GClb = (self.IClb_multiplier*self.GClb_b)
        GKb = (self.GKb_b*self.IKb_multiplier)
        GNa = (self.GNa_b*self.INa_multiplier)
        GNab = (self.INab_multiplier*self.GNab_b)
        Gto_fast = ((0.29856*self.Itof_multiplier) if (self.celltype==1.) else ((0.14928*self.Itof_multiplier) if (self.celltype==2.) else (0.01276*self.Itof_multiplier)))
        Gto_slow = ((0.02036*self.Itos_multiplier) if (self.celltype==1.) else ((0.04632*self.Itos_multiplier) if (self.celltype==2.) else (0.07210*self.Itos_multiplier)))
        IbarNaK = (self.INaK_multiplier*self.IbarNaK_b)
        IbarSLCaP = (((self.IbarSLCaP_b*0.8)*self.IpCa_multiplier) if (self.sex==1.) else (((self.IbarSLCaP_b*1.28)*self.IpCa_multiplier) if (self.sex==2.) else (self.IbarSLCaP_b*self.IpCa_multiplier)))
        Ii = ((0.5*(((self.Nai+self.Ki)+self.Cli)+(4.*self.Cai)))/1000.)
        Io = ((0.5*(((self.Nae+self.Ke)+self.Cle)+(4.*self.Cae)))/1000.)
        Iss = ((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)
        PCa = (((self.PCa_b*self.ICaLPCa_multiplier)*1.025) if (self.celltype==1.) else (((self.PCa_b*self.ICaLPCa_multiplier)*1.1) if (self.celltype==2.) else (self.PCa_b*self.ICaLPCa_multiplier)))
        Qpow = ((self.T-(310.))/10.)
        allo_i = (1./(1.+((self.KmCaAct/self.casl)*(self.KmCaAct/self.casl))))
        allo_ss = (1./(1.+((self.KmCaAct/self.caj)*(self.KmCaAct/self.caj))))
        celltype_factor = (1.25 if (self.celltype==1.) else (0.7 if (self.celltype==2.) else 1.0))
        celltype_factor_IK1 = (1.1 if (self.celltype==1.) else (1.3 if (self.celltype==2.) else 1.0))
        celltype_factor_IKs = (1.4 if (self.celltype==1.) else (0.5 if (self.celltype==2.) else 1.0))
        celltype_factor_INaCa = (1.1 if (self.celltype==1.) else (1.4 if (self.celltype==2.) else 1.0))
        constA = (1.82e6*(pow((self.dielConstant*self.T),-1.5)))
        fICaLP = self.fICaL_PKA
        fICaLp = self.camk_f_ICaL
        fINaLp = self.camk_f_RyR
        fINa_P = self.fINa_PKA
        fINap = self.camk_f_RyR
        fItop = self.camk_f_RyR
        fnak = (0.75+((0.00375-((((140.-(self.Nae))/50.)*0.001)))*V))
        h10_i = ((self.kasymm+1.)+((self.Nae/self.kna1)*(1.+(self.Nae/self.kna2))))
        h10_ss = ((self.kasymm+1.)+((self.Nae/self.kna1)*(1.+(self.Nae/self.kna2))))
        h4_i = (1.+((self.Nai/self.kna1)*(1.+(self.Nai/self.kna2))))
        h4_ss = (1.+((self.nasl/self.kna1)*(1.+(self.nasl/self.kna2))))
        hca = (torch.exp((((self.qca*(V-(8.3117)))*self.F)/(self.R*self.T))))
        hna = (torch.exp((((self.qna*(V-(8.3117)))*self.F)/(self.R*self.T))))
        k2_i = self.kcaoff
        k2_ss = self.kcaoff
        k5_i = self.kcaoff
        k5_ss = self.kcaoff
        kPKA_IKs = self.fIKs_PKA
        sex_factor = (1.1 if (self.sex==1.) else (0.87 if (self.sex==2.) else 1.0))
        sex_factor_IK1 = (1.07 if (self.sex==1.) else (0.92 if (self.sex==2.) else 1.0))
        sex_factor_IKs = (1.09 if (self.sex==1.) else (0.905 if (self.sex==2.) else 1.0))
        sex_factor_INaCa = ((1.0/1.0724) if (self.sex==1.) else (1.0724 if (self.sex==2.) else 1.0))
        vffrt = (((V*self.F)*self.F)/(self.R*self.T))
        vfrt = ((V*self.F)/(self.R*self.T))
        xkb = (1./(1.+(torch.exp(((-(V-(10.8968)))/23.9871)))))
        Afcas = (1.-(Afcaf))
        GK1 = (((self.GK1_b*celltype_factor_IK1)*sex_factor_IK1)*self.IK1_multiplier)
        GKr = (((self.GKr_b*celltype_factor)*sex_factor)*self.IKr_multiplier)
        GKs = (((self.GKs_b*celltype_factor_IKs)*sex_factor_IKs)*self.IKs_multiplier)
        GNaL = ((((self.GNaL_b*self.INaL_multiplier)*(1.+fINaLp))*0.7) if (self.celltype==1.) else ((self.GNaL_b*self.INaL_multiplier)*(1.+fINaLp)))
        GNa_P = (1.25*GNa)
        Gncx = (((self.Gncx_b*celltype_factor_INaCa)*sex_factor_INaCa)*self.INaCa_multiplier)
        ICab_junc = ((self.Fjunc*GCab)*(V-(ECaj)))
        ICab_sl = ((Fsl*GCab)*(V-(ECasl)))
        IClCa_junc = (((0.5*GClCa)/(1.+(self.KdClCa/self.caj)))*(V-(ECl)))
        IClCa_sl = (((0.5*GClCa)/(1.+(self.KdClCa/self.casl)))*(V-(ECl)))
        IClb = (GClb*(V-(ECl)))
        IKb = ((GKb*xkb)*(V-(EK)))
        INaBase_CaMK = (((GNa*(torch.pow(self.m,3.0)))*self.hp)*self.jp)
        INaBase_NP = (((GNa*(torch.pow(self.m,3.0)))*self.h)*self.j)
        INaK_junc_PKA = (((((self.Fjunc*IbarNaK)*fnak)/(1.+((((self.KmNaip_PKA/self.naj)*(self.KmNaip_PKA/self.naj))*(self.KmNaip_PKA/self.naj))*(self.KmNaip_PKA/self.naj))))*self.Ke)/(self.Ke+self.KmKo))
        INaK_junc_noPKA = (((((self.Fjunc*IbarNaK)*fnak)/(1.+((((self.KmNaip/self.naj)*(self.KmNaip/self.naj))*(self.KmNaip/self.naj))*(self.KmNaip/self.naj))))*self.Ke)/(self.Ke+self.KmKo))
        INaK_sl_PKA = (((((Fsl*IbarNaK)*fnak)/(1.+((((self.KmNaip_PKA/self.nasl)*(self.KmNaip_PKA/self.nasl))*(self.KmNaip_PKA/self.nasl))*(self.KmNaip_PKA/self.nasl))))*self.Ke)/(self.Ke+self.KmKo))
        INaK_sl_noPKA = (((((Fsl*IbarNaK)*fnak)/(1.+((((self.KmNaip/self.nasl)*(self.KmNaip/self.nasl))*(self.KmNaip/self.nasl))*(self.KmNaip/self.nasl))))*self.Ke)/(self.Ke+self.KmKo))
        INabj = ((self.Fjunc*GNab)*(V-(ENaj)))
        INabsl = ((Fsl*GNab)*(V-(ENasl)))
        IpCa_junc = ((((self.Fjunc*(pow(self.Q10SLCaP,Qpow)))*IbarSLCaP)*(torch.pow(self.caj,1.6)))/((pow(self.KmPCa,1.6))+(torch.pow(self.caj,1.6))))
        IpCa_sl = ((((Fsl*(pow(self.Q10SLCaP,Qpow)))*IbarSLCaP)*(torch.pow(self.casl,1.6)))/((pow(self.KmPCa,1.6))+(torch.pow(self.casl,1.6))))
        Itof = ((Gto_fast*(V-(EK)))*((((1.-(fItop))*self.xtof)*self.ytof)+((fItop*self.xtof_p)*self.ytof_p)))
        Itos = ((Gto_slow*(V-(EK)))*((((1.-(fItop))*self.xtos)*self.ytos)+((fItop*self.xtos_p)*self.ytos_p)))
        PCaK = (2.211399546628710e-04*PCa)
        PCaNa = (7.734329695819495e-04*PCa)
        PCa_Pb = (1.9*PCa)
        PCap = (1.1*PCa)
        P_g_0 = (self.gKs_factor*(0.2+(0.2*kPKA_IKs)))
        P_g_max = (self.gKs_factor*(0.8+(7.*kPKA_IKs)))
        aK1 = (4.094/(1.+(torch.exp((0.1217*((V-(EK))-(49.934)))))))
        bK1 = (((15.72*(torch.exp((0.0674*((V-(EK))-(3.257))))))+(torch.exp((0.0618*((V-(EK))-(594.31))))))/(1.+(torch.exp((-0.1629*((V-(EK))+14.207))))))
        f = ((self.Aff*self.ff)+(Afs*self.fs))
        fBP = ((self.Aff*self.fBPf)+(Afs*self.fs_P))
        fICaL_P = fICaLP
        fINa_BP = (fINap*fINa_P)
        f_P = ((self.Aff*self.ff_P)+(Afs*self.fs_P))
        fp = ((self.Aff*self.ffp)+(Afs*self.fs))
        gamma_cai = (torch.pow(10.,(((-constA)*4.)*(((torch.sqrt(Ii))/(1.+(torch.sqrt(Ii))))-((0.3*Ii))))))
        gamma_cao = (pow(10.,(((-constA)*4.)*(((sqrt(Io))/(1.+(sqrt(Io))))-((0.3*Io))))))
        gamma_cass = (torch.pow(10.,(((-constA)*4.)*(((torch.sqrt(Iss))/(1.+(torch.sqrt(Iss))))-((0.3*Iss))))))
        gamma_ki = (torch.pow(10.,((-constA)*(((torch.sqrt(Ii))/(1.+(torch.sqrt(Ii))))-((0.3*Ii))))))
        gamma_ko = (pow(10.,((-constA)*(((sqrt(Io))/(1.+(sqrt(Io))))-((0.3*Io))))))
        gamma_kss = (torch.pow(10.,((-constA)*(((torch.sqrt(Iss))/(1.+(torch.sqrt(Iss))))-((0.3*Iss))))))
        gamma_nai = (torch.pow(10.,((-constA)*(((torch.sqrt(Ii))/(1.+(torch.sqrt(Ii))))-((0.3*Ii))))))
        gamma_nao = (pow(10.,((-constA)*(((sqrt(Io))/(1.+(sqrt(Io))))-((0.3*Io))))))
        gamma_nass = (torch.pow(10.,((-constA)*(((torch.sqrt(Iss))/(1.+(torch.sqrt(Iss))))-((0.3*Iss))))))
        h11_i = ((self.Nae*self.Nae)/((h10_i*self.kna1)*self.kna2))
        h11_ss = ((self.Nae*self.Nae)/((h10_ss*self.kna1)*self.kna2))
        h12_i = (1./h10_i)
        h12_ss = (1./h10_ss)
        h1_i = (1.+((self.Nai/self.kna3)*(1.+hna)))
        h1_ss = (1.+((self.nasl/self.kna3)*(1.+hna)))
        h5_i = ((self.Nai*self.Nai)/((h4_i*self.kna1)*self.kna2))
        h5_ss = ((self.nasl*self.nasl)/((h4_ss*self.kna1)*self.kna2))
        h6_i = (1./h4_i)
        h6_ss = (1./h4_ss)
        h7_i = (1.+((self.Nae/self.kna3)*(1.+(1./hna))))
        h7_ss = (1.+((self.Nae/self.kna3)*(1.+(1./hna))))
        IClCa = (IClCa_junc+IClCa_sl)
        IKr = (((GKr*(sqrt((self.Ke/5.))))*self.O)*(V-(EK)))
        INaBase_BP = (((GNa_P*(torch.pow(self.m_P,3.0)))*self.hp_P)*self.jp_P)
        INaBase_PKA = (((GNa_P*(torch.pow(self.m_P,3.0)))*self.h_P)*self.j_P)
        INaKj = (((1.-(self.fINaK_PKA))*INaK_junc_noPKA)+(self.fINaK_PKA*INaK_junc_PKA))
        INaKsl = (((1.-(self.fINaK_PKA))*INaK_sl_noPKA)+(self.fINaK_PKA*INaK_sl_PKA))
        INaLj = ((((self.Fjunc*GNaL)*(V-(ENaj)))*self.mL)*(((1.-(fINaLp))*self.hL)+(fINaLp*self.hLp)))
        INaLsl = ((((Fsl*GNaL)*(V-(ENasl)))*self.mL)*(((1.-(fINaLp))*self.hL)+(fINaLp*self.hLp)))
        Ito = (Itos+Itof)
        K1ss = (aK1/(aK1+bK1))
        PCaKp = (2.211399546628710e-04*PCap)
        PCaNap = (7.734329695819495e-04*PCap)
        PCa_P = ((PCa_Pb*1.025) if (self.celltype==1.) else ((PCa_Pb*1.1) if (self.celltype==2.) else PCa_Pb))
        PhiCaK_i = ((vffrt*(((gamma_ki*self.Ki)*(torch.exp(vfrt)))-((gamma_ko*self.Ke))))/((torch.exp(vfrt))-(1.)))
        PhiCaK_ss = ((vffrt*(((gamma_kss*self.Ki)*(torch.exp(vfrt)))-((gamma_ko*self.Ke))))/((torch.exp(vfrt))-(1.)))
        PhiCaL_i = (((4.*vffrt)*(((gamma_cai*self.Cai)*(torch.exp((2.*vfrt))))-((gamma_cao*self.Cae))))/((torch.exp((2.*vfrt)))-(1.)))
        PhiCaL_ss = (((4.*vffrt)*(((gamma_cass*self.casl)*(torch.exp((2.*vfrt))))-((gamma_cao*self.Cae))))/((torch.exp((2.*vfrt)))-(1.)))
        PhiCaNa_i = ((vffrt*(((gamma_nai*self.Nai)*(torch.exp(vfrt)))-((gamma_nao*self.Nae))))/((torch.exp(vfrt))-(1.)))
        PhiCaNa_ss = ((vffrt*(((gamma_nass*self.nasl)*(torch.exp(vfrt)))-((gamma_nao*self.Nae))))/((torch.exp(vfrt))-(1.)))
        fICaL_BP = (fICaLp*fICaL_P)
        fINa_CaMKonly = (fINap-(fINa_BP))
        fINa_PKAonly = (fINa_P-(fINa_BP))
        fca = ((Afcaf*self.fcaf)+(Afcas*self.fcas))
        fcaBP = ((Afcaf*self.fcaBPf)+(Afcas*self.fcas_P))
        fcap = ((Afcaf*self.fcafp)+(Afcas*self.fcas))
        fcap_P = ((Afcaf*self.fcaf_P)+(Afcas*self.fcas_P))
        gks_junc = (P_g_0+((P_g_max-(P_g_0))/(1.+(torch.pow((150.e-6/self.caj),1.3)))))
        gks_sl = (P_g_0+((P_g_max-(P_g_0))/(1.+(torch.pow((150.e-6/self.casl),1.3)))))
        h2_i = ((self.Nai*hna)/(self.kna3*h1_i))
        h2_ss = ((self.nasl*hna)/(self.kna3*h1_ss))
        h3_i = (1./h1_i)
        h3_ss = (1./h1_ss)
        h8_i = (self.Nae/((self.kna3*hna)*h7_i))
        h8_ss = (self.Nae/((self.kna3*hna)*h7_ss))
        h9_i = (1./h7_i)
        h9_ss = (1./h7_ss)
        k1_i = ((h12_i*self.Cae)*self.kcaon)
        k1_ss = ((h12_ss*self.Cae)*self.kcaon)
        k6_i = ((h6_i*self.Cai)*self.kcaon)
        k6_ss = ((h6_ss*self.casl)*self.kcaon)
        ICaK_i_CaMK = (((PCaKp*PhiCaK_i)*self.d)*((fp*(1.0-(self.nca_i)))+((self.jca*fcap)*self.nca_i)))
        ICaK_i_NP = (((PCaK*PhiCaK_i)*self.d)*((f*(1.0-(self.nca_i)))+((self.jca*fca)*self.nca_i)))
        ICaK_ss_CaMK = (((PCaKp*PhiCaK_ss)*self.d)*((fp*(1.0-(self.nca)))+((self.jca*fcap)*self.nca)))
        ICaK_ss_NP = (((PCaK*PhiCaK_ss)*self.d)*((f*(1.0-(self.nca)))+((self.jca*fca)*self.nca)))
        ICaL_i_BP = (((PCa_P*PhiCaL_i)*self.d_P)*((fBP*(1.0-(self.nca_i)))+((self.jca*fcaBP)*self.nca_i)))
        ICaL_i_CaMK = (((PCap*PhiCaL_i)*self.d)*((fp*(1.0-(self.nca_i)))+((self.jca*fcap)*self.nca_i)))
        ICaL_i_NP = (((PCa*PhiCaL_i)*self.d)*((f*(1.0-(self.nca_i)))+((self.jca*fca)*self.nca_i)))
        ICaL_i_PKA = (((PCa_P*PhiCaL_i)*self.d_P)*((f_P*(1.0-(self.nca_i)))+((self.jca*fcap_P)*self.nca_i)))
        ICaL_ss_BP = (((PCa_P*PhiCaL_ss)*self.d_P)*((fBP*(1.0-(self.nca)))+((self.jca*fcaBP)*self.nca)))
        ICaL_ss_CaMK = (((PCap*PhiCaL_ss)*self.d)*((fp*(1.0-(self.nca)))+((self.jca*fcap)*self.nca)))
        ICaL_ss_NP = (((PCa*PhiCaL_ss)*self.d)*((f*(1.0-(self.nca)))+((self.jca*fca)*self.nca)))
        ICaL_ss_PKA = (((PCa_P*PhiCaL_ss)*self.d_P)*((f_P*(1.0-(self.nca)))+((self.jca*fcap_P)*self.nca)))
        ICaNa_i_CaMK = (((PCaNap*PhiCaNa_i)*self.d)*((fp*(1.0-(self.nca_i)))+((self.jca*fcap)*self.nca_i)))
        ICaNa_i_NP = (((PCaNa*PhiCaNa_i)*self.d)*((f*(1.0-(self.nca_i)))+((self.jca*fca)*self.nca_i)))
        ICaNa_ss_CaMK = (((PCaNap*PhiCaNa_ss)*self.d)*((fp*(1.0-(self.nca)))+((self.jca*fcap)*self.nca)))
        ICaNa_ss_NP = (((PCaNa*PhiCaNa_ss)*self.d)*((f*(1.0-(self.nca)))+((self.jca*fca)*self.nca)))
        ICl_tot = (IClCa+IClb)
        IK1 = (((GK1*(sqrt((self.Ke/5.))))*K1ss)*(V-(EK)))
        IKs_junc = ((((self.Fjunc*GKs)*gks_junc)*(self.xs_junc*self.xs_junc))*(V-(EKs)))
        IKs_sl = ((((Fsl*GKs)*gks_sl)*(self.xs_sl*self.xs_sl))*(V-(EKs)))
        INaBase = (((((((1.-(fINa_CaMKonly))-(fINa_PKAonly))-(fINa_BP))*INaBase_NP)+(fINa_CaMKonly*INaBase_CaMK))+(fINa_PKAonly*INaBase_PKA))+(fINa_BP*INaBase_BP))
        INaK = (INaKj+INaKsl)
        PCaK_P = (2.211399546628710e-04*PCa_P)
        PCaNa_P = (7.734329695819495e-04*PCa_P)
        fICaL_CaMKonly = (fICaLp-(fICaL_BP))
        fICaL_PKAonly = (fICaL_P-(fICaL_BP))
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
        ICaK_i_BP = (((PCaK_P*PhiCaK_i)*self.d_P)*((fBP*(1.0-(self.nca_i)))+((self.jca*fcaBP)*self.nca_i)))
        ICaK_i_PKA = (((PCaK_P*PhiCaK_i)*self.d_P)*((f_P*(1.0-(self.nca_i)))+((self.jca*fcap_P)*self.nca_i)))
        ICaK_ss_BP = (((PCaK_P*PhiCaK_ss)*self.d_P)*((fBP*(1.0-(self.nca)))+((self.jca*fcaBP)*self.nca)))
        ICaK_ss_PKA = (((PCaK_P*PhiCaK_ss)*self.d_P)*((f_P*(1.0-(self.nca)))+((self.jca*fcap_P)*self.nca)))
        ICaL_junc = (((((((((1.-(fICaL_CaMKonly))-(fICaL_PKAonly))-(fICaL_BP))*ICaL_ss_NP)+(fICaL_CaMKonly*ICaL_ss_CaMK))+(fICaL_PKAonly*ICaL_ss_PKA))+(fICaL_BP*ICaL_ss_BP))*self.ical_pureCDI_junc)*self.ICaL_fractionSS)
        ICaL_sl = (((((((((1.-(fICaL_CaMKonly))-(fICaL_PKAonly))-(fICaL_BP))*ICaL_i_NP)+(fICaL_CaMKonly*ICaL_i_CaMK))+(fICaL_PKAonly*ICaL_i_PKA))+(fICaL_BP*ICaL_i_BP))*self.ical_pureCDI_sl)*(1.-(self.ICaL_fractionSS)))
        ICaNa_i_BP = (((PCaNa_P*PhiCaNa_i)*self.d_P)*((fBP*(1.0-(self.nca_i)))+((self.jca*fcaBP)*self.nca_i)))
        ICaNa_i_PKA = (((PCaNa_P*PhiCaNa_i)*self.d_P)*((f_P*(1.0-(self.nca_i)))+((self.jca*fcap_P)*self.nca_i)))
        ICaNa_ss_BP = (((PCaNa_P*PhiCaNa_ss)*self.d_P)*((fBP*(1.0-(self.nca)))+((self.jca*fcaBP)*self.nca)))
        ICaNa_ss_PKA = (((PCaNa_P*PhiCaNa_ss)*self.d_P)*((f_P*(1.0-(self.nca)))+((self.jca*fcap_P)*self.nca)))
        IKs = (IKs_junc+IKs_sl)
        INaj = ((self.Fjunc*INaBase)*(V-(ENaj)))
        INasl = ((Fsl*INaBase)*(V-(ENasl)))
        k3_i = (k3p_i+k3pp_i)
        k3_ss = (k3p_ss+k3pp_ss)
        k4_i = (k4p_i+k4pp_i)
        k4_ss = (k4p_ss+k4pp_ss)
        ICaK_junc = (((((((((1.-(fICaL_CaMKonly))-(fICaL_PKAonly))-(fICaL_BP))*ICaK_ss_NP)+(fICaL_CaMKonly*ICaK_ss_CaMK))+(fICaL_PKAonly*ICaK_ss_PKA))+(fICaL_BP*ICaK_ss_BP))*self.ical_pureCDI_junc)*self.ICaL_fractionSS)
        ICaK_sl = (((((((((1.-(fICaL_CaMKonly))-(fICaL_PKAonly))-(fICaL_BP))*ICaK_i_NP)+(fICaL_CaMKonly*ICaK_i_CaMK))+(fICaL_PKAonly*ICaK_i_PKA))+(fICaL_BP*ICaK_i_BP))*self.ical_pureCDI_sl)*(1.-(self.ICaL_fractionSS)))
        ICaNa_junc = (((((((((1.-(fICaL_CaMKonly))-(fICaL_PKAonly))-(fICaL_BP))*ICaNa_ss_NP)+(fICaL_CaMKonly*ICaNa_ss_CaMK))+(fICaL_PKAonly*ICaNa_ss_PKA))+(fICaL_BP*ICaNa_ss_BP))*self.ical_pureCDI_junc)*self.ICaL_fractionSS)
        ICaNa_sl = (((((((((1.-(fICaL_CaMKonly))-(fICaL_PKAonly))-(fICaL_BP))*ICaNa_i_NP)+(fICaL_CaMKonly*ICaNa_i_CaMK))+(fICaL_PKAonly*ICaNa_i_PKA))+(fICaL_BP*ICaNa_i_BP))*self.ical_pureCDI_sl)*(1.-(self.ICaL_fractionSS)))
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
        ICaK = (ICaK_junc+ICaK_sl)
        IK_tot = ((((((Ito+IKr)+IKs)+IK1)-((2.*INaK)))+ICaK)+IKb)
        JncxCa_i = ((E2_i*k2_i)-((E1_i*k1_i)))
        JncxCa_ss = ((E2_ss*k2_ss)-((E1_ss*k1_ss)))
        JncxNa_i = (((3.*((E4_i*k7_i)-((E1_i*k8_i))))+(E3_i*k4pp_i))-((E2_i*k3pp_i)))
        JncxNa_ss = (((3.*((E4_ss*k7_ss)-((E1_ss*k8_ss))))+(E3_ss*k4pp_ss))-((E2_ss*k3pp_ss)))
        INaCa_i = ((((1.-(self.INaCa_fractionSS))*Gncx)*allo_i)*((self.zna*JncxNa_i)+(self.zca*JncxCa_i)))
        INaCa_ss = (((self.INaCa_fractionSS*Gncx)*allo_ss)*((self.zna*JncxNa_ss)+(self.zca*JncxCa_ss)))
        ICa_tot_junc = (((ICaL_junc+ICab_junc)+IpCa_junc)-((2.*INaCa_ss)))
        ICa_tot_sl = (((ICaL_sl+ICab_sl)+IpCa_sl)-((2.*INaCa_i)))
        INa_tot_junc = (((((INaj+INaLj)+INabj)+(3.*INaCa_ss))+(3.*INaKj))+ICaNa_junc)
        INa_tot_sl = (((((INasl+INaLsl)+INabsl)+(3.*INaCa_i))+(3.*INaKsl))+ICaNa_sl)
        Iion = (((((INa_tot_junc+INa_tot_sl)+ICa_tot_junc)+ICa_tot_sl)+IK_tot)+ICl_tot)

        # Complete Forward Euler Update
        A = ((self.TOT_A*self.dr)/(((1.-(self.dr))*self.wfrac)+self.dr))
        Bmax_CaM = ((0.024/1.1) if (self.sex==1.) else ((0.024*1.1) if (self.sex==2.) else 0.024))
        CaMKIILeakMultiplier = (1.+(2.*self.camk_f_RyR))
        ICaL_junc_positive = (torch.abs(ICaL_junc))
        Jrel_ICaLdep = (((0.00174*self.jrel_icaldep_act)*self.jrel_icaldep_f1)*self.jrel_icaldep_f2)
        Kmf_p = (self.Kmf*0.5)
        PP1_tot = self.Whole_cell_PP1
        Vmax_SRCaP = (((1.2*self.Vmax_SRCaP_b)*self.Jup_multiplier) if (self.celltype==1.) else (self.Vmax_SRCaP_b*self.Jup_multiplier))
        XSSS = (self.dr*0.5)
        XU = (((1.-(self.contraction_TmBlocked))-(self.contraction_XW))-(self.contraction_XS))
        XWSS = (((1.-(self.dr))*self.wfrac)*0.5)
        a_rel = (1.25*self.bt)
        bound = ((self.CaMK0*(1.-(self.camk_trap)))/(1.+(self.Km_CaMK_Ca/self.caj)))
        bound_serca = ((self.CaMK0*(1.-(self.casig_serca_trap)))/(1.+(self.Km_CaMK_Ca/self.caj)))
        dPss = (torch.clamp((1.0323*(torch.exp((-1.0553*(torch.exp((-0.0810*(V+12.62483)))))))), max=1.))
        d_buffers_Myosin_ca = (((self.kon_myoca*self.Cai)*((self.Bmax_myosin-(self.buffers_Myosin_ca))-(self.buffers_Myosin_mg)))-((self.koff_myoca*self.buffers_Myosin_ca)))
        d_buffers_NaBj = (((self.kon_na*self.naj)*(self.Bmax_Naj-(self.buffers_NaBj)))-((self.koff_na*self.buffers_NaBj)))
        d_buffers_NaBsl = (((self.kon_na*self.nasl)*(self.Bmax_Nasl-(self.buffers_NaBsl)))-((self.koff_na*self.buffers_NaBsl)))
        d_buffers_SRB = (((self.kon_sr*self.Cai)*(self.Bmax_SR-(self.buffers_SRB)))-((self.koff_sr*self.buffers_SRB)))
        d_buffers_TnCHc = (((self.kon_tnchca*self.Cai)*((self.Bmax_TnChigh-(self.buffers_TnCHc))-(self.buffers_TnCHm)))-((self.koff_tnchca*self.buffers_TnCHc)))
        diff_buffers_Myosin_mg = (((self.kon_myomg*self.mgi)*((self.Bmax_myosin-(self.buffers_Myosin_ca))-(self.buffers_Myosin_mg)))-((self.koff_myomg*self.buffers_Myosin_mg)))
        diff_buffers_TnCHm = (((self.kon_tnchmg*self.mgi)*((self.Bmax_TnChigh-(self.buffers_TnCHc))-(self.buffers_TnCHm)))-((self.koff_tnchmg*self.buffers_TnCHm)))
        dss = (torch.clamp((1.0763*(torch.exp((-1.007*(torch.exp((-0.0829*(V+3.62483)))))))), max=1.))
        fPKA_TnI = (1.45-(((0.45*(1.-(self.fTnI_PKA)))/(1.-(self.fracTnIpo)))))
        fss_P = (1.0/(1.0+(torch.exp(((V+25.58)/3.696)))))
        gamma_rate = (torch.where((self.contraction_ZETAS>0.), (self.gamma*self.contraction_ZETAS), (torch.where((self.contraction_ZETAS<-1.), (self.gamma*((-self.contraction_ZETAS)-(1.))), 0.0))))
        gamma_rate_w = (self.gamma_wu*(torch.abs(self.contraction_ZETAW)))
        k_uw = (0.026*self.nu)
        k_ws = ((0.004*(1.+(self.fMyBPC_PKA/2.)))*self.mu)
        km2n = (self.jca*0.84191)
        ks = (self.Jrel_multiplier*26.6)
        nonlinearModifier = (0.2144*(torch.exp((1.83*self.casr))))
        phosphorylationTotal = ((self.camk_f_PLB+self.fPLB_PKA)-((self.camk_f_PLB*self.fPLB_PKA)))
        sigmoidTransition = (1.-((1./(1.+((1.86532*self.caj)/0.032)))))
        sigmoidTransition2 = (1.-((1./(1.+((1.86532*self.casl)/0.032)))))
        tau_cal = self.tau_ryr
        tau_rel = (torch.clamp((self.bt/(1.0+(0.0123/self.casr))), min=0.001))
        td = (1.5+(1./((torch.exp((-0.05*(V+6.))))+(torch.exp((0.09*(V+14.)))))))
        tfcaf = (13.50673+(1./((0.1542*(torch.exp(((-(V-(1.31611)))/11.3396))))+(0.1542*(torch.exp(((V-(1.31611))/11.3396)))))))
        tfcas = (177.95813+(1./((4.73955e-4*(torch.exp(((-(V-(0.79049)))/0.81777))))+(4.73955e-4*(torch.exp(((V+2.40474)/1.90812)))))))
        tff = (6.17111+(1./((0.00126*(torch.exp(((-(V+26.63596))/9.69961))))+(0.00126*(torch.exp(((V+26.63596)/9.69961)))))))
        tfs = (2719.22489+(1./((7.19411e-5*(torch.exp(((-(V+5.74631))/10.8769))))+(7.19411e-5*(torch.exp(((V+5.74631)/16.31535)))))))
        vcell = (((1.e-15*math.pi)*(self.cellRadius*self.cellRadius))*self.cellLength)
        CaMK_active = (bound+self.camk_trap)
        ICaL_junc_sigmoided = (1.-((1./(1.+(torch.pow((ICaL_junc_positive/0.45),4.5))))))
        J_SRCarel_np = (((ks*self.ryr_O)*(self.casr-(self.caj)))+Jrel_ICaLdep)
        J_SRCarel_p = (((ks*self.ryr_O_p)*(self.casr-(self.caj)))+Jrel_ICaLdep)
        J_SRleak = (((1.59306e-6*(self.casr-(self.caj)))*nonlinearModifier)*CaMKIILeakMultiplier)
        anca = (1./((self.k2n/km2n)+(torch.pow((1.+(self.Kmn/self.caj)),3.80763))))
        anca_i = (1./((self.k2n/km2n)+(torch.pow((1.+(self.Kmn/self.casl)),3.80763))))
        ca50 = ((fPKA_TnI*0.7645)+(self.beta_1*(min((self.lambda_-(1.)),0.2))))
        casig_SERCA_act = (bound_serca+self.casig_serca_trap)
        cds = ((((self.phi*k_ws)*self.wfrac)*(1.-(self.dr)))/self.dr)
        cdw = (((self.phi*k_uw)*(1.-(self.wfrac)))/self.wfrac)
        d_buffers_CaM = (((self.kon_cam*self.Cai)*(Bmax_CaM-(self.buffers_CaM)))-((self.koff_cam*self.buffers_CaM)))
        diff_buffers_Myosin_ca = d_buffers_Myosin_ca
        diff_buffers_NaBj = d_buffers_NaBj
        diff_buffers_NaBsl = d_buffers_NaBsl
        diff_buffers_SRB = d_buffers_SRB
        diff_buffers_TnCHc = d_buffers_TnCHc
        diff_d = ((dss-(self.d))/td)
        diff_d_P = ((dPss-(self.d_P))/td)
        diff_ff_P = ((fss_P-(self.ff_P))/tff)
        diff_fs_P = ((fss_P-(self.fs_P))/tfs)
        fBPss = fss_P
        fcass_P = fss_P
        k_su = ((k_ws*((1./self.dr)-(1.)))*self.wfrac)
        k_wu = ((k_uw*((1./self.wfrac)-(1.)))-(k_ws))
        ktm_block = (((self.ktm_unblock*(pow(self.perm50,self.nperm)))*0.5)/((0.5-(XSSS))-(XWSS)))
        tauTransition = (1.0967+((1.-(sigmoidTransition))*141.4299))
        tauTransition2 = (1.0967+((1.-(sigmoidTransition2))*141.4299))
        tfcafp = (2.5*tfcaf)
        tffp = (2.5*tff)
        vjunc = (5.39e-4*vcell)
        vmyo = (0.65*vcell)
        vsl = (0.02*vcell)
        vsr = (0.035*vcell)
        xb_su_gamma = (gamma_rate*self.contraction_XS)
        xb_uw = ((k_uw*XU)*(1.0+(self.fMyBPC_PKA/2.)))
        xb_ws = (k_ws*self.contraction_XW)
        xb_wu_gamma = (gamma_rate_w*self.contraction_XW)
        Bmax_SLhighj = ((1.75755e-4*vmyo)/vjunc)
        Bmax_SLhighsl = ((1.215423e-2*vmyo)/vsl)
        Bmax_SLlowj = ((4.89983e-4*vmyo)/vjunc)
        Bmax_SLlowsl = ((3.3923e-2*vmyo)/vsl)
        CaMK_Phos_ss_ICaL = (CaMK_active/(CaMK_active+self.K_Phos_CaMK))
        CaMK_Phos_ss_PLB = (CaMK_active/(CaMK_active+10.))
        CaMK_Phos_ss_RyR = (CaMK_active/(CaMK_active+1.))
        J_SRCarel = ((J_SRCarel_p*self.camk_f_RyR)+(J_SRCarel_np*(1.-(self.camk_f_RyR))))
        Jrel_inf = ((a_rel*ICaL_junc_sigmoided)/(1.0+(torch.pow((self.directRelMidpoint/self.casr),7.72672))))
        Vmax_mult = (1.+(self.Max_Vmax_SERCA_Ca/(1.+((self.Km_SERCA_Ca/casig_SERCA_act)*(self.Km_SERCA_Ca/casig_SERCA_act)))))
        d_contraction_Ca_TRPN = (self.koff*(((torch.pow(((self.Cai*1000.)/ca50),self.TRPN_n))*(1.-(self.contraction_Ca_TRPN)))-(self.contraction_Ca_TRPN)))
        diff_Cli = ((ICl_tot*self.Cmem)/(vmyo*self.F))
        diff_Ki = (((-IK_tot)*self.Cmem)/(vmyo*self.F))
        diff_Nai = ((self.J_na_slmyo/vmyo)*(self.nasl-(self.Nai)))
        diff_buffers_CaM = d_buffers_CaM
        diff_camk_trap = (((self.alpha*bound)*CaMK_active)-(((self.beta*self.camk_trap)*(0.1+((0.9*PP1_tot)/0.1371)))))
        diff_casig_serca_trap = (((self.alpha_serca*bound_serca)*casig_SERCA_act)-(((self.beta*self.casig_serca_trap)*(0.1+((0.9*PP1_tot)/0.1371)))))
        diff_contraction_TmBlocked = (((ktm_block*XU)*(torch.clamp((torch.pow(self.contraction_Ca_TRPN,((-self.nperm)/2.))), max=100.)))-(((self.ktm_unblock*(torch.pow(self.contraction_Ca_TRPN,(self.nperm/2.))))*self.contraction_TmBlocked)))
        diff_contraction_ZETAS = ((A*self.lambda_rate)-((cds*self.contraction_ZETAS)))
        diff_contraction_ZETAW = ((A*self.lambda_rate)-((cdw*self.contraction_ZETAW)))
        diff_fBPf = ((fBPss-(self.fBPf))/tffp)
        diff_fcaf_P = ((fcass_P-(self.fcaf_P))/tfcaf)
        diff_fcas_P = ((fcass_P-(self.fcas_P))/tfcas)
        diff_ical_pureCDI_junc = ((((-self.ical_pureCDI_junc)*sigmoidTransition)/tauTransition)+((1.-(self.ical_pureCDI_junc))*self.rateRecovery))
        diff_ical_pureCDI_sl = ((((-self.ical_pureCDI_sl)*sigmoidTransition2)/tauTransition2)+((1.-(self.ical_pureCDI_sl))*self.rateRecovery))
        diff_naj = (((((-INa_tot_junc)*self.Cmem)/(vjunc*self.F))+((self.J_na_juncsl/vjunc)*(self.nasl-(self.naj))))-(d_buffers_NaBj))
        diff_nasl = ((((((-INa_tot_sl)*self.Cmem)/(vsl*self.F))+((self.J_na_juncsl/vsl)*(self.naj-(self.nasl))))+((self.J_na_slmyo/vsl)*(self.Nai-(self.nasl))))-(d_buffers_NaBsl))
        diff_nca = ((anca*self.k2n)-((self.nca*km2n)))
        diff_nca_i = ((anca_i*self.k2n)-((self.nca_i*km2n)))
        fcaBPss = fcass_P
        xb_su = (k_su*self.contraction_XS)
        xb_wu = (k_wu*self.contraction_XW)
        Jserca_np = (((((pow(self.Q10SRCaP,Qpow))*Vmax_SRCaP)*Vmax_mult)*((torch.pow((self.Cai/self.Kmf),self.hillSRCaP))-((torch.pow((self.casr/self.Kmr),self.hillSRCaP)))))/((1.+(torch.pow((self.Cai/self.Kmf),self.hillSRCaP)))+(torch.pow((self.casr/self.Kmr),self.hillSRCaP))))
        Jserca_p = (((((pow(self.Q10SRCaP,Qpow))*Vmax_SRCaP)*Vmax_mult)*((torch.pow((self.Cai/Kmf_p),self.hillSRCaP))-((torch.pow((self.casr/self.Kmr),self.hillSRCaP)))))/((1.+(torch.pow((self.Cai/Kmf_p),self.hillSRCaP)))+(torch.pow((self.casr/self.Kmr),self.hillSRCaP))))
        d_buffers_SLHj = (((self.kon_slh*self.caj)*(Bmax_SLhighj-(self.buffers_SLHj)))-((self.koff_slh*self.buffers_SLHj)))
        d_buffers_SLHsl = (((self.kon_slh*self.casl)*(Bmax_SLhighsl-(self.buffers_SLHsl)))-((self.koff_slh*self.buffers_SLHsl)))
        d_buffers_SLLj = (((self.kon_sll*self.caj)*(Bmax_SLlowj-(self.buffers_SLLj)))-((self.koff_sll*self.buffers_SLLj)))
        d_buffers_SLLsl = (((self.kon_sll*self.casl)*(Bmax_SLlowsl-(self.buffers_SLLsl)))-((self.koff_sll*self.buffers_SLLsl)))
        d_buffers_TnClow = (self.Bmax_TnClow*d_contraction_Ca_TRPN)
        diff_camk_f_ICaL = ((CaMK_Phos_ss_ICaL-(self.camk_f_ICaL))/tau_cal)
        diff_camk_f_PLB = ((CaMK_Phos_ss_PLB-(self.camk_f_PLB))/self.tau_plb)
        diff_camk_f_RyR = ((CaMK_Phos_ss_RyR-(self.camk_f_RyR))/self.tau_ryr)
        diff_contraction_Ca_TRPN = d_contraction_Ca_TRPN
        diff_contraction_XS = ((xb_ws-(xb_su))-(xb_su_gamma))
        diff_contraction_XW = (((xb_uw-(xb_wu))-(xb_ws))-(xb_wu_gamma))
        diff_fcaBPf = ((fcaBPss-(self.fcaBPf))/tfcafp)
        diff_jrel_icaldep_act = ((Jrel_inf-(self.jrel_icaldep_act))/tau_rel)
        J_CaB_cytosol = ((((d_buffers_TnClow+d_buffers_TnCHc)+d_buffers_CaM)+d_buffers_Myosin_ca)+d_buffers_SRB)
        J_CaB_junction = (d_buffers_SLLj+d_buffers_SLHj)
        J_CaB_sl = (d_buffers_SLLsl+d_buffers_SLHsl)
        Jserca = ((Jserca_np*(1.-(phosphorylationTotal)))+(Jserca_p*phosphorylationTotal))
        diff_buffers_SLHj = d_buffers_SLHj
        diff_buffers_SLHsl = d_buffers_SLHsl
        diff_buffers_SLLj = d_buffers_SLLj
        diff_buffers_SLLsl = d_buffers_SLLsl
        diff_buffers_TnClow = d_buffers_TnClow
        diff_Cai = (((((-Jserca)*vsr)/vmyo)-(J_CaB_cytosol))+((self.J_ca_slmyo/vmyo)*(self.casl-(self.Cai))))
        diff_caj = (((((((-ICa_tot_junc)*self.Cmem)/((vjunc*2.)*self.F))+((self.J_ca_juncsl/vjunc)*(self.casl-(self.caj))))-(J_CaB_junction))+((J_SRCarel*vsr)/vjunc))+((J_SRleak*vmyo)/vjunc))
        diff_casl = ((((((-ICa_tot_sl)*self.Cmem)/((vsl*2.)*self.F))+((self.J_ca_juncsl/vsl)*(self.caj-(self.casl))))+((self.J_ca_slmyo/vsl)*(self.Cai-(self.casl))))-(J_CaB_sl))
        Cai_new = self.Cai+diff_Cai*self.dt
        Cli_new = self.Cli+diff_Cli*self.dt
        Ki_new = self.Ki+diff_Ki*self.dt
        Nai_new = self.Nai+diff_Nai*self.dt
        buffers_CaM_new = self.buffers_CaM+diff_buffers_CaM*self.dt
        buffers_Myosin_ca_new = self.buffers_Myosin_ca+diff_buffers_Myosin_ca*self.dt
        buffers_Myosin_mg_new = self.buffers_Myosin_mg+diff_buffers_Myosin_mg*self.dt
        buffers_NaBj_new = self.buffers_NaBj+diff_buffers_NaBj*self.dt
        buffers_NaBsl_new = self.buffers_NaBsl+diff_buffers_NaBsl*self.dt
        buffers_SLHj_new = self.buffers_SLHj+diff_buffers_SLHj*self.dt
        buffers_SLHsl_new = self.buffers_SLHsl+diff_buffers_SLHsl*self.dt
        buffers_SLLj_new = self.buffers_SLLj+diff_buffers_SLLj*self.dt
        buffers_SLLsl_new = self.buffers_SLLsl+diff_buffers_SLLsl*self.dt
        buffers_SRB_new = self.buffers_SRB+diff_buffers_SRB*self.dt
        buffers_TnCHc_new = self.buffers_TnCHc+diff_buffers_TnCHc*self.dt
        buffers_TnCHm_new = self.buffers_TnCHm+diff_buffers_TnCHm*self.dt
        buffers_TnClow_new = self.buffers_TnClow+diff_buffers_TnClow*self.dt
        caj_new = self.caj+diff_caj*self.dt
        camk_f_ICaL_new = self.camk_f_ICaL+diff_camk_f_ICaL*self.dt
        camk_f_PLB_new = self.camk_f_PLB+diff_camk_f_PLB*self.dt
        camk_f_RyR_new = self.camk_f_RyR+diff_camk_f_RyR*self.dt
        camk_trap_new = self.camk_trap+diff_camk_trap*self.dt
        casig_serca_trap_new = self.casig_serca_trap+diff_casig_serca_trap*self.dt
        casl_new = self.casl+diff_casl*self.dt
        contraction_Ca_TRPN_new = self.contraction_Ca_TRPN+diff_contraction_Ca_TRPN*self.dt
        contraction_TmBlocked_new = self.contraction_TmBlocked+diff_contraction_TmBlocked*self.dt
        contraction_XS_new = self.contraction_XS+diff_contraction_XS*self.dt
        contraction_XW_new = self.contraction_XW+diff_contraction_XW*self.dt
        contraction_ZETAS_new = self.contraction_ZETAS+diff_contraction_ZETAS*self.dt
        contraction_ZETAW_new = self.contraction_ZETAW+diff_contraction_ZETAW*self.dt
        d_new = self.d+diff_d*self.dt
        d_P_new = self.d_P+diff_d_P*self.dt
        fBPf_new = self.fBPf+diff_fBPf*self.dt
        fcaBPf_new = self.fcaBPf+diff_fcaBPf*self.dt
        fcaf_P_new = self.fcaf_P+diff_fcaf_P*self.dt
        fcas_P_new = self.fcas_P+diff_fcas_P*self.dt
        ff_P_new = self.ff_P+diff_ff_P*self.dt
        fs_P_new = self.fs_P+diff_fs_P*self.dt
        ical_pureCDI_junc_new = self.ical_pureCDI_junc+diff_ical_pureCDI_junc*self.dt
        ical_pureCDI_sl_new = self.ical_pureCDI_sl+diff_ical_pureCDI_sl*self.dt
        jrel_icaldep_act_new = self.jrel_icaldep_act+diff_jrel_icaldep_act*self.dt
        naj_new = self.naj+diff_naj*self.dt
        nasl_new = self.nasl+diff_nasl*self.dt
        nca_new = self.nca+diff_nca*self.dt
        nca_i_new = self.nca_i+diff_nca_i*self.dt

        # Complete Rush Larsen Update
        P_tau_0 = (26.+(9.*kPKA_IKs))
        P_tau_max = (40.+(4.*kPKA_IKs))
        ah = (torch.where((V>=-40.), 0., (0.057*(torch.exp(((-(V+80.))/6.8))))))
        aj = (torch.where((V>=-40.), 0., ((((-25428.0*(torch.exp((0.2444*V))))-((6.948e-6*(torch.exp((-0.04391*V))))))*(V+37.78))/(1.+(torch.exp((0.311*(V+79.23))))))))
        bh = (torch.where((V>=-40.), (0.77/(0.13*(1.+(torch.exp(((-(V+10.66))/11.1)))))), ((2.7*(torch.exp((0.079*V))))+(3.1e5*(torch.exp((0.3485*V)))))))
        bj = (torch.where((V>=-40.), ((0.6*(torch.exp((0.057*V))))/(1.+(torch.exp((-0.1*(V+32.)))))), ((0.02424*(torch.exp((-0.01052*V))))/(1.+(torch.exp((-0.1378*(V+40.14))))))))
        dti_develop = (1.354+(1.0e-4/((torch.exp(((V-(167.4))/15.89)))+(torch.exp(((-(V-(12.23)))/0.2154))))))
        dti_recover = (1.0-((0.5/(1.0+(torch.exp(((V+70.0)/20.0)))))))
        partial_diff_fcaf_del_fcaf = ((tfcaf*-1.)/(tfcaf*tfcaf))
        partial_diff_fcafp_del_fcafp = ((tfcafp*-1.)/(tfcafp*tfcafp))
        partial_diff_fcas_del_fcas = ((tfcas*-1.)/(tfcas*tfcas))
        partial_diff_ff_del_ff = ((tff*-1.)/(tff*tff))
        partial_diff_ffp_del_ffp = ((tffp*-1.)/(tffp*tffp))
        partial_diff_fs_del_fs = ((tfs*-1.)/(tfs*tfs))
        partial_diff_hL_del_hL = ((self.thL*-1.)/(self.thL*self.thL))
        partial_diff_jca_del_jca = ((self.tjca*-1.)/(self.tjca*self.tjca))
        partial_diff_jrel_icaldep_f1_del_jrel_icaldep_f1 = ((self.tauInact*-1.)/(self.tauInact*self.tauInact))
        partial_diff_jrel_icaldep_f2_del_jrel_icaldep_f2 = ((self.tauInact2*-1.)/(self.tauInact2*self.tauInact2))
        set_fcaf_tozero_in_diff_fcaf = ((1./(1.+(torch.exp(((V+19.58)/3.696)))))/(13.50673+(1./((0.1542*(torch.exp(((-(V-(1.31611)))/11.3396))))+(0.1542*(torch.exp(((V-(1.31611))/11.3396))))))))
        set_fcafp_tozero_in_diff_fcafp = ((1./(1.+(torch.exp(((V+19.58)/3.696)))))/(2.5*(13.50673+(1./((0.1542*(torch.exp(((-(V-(1.31611)))/11.3396))))+(0.1542*(torch.exp(((V-(1.31611))/11.3396)))))))))
        set_fcas_tozero_in_diff_fcas = ((1./(1.+(torch.exp(((V+19.58)/3.696)))))/(177.95813+(1./((4.73955e-4*(torch.exp(((-(V-(0.79049)))/0.81777))))+(4.73955e-4*(torch.exp(((V+2.40474)/1.90812))))))))
        set_ff_tozero_in_diff_ff = ((1./(1.+(torch.exp(((V+19.58)/3.696)))))/(6.17111+(1./((0.00126*(torch.exp(((-(V+26.63596))/9.69961))))+(0.00126*(torch.exp(((V+26.63596)/9.69961))))))))
        set_ffp_tozero_in_diff_ffp = ((1./(1.+(torch.exp(((V+19.58)/3.696)))))/(2.5*(6.17111+(1./((0.00126*(torch.exp(((-(V+26.63596))/9.69961))))+(0.00126*(torch.exp(((V+26.63596)/9.69961)))))))))
        set_fs_tozero_in_diff_fs = ((1./(1.+(torch.exp(((V+19.58)/3.696)))))/(2719.22489+(1./((7.19411e-5*(torch.exp(((-(V+5.74631))/10.8769))))+(7.19411e-5*(torch.exp(((V+5.74631)/16.31535))))))))
        set_hL_tozero_in_diff_hL = ((1./(1.+(torch.exp(((V+87.61)/7.488)))))/145.)
        set_hLp_tozero_in_diff_hLp = ((1./(1.+(torch.exp(((V+93.81)/7.488)))))/(3.*145.))
        set_h_P_tozero_in_diff_h_P = ((1./((1.+(torch.exp(((V+76.55)/7.43))))*(1.+(torch.exp(((V+76.55)/7.43))))))/(1./((torch.where((V>=-40.), 0., (0.057*(torch.exp(((-(V+80.))/6.8))))))+(torch.where((V>=-40.), (0.77/(0.13*(1.+(torch.exp(((-(V+10.66))/11.1)))))), ((2.7*(torch.exp((0.079*V))))+(3.1e5*(torch.exp((0.3485*V))))))))))
        set_h_tozero_in_diff_h = ((1./((1.+(torch.exp(((V+71.55)/7.43))))*(1.+(torch.exp(((V+71.55)/7.43))))))/(1./((torch.where((V>=-40.), 0., (0.057*(torch.exp(((-(V+80.))/6.8))))))+(torch.where((V>=-40.), (0.77/(0.13*(1.+(torch.exp(((-(V+10.66))/11.1)))))), ((2.7*(torch.exp((0.079*V))))+(3.1e5*(torch.exp((0.3485*V))))))))))
        set_hp_P_tozero_in_diff_hp_P = ((1./((1.+(torch.exp(((V+82.55)/7.43))))*(1.+(torch.exp(((V+82.55)/7.43))))))/(1./((torch.where((V>=-40.), 0., (0.057*(torch.exp(((-(V+80.))/6.8))))))+(torch.where((V>=-40.), (0.77/(0.13*(1.+(torch.exp(((-(V+10.66))/11.1)))))), ((2.7*(torch.exp((0.079*V))))+(3.1e5*(torch.exp((0.3485*V))))))))))
        set_hp_tozero_in_diff_hp = ((1./((1.+(torch.exp(((V+77.55)/7.43))))*(1.+(torch.exp(((V+77.55)/7.43))))))/(1./((torch.where((V>=-40.), 0., (0.057*(torch.exp(((-(V+80.))/6.8))))))+(torch.where((V>=-40.), (0.77/(0.13*(1.+(torch.exp(((-(V+10.66))/11.1)))))), ((2.7*(torch.exp((0.079*V))))+(3.1e5*(torch.exp((0.3485*V))))))))))
        set_j_P_tozero_in_diff_j_P = ((1./((1.+(torch.exp(((V+76.55)/7.43))))*(1.+(torch.exp(((V+76.55)/7.43))))))/(1./((torch.where((V>=-40.), 0., ((((-25428.0*(torch.exp((0.2444*V))))-((6.948e-6*(torch.exp((-0.04391*V))))))*(V+37.78))/(1.+(torch.exp((0.311*(V+79.23))))))))+(torch.where((V>=-40.), ((0.6*(torch.exp((0.057*V))))/(1.+(torch.exp((-0.1*(V+32.)))))), ((0.02424*(torch.exp((-0.01052*V))))/(1.+(torch.exp((-0.1378*(V+40.14)))))))))))
        set_j_tozero_in_diff_j = ((1./((1.+(torch.exp(((V+71.55)/7.43))))*(1.+(torch.exp(((V+71.55)/7.43))))))/(1./((torch.where((V>=-40.), 0., ((((-25428.0*(torch.exp((0.2444*V))))-((6.948e-6*(torch.exp((-0.04391*V))))))*(V+37.78))/(1.+(torch.exp((0.311*(V+79.23))))))))+(torch.where((V>=-40.), ((0.6*(torch.exp((0.057*V))))/(1.+(torch.exp((-0.1*(V+32.)))))), ((0.02424*(torch.exp((-0.01052*V))))/(1.+(torch.exp((-0.1378*(V+40.14)))))))))))
        set_jca_tozero_in_diff_jca = ((1.0/(1.0+(torch.exp(((V+17.66945)/3.21501)))))/66.0)
        set_jp_P_tozero_in_diff_jp_P = ((1./((1.+(torch.exp(((V+82.55)/7.43))))*(1.+(torch.exp(((V+82.55)/7.43))))))/(1.46*(1./((torch.where((V>=-40.), 0., ((((-25428.0*(torch.exp((0.2444*V))))-((6.948e-6*(torch.exp((-0.04391*V))))))*(V+37.78))/(1.+(torch.exp((0.311*(V+79.23))))))))+(torch.where((V>=-40.), ((0.6*(torch.exp((0.057*V))))/(1.+(torch.exp((-0.1*(V+32.)))))), ((0.02424*(torch.exp((-0.01052*V))))/(1.+(torch.exp((-0.1378*(V+40.14))))))))))))
        set_jp_tozero_in_diff_jp = ((1./((1.+(torch.exp(((V+71.55)/7.43))))*(1.+(torch.exp(((V+71.55)/7.43))))))/(1.46*(1./((torch.where((V>=-40.), 0., ((((-25428.0*(torch.exp((0.2444*V))))-((6.948e-6*(torch.exp((-0.04391*V))))))*(V+37.78))/(1.+(torch.exp((0.311*(V+79.23))))))))+(torch.where((V>=-40.), ((0.6*(torch.exp((0.057*V))))/(1.+(torch.exp((-0.1*(V+32.)))))), ((0.02424*(torch.exp((-0.01052*V))))/(1.+(torch.exp((-0.1378*(V+40.14))))))))))))
        set_jrel_icaldep_f1_tozero_in_diff_jrel_icaldep_f1 = ((1./(1.+((1.-((1./(1.+(torch.pow(((torch.abs((((((1.-(self.camk_f_ICaL))*(((((1.5768e-04))*(((4.*(((V*96485.)*96485.)/(8314.*310.)))*((((torch.pow(10.,(((-(1.82e6*(pow((74.*310.),-1.5))))*4.)*(((torch.sqrt(((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)))/(1.+(torch.sqrt(((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)))))-((0.3*((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)))))))*self.casl)*(torch.exp((2.*((V*96485.)/(8314.*310.))))))-(((pow(10.,(((-(1.82e6*(pow((74.*310.),-1.5))))*4.)*(((sqrt(((0.5*(((140.0+5.0)+150.0)+(4.*1.8)))/1000.)))/(1.+(sqrt(((0.5*(((140.0+5.0)+150.0)+(4.*1.8)))/1000.)))))-((0.3*((0.5*(((140.0+5.0)+150.0)+(4.*1.8)))/1000.)))))))*1.8))))/((torch.exp((2.*((V*96485.)/(8314.*310.)))))-(1.))))*self.d)*((((0.52477*self.ff)+((1.-(0.52477))*self.fs))*(1.0-(self.nca)))+((self.jca*(((0.3+(0.6/(1.+(torch.exp(((V-(9.24247))/27.96201))))))*self.fcaf)+((1.-((0.3+(0.6/(1.+(torch.exp(((V-(9.24247))/27.96201))))))))*self.fcas)))*self.nca))))+(self.camk_f_ICaL*((((1.1*((1.5768e-04)))*(((4.*(((V*96485.)*96485.)/(8314.*310.)))*((((torch.pow(10.,(((-(1.82e6*(pow((74.*310.),-1.5))))*4.)*(((torch.sqrt(((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)))/(1.+(torch.sqrt(((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)))))-((0.3*((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)))))))*self.casl)*(torch.exp((2.*((V*96485.)/(8314.*310.))))))-(((pow(10.,(((-(1.82e6*(pow((74.*310.),-1.5))))*4.)*(((sqrt(((0.5*(((140.0+5.0)+150.0)+(4.*1.8)))/1000.)))/(1.+(sqrt(((0.5*(((140.0+5.0)+150.0)+(4.*1.8)))/1000.)))))-((0.3*((0.5*(((140.0+5.0)+150.0)+(4.*1.8)))/1000.)))))))*1.8))))/((torch.exp((2.*((V*96485.)/(8314.*310.)))))-(1.))))*self.d)*((((0.52477*self.ffp)+((1.-(0.52477))*self.fs))*(1.0-(self.nca)))+((self.jca*(((0.3+(0.6/(1.+(torch.exp(((V-(9.24247))/27.96201))))))*self.fcafp)+((1.-((0.3+(0.6/(1.+(torch.exp(((V-(9.24247))/27.96201))))))))*self.fcas)))*self.nca)))))*self.ical_pureCDI_junc)*0.8)))/0.45),4.5))))))/1.e-3)))/64.11202)
        set_jrel_icaldep_f2_tozero_in_diff_jrel_icaldep_f2 = ((1./(1.+((1.-((1./(1.+(torch.pow(((torch.abs((((((1.-(self.camk_f_ICaL))*(((((1.5768e-04))*(((4.*(((V*96485.)*96485.)/(8314.*310.)))*((((torch.pow(10.,(((-(1.82e6*(pow((74.*310.),-1.5))))*4.)*(((torch.sqrt(((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)))/(1.+(torch.sqrt(((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)))))-((0.3*((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)))))))*self.casl)*(torch.exp((2.*((V*96485.)/(8314.*310.))))))-(((pow(10.,(((-(1.82e6*(pow((74.*310.),-1.5))))*4.)*(((sqrt(((0.5*(((140.0+5.0)+150.0)+(4.*1.8)))/1000.)))/(1.+(sqrt(((0.5*(((140.0+5.0)+150.0)+(4.*1.8)))/1000.)))))-((0.3*((0.5*(((140.0+5.0)+150.0)+(4.*1.8)))/1000.)))))))*1.8))))/((torch.exp((2.*((V*96485.)/(8314.*310.)))))-(1.))))*self.d)*((((0.52477*self.ff)+((1.-(0.52477))*self.fs))*(1.0-(self.nca)))+((self.jca*(((0.3+(0.6/(1.+(torch.exp(((V-(9.24247))/27.96201))))))*self.fcaf)+((1.-((0.3+(0.6/(1.+(torch.exp(((V-(9.24247))/27.96201))))))))*self.fcas)))*self.nca))))+(self.camk_f_ICaL*((((1.1*((1.5768e-04)))*(((4.*(((V*96485.)*96485.)/(8314.*310.)))*((((torch.pow(10.,(((-(1.82e6*(pow((74.*310.),-1.5))))*4.)*(((torch.sqrt(((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)))/(1.+(torch.sqrt(((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)))))-((0.3*((0.5*(((self.nasl+self.Ki)+self.Cli)+(4.*self.casl)))/1000.)))))))*self.casl)*(torch.exp((2.*((V*96485.)/(8314.*310.))))))-(((pow(10.,(((-(1.82e6*(pow((74.*310.),-1.5))))*4.)*(((sqrt(((0.5*(((140.0+5.0)+150.0)+(4.*1.8)))/1000.)))/(1.+(sqrt(((0.5*(((140.0+5.0)+150.0)+(4.*1.8)))/1000.)))))-((0.3*((0.5*(((140.0+5.0)+150.0)+(4.*1.8)))/1000.)))))))*1.8))))/((torch.exp((2.*((V*96485.)/(8314.*310.)))))-(1.))))*self.d)*((((0.52477*self.ffp)+((1.-(0.52477))*self.fs))*(1.0-(self.nca)))+((self.jca*(((0.3+(0.6/(1.+(torch.exp(((V-(9.24247))/27.96201))))))*self.fcafp)+((1.-((0.3+(0.6/(1.+(torch.exp(((V-(9.24247))/27.96201))))))))*self.fcas)))*self.nca)))))*self.ical_pureCDI_junc)*0.8)))/0.45),4.5))))))/6.e-4)))/119.48978)
        set_mL_tozero_in_diff_mL = ((1./(1.+(torch.exp(((-(V+42.85))/5.264)))))/((0.1292*(torch.exp((-(((V+45.79)/15.54)*((V+45.79)/15.54))))))+(0.06487*(torch.exp((-(((V-(4.823))/51.12)*((V-(4.823))/51.12))))))))
        set_m_P_tozero_in_diff_m_P = ((1./((1.+(torch.exp(((-(V+61.86))/9.03))))*(1.+(torch.exp(((-(V+61.86))/9.03))))))/((0.1292*(torch.exp((-(((V+45.79)/15.54)*((V+45.79)/15.54))))))+(0.06487*(torch.exp((-(((V-(4.823))/51.12)*((V-(4.823))/51.12))))))))
        set_m_tozero_in_diff_m = ((1./((1.+(torch.exp(((-(V+56.86))/9.03))))*(1.+(torch.exp(((-(V+56.86))/9.03))))))/((0.1292*(torch.exp((-(((V+45.79)/15.54)*((V+45.79)/15.54))))))+(0.06487*(torch.exp((-(((V-(4.823))/51.12)*((V-(4.823))/51.12))))))))
        set_xs_junc_tozero_in_diff_xs_junc = ((1./(1.+(torch.exp(((-(V-((-1.+((-12.-(-1.))/(1.+((((350.e-6/self.caj)*(350.e-6/self.caj))*(350.e-6/self.caj))*(350.e-6/self.caj))))))))/25.)))))/(2.*(50.+((50.+(350.*(torch.exp(((-((V+30.)*(V+30.)))/4000.)))))/(1.+(torch.exp(((-(V+(26.+((40.-(26.))/(1.+(((150.e-6/self.caj)*(150.e-6/self.caj))*(150.e-6/self.caj)))))))/10.))))))))
        set_xs_sl_tozero_in_diff_xs_sl = ((1./(1.+(torch.exp(((-(V-((-1.+((-12.-(-1.))/(1.+((((350.e-6/self.casl)*(350.e-6/self.casl))*(350.e-6/self.casl))*(350.e-6/self.casl))))))))/25.)))))/(2.*(50.+((50.+(350.*(torch.exp(((-((V+30.)*(V+30.)))/4000.)))))/(1.+(torch.exp(((-(V+(26.+((40.-(26.))/(1.+(((150.e-6/self.casl)*(150.e-6/self.casl))*(150.e-6/self.casl)))))))/10.))))))))
        set_xtof_p_tozero_in_diff_xtof_p = ((1./(1.+(torch.exp(((-(V-(29.)))/13.)))))/((8.5*(torch.exp((-(((V+45.)/50.)*((V+45.)/50.))))))+0.5))
        set_xtof_tozero_in_diff_xtof = ((1./(1.+(torch.exp(((-(V-(19.0)))/13.)))))/((8.5*(torch.exp((-(((V+45.)/50.)*((V+45.)/50.))))))+0.5))
        set_xtos_p_tozero_in_diff_xtos_p = ((1./(1.+(torch.exp(((-(V-(29.)))/13.)))))/((9./(1.+(torch.exp(((V+3.0)/15.)))))+0.5))
        set_xtos_tozero_in_diff_xtos = ((1./(1.+(torch.exp(((-(V-(19.0)))/13.)))))/((9./(1.+(torch.exp(((V+3.0)/15.)))))+0.5))
        set_ytof_p_tozero_in_diff_ytof_p = ((1./(1.+(torch.exp(((V+19.5)/5.)))))/((((85.*(torch.exp(((-((V+40.)*(V+40.)))/220.))))+7.)*(1.354+(1.0e-4/((torch.exp(((V-(167.4))/15.89)))+(torch.exp(((-(V-(12.23)))/0.2154)))))))*(1.0-((0.5/(1.0+(torch.exp(((V+70.0)/20.0)))))))))
        set_ytof_tozero_in_diff_ytof = ((1./(1.+(torch.exp(((V+19.5)/5.)))))/((85.*(torch.exp(((-((V+40.)*(V+40.)))/220.))))+7.))
        set_ytos_p_tozero_in_diff_ytos_p = ((1./(1.+(torch.exp(((V+19.5)/5.)))))/((((800./(1.+(torch.exp(((V+60.0)/10.)))))+30.)*(1.354+(1.0e-4/((torch.exp(((V-(167.4))/15.89)))+(torch.exp(((-(V-(12.23)))/0.2154)))))))*(1.0-((0.5/(1.0+(torch.exp(((V+70.0)/20.0)))))))))
        set_ytos_tozero_in_diff_ytos = ((1./(1.+(torch.exp(((V+19.5)/5.)))))/((800./(1.+(torch.exp(((V+60.0)/10.)))))+30.))
        tauxtof = ((8.5*(torch.exp((-(((V+45.)/50.)*((V+45.)/50.))))))+0.5)
        tauxtos = ((9./(1.+(torch.exp(((V+3.0)/15.)))))+0.5)
        tauytof = ((85.*(torch.exp(((-((V+40.)*(V+40.)))/220.))))+7.)
        tauytos = ((800./(1.+(torch.exp(((V+60.0)/10.)))))+30.)
        thLp = (3.*self.thL)
        tm = ((0.1292*(torch.exp((-(((V+45.79)/15.54)*((V+45.79)/15.54))))))+(0.06487*(torch.exp((-(((V-(4.823))/51.12)*((V-(4.823))/51.12)))))))
        tmL = ((0.1292*(torch.exp((-(((V+45.79)/15.54)*((V+45.79)/15.54))))))+(0.06487*(torch.exp((-(((V-(4.823))/51.12)*((V-(4.823))/51.12)))))))
        VsTs_Ca_junc = (P_tau_0+((P_tau_max-(P_tau_0))/(1.+(((150.e-6/self.caj)*(150.e-6/self.caj))*(150.e-6/self.caj)))))
        VsTs_Ca_sl = (P_tau_0+((P_tau_max-(P_tau_0))/(1.+(((150.e-6/self.casl)*(150.e-6/self.casl))*(150.e-6/self.casl)))))
        fcaf_rush_larsen_A = ((set_fcaf_tozero_in_diff_fcaf/partial_diff_fcaf_del_fcaf)*(torch.expm1((self.dt*partial_diff_fcaf_del_fcaf))))
        fcaf_rush_larsen_B = (torch.exp((self.dt*partial_diff_fcaf_del_fcaf)))
        fcafp_rush_larsen_A = ((set_fcafp_tozero_in_diff_fcafp/partial_diff_fcafp_del_fcafp)*(torch.expm1((self.dt*partial_diff_fcafp_del_fcafp))))
        fcafp_rush_larsen_B = (torch.exp((self.dt*partial_diff_fcafp_del_fcafp)))
        fcas_rush_larsen_A = ((set_fcas_tozero_in_diff_fcas/partial_diff_fcas_del_fcas)*(torch.expm1((self.dt*partial_diff_fcas_del_fcas))))
        fcas_rush_larsen_B = (torch.exp((self.dt*partial_diff_fcas_del_fcas)))
        ff_rush_larsen_A = ((set_ff_tozero_in_diff_ff/partial_diff_ff_del_ff)*(torch.expm1((self.dt*partial_diff_ff_del_ff))))
        ff_rush_larsen_B = (torch.exp((self.dt*partial_diff_ff_del_ff)))
        ffp_rush_larsen_A = ((set_ffp_tozero_in_diff_ffp/partial_diff_ffp_del_ffp)*(torch.expm1((self.dt*partial_diff_ffp_del_ffp))))
        ffp_rush_larsen_B = (torch.exp((self.dt*partial_diff_ffp_del_ffp)))
        fs_rush_larsen_A = ((set_fs_tozero_in_diff_fs/partial_diff_fs_del_fs)*(torch.expm1((self.dt*partial_diff_fs_del_fs))))
        fs_rush_larsen_B = (torch.exp((self.dt*partial_diff_fs_del_fs)))
        hL_rush_larsen_A = ((set_hL_tozero_in_diff_hL/partial_diff_hL_del_hL)*(expm1((self.dt*partial_diff_hL_del_hL))))
        hL_rush_larsen_B = (exp((self.dt*partial_diff_hL_del_hL)))
        jca_rush_larsen_A = ((set_jca_tozero_in_diff_jca/partial_diff_jca_del_jca)*(expm1((self.dt*partial_diff_jca_del_jca))))
        jca_rush_larsen_B = (exp((self.dt*partial_diff_jca_del_jca)))
        jrel_icaldep_f1_rush_larsen_A = ((set_jrel_icaldep_f1_tozero_in_diff_jrel_icaldep_f1/partial_diff_jrel_icaldep_f1_del_jrel_icaldep_f1)*(expm1((self.dt*partial_diff_jrel_icaldep_f1_del_jrel_icaldep_f1))))
        jrel_icaldep_f1_rush_larsen_B = (exp((self.dt*partial_diff_jrel_icaldep_f1_del_jrel_icaldep_f1)))
        jrel_icaldep_f2_rush_larsen_A = ((set_jrel_icaldep_f2_tozero_in_diff_jrel_icaldep_f2/partial_diff_jrel_icaldep_f2_del_jrel_icaldep_f2)*(expm1((self.dt*partial_diff_jrel_icaldep_f2_del_jrel_icaldep_f2))))
        jrel_icaldep_f2_rush_larsen_B = (exp((self.dt*partial_diff_jrel_icaldep_f2_del_jrel_icaldep_f2)))
        partial_diff_hLp_del_hLp = ((thLp*-1.)/(thLp*thLp))
        partial_diff_mL_del_mL = ((tmL*-1.)/(tmL*tmL))
        partial_diff_m_P_del_m_P = ((tm*-1.)/(tm*tm))
        partial_diff_m_del_m = ((tm*-1.)/(tm*tm))
        partial_diff_xtof_del_xtof = ((tauxtof*-1.)/(tauxtof*tauxtof))
        partial_diff_xtof_p_del_xtof_p = ((tauxtof*-1.)/(tauxtof*tauxtof))
        partial_diff_xtos_del_xtos = ((tauxtos*-1.)/(tauxtos*tauxtos))
        partial_diff_xtos_p_del_xtos_p = ((tauxtos*-1.)/(tauxtos*tauxtos))
        partial_diff_ytof_del_ytof = ((tauytof*-1.)/(tauytof*tauytof))
        partial_diff_ytos_del_ytos = ((tauytos*-1.)/(tauytos*tauytos))
        tauytof_p = ((tauytof*dti_develop)*dti_recover)
        tauytos_p = ((tauytos*dti_develop)*dti_recover)
        th = (1./(ah+bh))
        tj = (1./(aj+bj))
        hLp_rush_larsen_A = ((set_hLp_tozero_in_diff_hLp/partial_diff_hLp_del_hLp)*(expm1((self.dt*partial_diff_hLp_del_hLp))))
        hLp_rush_larsen_B = (exp((self.dt*partial_diff_hLp_del_hLp)))
        mL_rush_larsen_A = ((set_mL_tozero_in_diff_mL/partial_diff_mL_del_mL)*(torch.expm1((self.dt*partial_diff_mL_del_mL))))
        mL_rush_larsen_B = (torch.exp((self.dt*partial_diff_mL_del_mL)))
        m_P_rush_larsen_A = ((set_m_P_tozero_in_diff_m_P/partial_diff_m_P_del_m_P)*(torch.expm1((self.dt*partial_diff_m_P_del_m_P))))
        m_P_rush_larsen_B = (torch.exp((self.dt*partial_diff_m_P_del_m_P)))
        m_rush_larsen_A = ((set_m_tozero_in_diff_m/partial_diff_m_del_m)*(torch.expm1((self.dt*partial_diff_m_del_m))))
        m_rush_larsen_B = (torch.exp((self.dt*partial_diff_m_del_m)))
        partial_diff_h_P_del_h_P = ((th*-1.)/(th*th))
        partial_diff_h_del_h = ((th*-1.)/(th*th))
        partial_diff_hp_P_del_hp_P = ((th*-1.)/(th*th))
        partial_diff_hp_del_hp = ((th*-1.)/(th*th))
        partial_diff_j_P_del_j_P = ((tj*-1.)/(tj*tj))
        partial_diff_j_del_j = ((tj*-1.)/(tj*tj))
        partial_diff_ytof_p_del_ytof_p = ((tauytof_p*-1.)/(tauytof_p*tauytof_p))
        partial_diff_ytos_p_del_ytos_p = ((tauytos_p*-1.)/(tauytos_p*tauytos_p))
        tauxs_junc = (2.*(50.+((50.+(350.*(torch.exp(((-((V+30.)*(V+30.)))/4000.)))))/(1.+(torch.exp(((-(V+VsTs_Ca_junc))/10.)))))))
        tauxs_sl = (2.*(50.+((50.+(350.*(torch.exp(((-((V+30.)*(V+30.)))/4000.)))))/(1.+(torch.exp(((-(V+VsTs_Ca_sl))/10.)))))))
        tjp = (1.46*tj)
        xtof_p_rush_larsen_A = ((set_xtof_p_tozero_in_diff_xtof_p/partial_diff_xtof_p_del_xtof_p)*(torch.expm1((self.dt*partial_diff_xtof_p_del_xtof_p))))
        xtof_p_rush_larsen_B = (torch.exp((self.dt*partial_diff_xtof_p_del_xtof_p)))
        xtof_rush_larsen_A = ((set_xtof_tozero_in_diff_xtof/partial_diff_xtof_del_xtof)*(torch.expm1((self.dt*partial_diff_xtof_del_xtof))))
        xtof_rush_larsen_B = (torch.exp((self.dt*partial_diff_xtof_del_xtof)))
        xtos_p_rush_larsen_A = ((set_xtos_p_tozero_in_diff_xtos_p/partial_diff_xtos_p_del_xtos_p)*(torch.expm1((self.dt*partial_diff_xtos_p_del_xtos_p))))
        xtos_p_rush_larsen_B = (torch.exp((self.dt*partial_diff_xtos_p_del_xtos_p)))
        xtos_rush_larsen_A = ((set_xtos_tozero_in_diff_xtos/partial_diff_xtos_del_xtos)*(torch.expm1((self.dt*partial_diff_xtos_del_xtos))))
        xtos_rush_larsen_B = (torch.exp((self.dt*partial_diff_xtos_del_xtos)))
        ytof_rush_larsen_A = ((set_ytof_tozero_in_diff_ytof/partial_diff_ytof_del_ytof)*(torch.expm1((self.dt*partial_diff_ytof_del_ytof))))
        ytof_rush_larsen_B = (torch.exp((self.dt*partial_diff_ytof_del_ytof)))
        ytos_rush_larsen_A = ((set_ytos_tozero_in_diff_ytos/partial_diff_ytos_del_ytos)*(torch.expm1((self.dt*partial_diff_ytos_del_ytos))))
        ytos_rush_larsen_B = (torch.exp((self.dt*partial_diff_ytos_del_ytos)))
        h_P_rush_larsen_A = ((set_h_P_tozero_in_diff_h_P/partial_diff_h_P_del_h_P)*(torch.expm1((self.dt*partial_diff_h_P_del_h_P))))
        h_P_rush_larsen_B = (torch.exp((self.dt*partial_diff_h_P_del_h_P)))
        h_rush_larsen_A = ((set_h_tozero_in_diff_h/partial_diff_h_del_h)*(torch.expm1((self.dt*partial_diff_h_del_h))))
        h_rush_larsen_B = (torch.exp((self.dt*partial_diff_h_del_h)))
        hp_P_rush_larsen_A = ((set_hp_P_tozero_in_diff_hp_P/partial_diff_hp_P_del_hp_P)*(torch.expm1((self.dt*partial_diff_hp_P_del_hp_P))))
        hp_P_rush_larsen_B = (torch.exp((self.dt*partial_diff_hp_P_del_hp_P)))
        hp_rush_larsen_A = ((set_hp_tozero_in_diff_hp/partial_diff_hp_del_hp)*(torch.expm1((self.dt*partial_diff_hp_del_hp))))
        hp_rush_larsen_B = (torch.exp((self.dt*partial_diff_hp_del_hp)))
        j_P_rush_larsen_A = ((set_j_P_tozero_in_diff_j_P/partial_diff_j_P_del_j_P)*(torch.expm1((self.dt*partial_diff_j_P_del_j_P))))
        j_P_rush_larsen_B = (torch.exp((self.dt*partial_diff_j_P_del_j_P)))
        j_rush_larsen_A = ((set_j_tozero_in_diff_j/partial_diff_j_del_j)*(torch.expm1((self.dt*partial_diff_j_del_j))))
        j_rush_larsen_B = (torch.exp((self.dt*partial_diff_j_del_j)))
        partial_diff_jp_P_del_jp_P = ((tjp*-1.)/(tjp*tjp))
        partial_diff_jp_del_jp = ((tjp*-1.)/(tjp*tjp))
        partial_diff_xs_junc_del_xs_junc = ((tauxs_junc*-1.)/(tauxs_junc*tauxs_junc))
        partial_diff_xs_sl_del_xs_sl = ((tauxs_sl*-1.)/(tauxs_sl*tauxs_sl))
        ytof_p_rush_larsen_A = ((set_ytof_p_tozero_in_diff_ytof_p/partial_diff_ytof_p_del_ytof_p)*(torch.expm1((self.dt*partial_diff_ytof_p_del_ytof_p))))
        ytof_p_rush_larsen_B = (torch.exp((self.dt*partial_diff_ytof_p_del_ytof_p)))
        ytos_p_rush_larsen_A = ((set_ytos_p_tozero_in_diff_ytos_p/partial_diff_ytos_p_del_ytos_p)*(torch.expm1((self.dt*partial_diff_ytos_p_del_ytos_p))))
        ytos_p_rush_larsen_B = (torch.exp((self.dt*partial_diff_ytos_p_del_ytos_p)))
        jp_P_rush_larsen_A = ((set_jp_P_tozero_in_diff_jp_P/partial_diff_jp_P_del_jp_P)*(torch.expm1((self.dt*partial_diff_jp_P_del_jp_P))))
        jp_P_rush_larsen_B = (torch.exp((self.dt*partial_diff_jp_P_del_jp_P)))
        jp_rush_larsen_A = ((set_jp_tozero_in_diff_jp/partial_diff_jp_del_jp)*(torch.expm1((self.dt*partial_diff_jp_del_jp))))
        jp_rush_larsen_B = (torch.exp((self.dt*partial_diff_jp_del_jp)))
        xs_junc_rush_larsen_A = ((set_xs_junc_tozero_in_diff_xs_junc/partial_diff_xs_junc_del_xs_junc)*(torch.expm1((self.dt*partial_diff_xs_junc_del_xs_junc))))
        xs_junc_rush_larsen_B = (torch.exp((self.dt*partial_diff_xs_junc_del_xs_junc)))
        xs_sl_rush_larsen_A = ((set_xs_sl_tozero_in_diff_xs_sl/partial_diff_xs_sl_del_xs_sl)*(torch.expm1((self.dt*partial_diff_xs_sl_del_xs_sl))))
        xs_sl_rush_larsen_B = (torch.exp((self.dt*partial_diff_xs_sl_del_xs_sl)))
        fcaf_new = fcaf_rush_larsen_A+fcaf_rush_larsen_B*self.fcaf
        fcafp_new = fcafp_rush_larsen_A+fcafp_rush_larsen_B*self.fcafp
        fcas_new = fcas_rush_larsen_A+fcas_rush_larsen_B*self.fcas
        ff_new = ff_rush_larsen_A+ff_rush_larsen_B*self.ff
        ffp_new = ffp_rush_larsen_A+ffp_rush_larsen_B*self.ffp
        fs_new = fs_rush_larsen_A+fs_rush_larsen_B*self.fs
        h_new = h_rush_larsen_A+h_rush_larsen_B*self.h
        hL_new = hL_rush_larsen_A+hL_rush_larsen_B*self.hL
        hLp_new = hLp_rush_larsen_A+hLp_rush_larsen_B*self.hLp
        h_P_new = h_P_rush_larsen_A+h_P_rush_larsen_B*self.h_P
        hp_new = hp_rush_larsen_A+hp_rush_larsen_B*self.hp
        hp_P_new = hp_P_rush_larsen_A+hp_P_rush_larsen_B*self.hp_P
        j_new = j_rush_larsen_A+j_rush_larsen_B*self.j
        j_P_new = j_P_rush_larsen_A+j_P_rush_larsen_B*self.j_P
        jca_new = jca_rush_larsen_A+jca_rush_larsen_B*self.jca
        jp_new = jp_rush_larsen_A+jp_rush_larsen_B*self.jp
        jp_P_new = jp_P_rush_larsen_A+jp_P_rush_larsen_B*self.jp_P
        jrel_icaldep_f1_new = jrel_icaldep_f1_rush_larsen_A+jrel_icaldep_f1_rush_larsen_B*self.jrel_icaldep_f1
        jrel_icaldep_f2_new = jrel_icaldep_f2_rush_larsen_A+jrel_icaldep_f2_rush_larsen_B*self.jrel_icaldep_f2
        m_new = m_rush_larsen_A+m_rush_larsen_B*self.m
        mL_new = mL_rush_larsen_A+mL_rush_larsen_B*self.mL
        m_P_new = m_P_rush_larsen_A+m_P_rush_larsen_B*self.m_P
        xs_junc_new = xs_junc_rush_larsen_A+xs_junc_rush_larsen_B*self.xs_junc
        xs_sl_new = xs_sl_rush_larsen_A+xs_sl_rush_larsen_B*self.xs_sl
        xtof_new = xtof_rush_larsen_A+xtof_rush_larsen_B*self.xtof
        xtof_p_new = xtof_p_rush_larsen_A+xtof_p_rush_larsen_B*self.xtof_p
        xtos_new = xtos_rush_larsen_A+xtos_rush_larsen_B*self.xtos
        xtos_p_new = xtos_p_rush_larsen_A+xtos_p_rush_larsen_B*self.xtos_p
        ytof_new = ytof_rush_larsen_A+ytof_rush_larsen_B*self.ytof
        ytof_p_new = ytof_p_rush_larsen_A+ytof_p_rush_larsen_B*self.ytof_p
        ytos_new = ytos_rush_larsen_A+ytos_rush_larsen_B*self.ytos
        ytos_p_new = ytos_p_rush_larsen_A+ytos_p_rush_larsen_B*self.ytos_p

        # Complete RK4 Update
        Bmax_Csqn = ((1.3655214e-1*vmyo)/vsr)
        RIcleft = ((((1.-(self.ryr_R))-(self.ryr_O))-(self.ryr_I))-(self.ryr_CaRI))
        RIcleft_p = ((((1.-(self.ryr_R_p))-(self.ryr_O_p))-(self.ryr_I_p))-(self.ryr_CaRI_p))
        alphaIKr = (0.1161*(torch.exp((0.2990*vfrt))))
        alpha_2 = (0.0578*(torch.exp((0.9710*vfrt))))
        alpha_C2ToI = (0.52e-4*(torch.exp((1.525*vfrt))))
        alpha_i = (0.2533*(torch.exp((0.5953*vfrt))))
        betaIKr = (0.2442*(torch.exp((-1.604*vfrt))))
        beta_2 = (0.349e-3*(torch.exp((-1.062*vfrt))))
        beta_i = (0.04568*(torch.exp((-0.8209*vfrt))))
        caTransFactor2p = (self.caTransFactor2*1.5)
        kCaSR = (self.MaxSR-(((self.MaxSR-(self.MinSR))/(1.+(torch.pow((self.ec50SR/self.casr),self.steepnessCaSR))))))
        sigmoidBaseCaI = (self.minCaI+((self.maxCaI-(self.minCaI))/(1.+(torch.pow((self.ecCaI/self.caj),self.steepnessCaI)))))
        RI_to_CI = (self.baseRateCaI*sigmoidBaseCaI)
        beta_ItoC2 = (((beta_2*beta_i)*alpha_C2ToI)/(alpha_2*alpha_i))
        d_buffers_Csqn = (((self.kon_csqn*self.casr)*(Bmax_Csqn-(self.buffers_Csqn)))-((self.koff_csqn*self.buffers_Csqn)))
        diff_C0 = ((betaIKr*self.C1)-((alphaIKr*self.C0)))
        diff_C1 = (((alphaIKr*self.C0)+(self.beta_1*self.C2))-(((betaIKr+self.alpha_1)*self.C1)))
        diff_O = (((alpha_2*self.C2)+(beta_i*self.I))-(((beta_2+alpha_i)*self.O)))
        kiSRCa = (self.kiCa*kCaSR)
        koSRCa = (self.koCa/kCaSR)
        diff_C2 = ((((self.alpha_1*self.C1)+(beta_2*self.O))+(beta_ItoC2*self.I))-((((self.beta_1+alpha_2)+alpha_C2ToI)*self.C2)))
        diff_I = (((alpha_C2ToI*self.C2)+(alpha_i*self.O))-(((beta_ItoC2+beta_i)*self.I)))
        diff_buffers_Csqn = d_buffers_Csqn
        diff_casr = ((Jserca-((((J_SRleak*vmyo)/vsr)+J_SRCarel)))-(d_buffers_Csqn))
        diff_ryr_CaRI = ((RI_to_CI*RIcleft)-((self.CI_to_RI*self.ryr_CaRI)))
        diff_ryr_CaRI_p = ((RI_to_CI*RIcleft_p)-((self.CI_to_RI*self.ryr_CaRI_p)))
        diff_ryr_I = (((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*self.ryr_O)-((self.kim*self.ryr_I)))-(((self.kom*self.ryr_I)-((((self.caTransFactor2*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*RIcleft)))))
        diff_ryr_I_p = (((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*self.ryr_O_p)-((self.kim*self.ryr_I_p)))-(((self.kom*self.ryr_I_p)-((((caTransFactor2p*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*RIcleft_p)))))
        diff_ryr_O = (((((self.caTransFactor2*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*self.ryr_R)-((self.kom*self.ryr_O)))-(((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*self.ryr_O)-((self.kim*self.ryr_I)))))
        diff_ryr_O_p = (((((caTransFactor2p*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*self.ryr_R_p)-((self.kom*self.ryr_O_p)))-(((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*self.ryr_O_p)-((self.kim*self.ryr_I_p)))))
        diff_ryr_R = (((self.kim*RIcleft)-((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*self.ryr_R)))-(((((self.caTransFactor2*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*self.ryr_R)-((self.kom*self.ryr_O)))))
        diff_ryr_R_p = (((self.kim*RIcleft_p)-((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*self.ryr_R_p)))-(((((caTransFactor2p*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*self.ryr_R_p)-((self.kom*self.ryr_O_p)))))
        rk4_k1_C0 = diff_C0*self.dt
        rk4_k1_C1 = diff_C1*self.dt
        rk4_k1_C2 = diff_C2*self.dt
        rk4_k1_I = diff_I*self.dt
        rk4_k1_O = diff_O*self.dt
        rk4_k1_buffers_Csqn = diff_buffers_Csqn*self.dt
        rk4_k1_casr = diff_casr*self.dt
        rk4_k1_ryr_CaRI = diff_ryr_CaRI*self.dt
        rk4_k1_ryr_CaRI_p = diff_ryr_CaRI_p*self.dt
        rk4_k1_ryr_I = diff_ryr_I*self.dt
        rk4_k1_ryr_I_p = diff_ryr_I_p*self.dt
        rk4_k1_ryr_O = diff_ryr_O*self.dt
        rk4_k1_ryr_O_p = diff_ryr_O_p*self.dt
        rk4_k1_ryr_R = diff_ryr_R*self.dt
        rk4_k1_ryr_R_p = diff_ryr_R_p*self.dt

        # RK4 stage 2
        sv_intermediate_C0 = self.C0+rk4_k1_C0/2
        sv_intermediate_C1 = self.C1+rk4_k1_C1/2
        sv_intermediate_C2 = self.C2+rk4_k1_C2/2
        sv_intermediate_I = self.I+rk4_k1_I/2
        sv_intermediate_O = self.O+rk4_k1_O/2
        sv_intermediate_buffers_Csqn = self.buffers_Csqn+rk4_k1_buffers_Csqn/2
        sv_intermediate_casr = self.casr+rk4_k1_casr/2
        sv_intermediate_ryr_CaRI = self.ryr_CaRI+rk4_k1_ryr_CaRI/2
        sv_intermediate_ryr_CaRI_p = self.ryr_CaRI_p+rk4_k1_ryr_CaRI_p/2
        sv_intermediate_ryr_I = self.ryr_I+rk4_k1_ryr_I/2
        sv_intermediate_ryr_I_p = self.ryr_I_p+rk4_k1_ryr_I_p/2
        sv_intermediate_ryr_O = self.ryr_O+rk4_k1_ryr_O/2
        sv_intermediate_ryr_O_p = self.ryr_O_p+rk4_k1_ryr_O_p/2
        sv_intermediate_ryr_R = self.ryr_R+rk4_k1_ryr_R/2
        sv_intermediate_ryr_R_p = self.ryr_R_p+rk4_k1_ryr_R_p/2
        J_SRCarel_np = (((ks*sv_intermediate_ryr_O)*(sv_intermediate_casr-(self.caj)))+Jrel_ICaLdep)
        J_SRCarel_p = (((ks*sv_intermediate_ryr_O_p)*(sv_intermediate_casr-(self.caj)))+Jrel_ICaLdep)
        Jserca_np = (((((pow(self.Q10SRCaP,Qpow))*Vmax_SRCaP)*Vmax_mult)*((torch.pow((self.Cai/self.Kmf),self.hillSRCaP))-((torch.pow((sv_intermediate_casr/self.Kmr),self.hillSRCaP)))))/((1.+(torch.pow((self.Cai/self.Kmf),self.hillSRCaP)))+(torch.pow((sv_intermediate_casr/self.Kmr),self.hillSRCaP))))
        Jserca_p = (((((pow(self.Q10SRCaP,Qpow))*Vmax_SRCaP)*Vmax_mult)*((torch.pow((self.Cai/Kmf_p),self.hillSRCaP))-((torch.pow((sv_intermediate_casr/self.Kmr),self.hillSRCaP)))))/((1.+(torch.pow((self.Cai/Kmf_p),self.hillSRCaP)))+(torch.pow((sv_intermediate_casr/self.Kmr),self.hillSRCaP))))
        RIcleft = ((((1.-(sv_intermediate_ryr_R))-(sv_intermediate_ryr_O))-(sv_intermediate_ryr_I))-(sv_intermediate_ryr_CaRI))
        RIcleft_p = ((((1.-(sv_intermediate_ryr_R_p))-(sv_intermediate_ryr_O_p))-(sv_intermediate_ryr_I_p))-(sv_intermediate_ryr_CaRI_p))
        d_buffers_Csqn = (((self.kon_csqn*sv_intermediate_casr)*(Bmax_Csqn-(sv_intermediate_buffers_Csqn)))-((self.koff_csqn*sv_intermediate_buffers_Csqn)))
        diff_C0 = ((betaIKr*sv_intermediate_C1)-((alphaIKr*sv_intermediate_C0)))
        diff_C1 = (((alphaIKr*sv_intermediate_C0)+(self.beta_1*sv_intermediate_C2))-(((betaIKr+self.alpha_1)*sv_intermediate_C1)))
        diff_C2 = ((((self.alpha_1*sv_intermediate_C1)+(beta_2*sv_intermediate_O))+(beta_ItoC2*sv_intermediate_I))-((((self.beta_1+alpha_2)+alpha_C2ToI)*sv_intermediate_C2)))
        diff_I = (((alpha_C2ToI*sv_intermediate_C2)+(alpha_i*sv_intermediate_O))-(((beta_ItoC2+beta_i)*sv_intermediate_I)))
        diff_O = (((alpha_2*sv_intermediate_C2)+(beta_i*sv_intermediate_I))-(((beta_2+alpha_i)*sv_intermediate_O)))
        kCaSR = (self.MaxSR-(((self.MaxSR-(self.MinSR))/(1.+(torch.pow((self.ec50SR/sv_intermediate_casr),self.steepnessCaSR))))))
        nonlinearModifier = (0.2144*(torch.exp((1.83*sv_intermediate_casr))))
        J_SRCarel = ((J_SRCarel_p*self.camk_f_RyR)+(J_SRCarel_np*(1.-(self.camk_f_RyR))))
        J_SRleak = (((1.59306e-6*(sv_intermediate_casr-(self.caj)))*nonlinearModifier)*CaMKIILeakMultiplier)
        Jserca = ((Jserca_np*(1.-(phosphorylationTotal)))+(Jserca_p*phosphorylationTotal))
        diff_buffers_Csqn = d_buffers_Csqn
        diff_ryr_CaRI = ((RI_to_CI*RIcleft)-((self.CI_to_RI*sv_intermediate_ryr_CaRI)))
        diff_ryr_CaRI_p = ((RI_to_CI*RIcleft_p)-((self.CI_to_RI*sv_intermediate_ryr_CaRI_p)))
        kiSRCa = (self.kiCa*kCaSR)
        koSRCa = (self.koCa/kCaSR)
        diff_casr = ((Jserca-((((J_SRleak*vmyo)/vsr)+J_SRCarel)))-(d_buffers_Csqn))
        diff_ryr_I = (((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_O)-((self.kim*sv_intermediate_ryr_I)))-(((self.kom*sv_intermediate_ryr_I)-((((self.caTransFactor2*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*RIcleft)))))
        diff_ryr_I_p = (((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_O_p)-((self.kim*sv_intermediate_ryr_I_p)))-(((self.kom*sv_intermediate_ryr_I_p)-((((caTransFactor2p*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*RIcleft_p)))))
        diff_ryr_O = (((((self.caTransFactor2*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*sv_intermediate_ryr_R)-((self.kom*sv_intermediate_ryr_O)))-(((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_O)-((self.kim*sv_intermediate_ryr_I)))))
        diff_ryr_O_p = (((((caTransFactor2p*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*sv_intermediate_ryr_R_p)-((self.kom*sv_intermediate_ryr_O_p)))-(((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_O_p)-((self.kim*sv_intermediate_ryr_I_p)))))
        diff_ryr_R = (((self.kim*RIcleft)-((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_R)))-(((((self.caTransFactor2*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*sv_intermediate_ryr_R)-((self.kom*sv_intermediate_ryr_O)))))
        diff_ryr_R_p = (((self.kim*RIcleft_p)-((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_R_p)))-(((((caTransFactor2p*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*sv_intermediate_ryr_R_p)-((self.kom*sv_intermediate_ryr_O_p)))))
        rk4_k2_C0 = self.dt*diff_C0
        rk4_k2_C1 = self.dt*diff_C1
        rk4_k2_C2 = self.dt*diff_C2
        rk4_k2_I = self.dt*diff_I
        rk4_k2_O = self.dt*diff_O
        rk4_k2_buffers_Csqn = self.dt*diff_buffers_Csqn
        rk4_k2_casr = self.dt*diff_casr
        rk4_k2_ryr_CaRI = self.dt*diff_ryr_CaRI
        rk4_k2_ryr_CaRI_p = self.dt*diff_ryr_CaRI_p
        rk4_k2_ryr_I = self.dt*diff_ryr_I
        rk4_k2_ryr_I_p = self.dt*diff_ryr_I_p
        rk4_k2_ryr_O = self.dt*diff_ryr_O
        rk4_k2_ryr_O_p = self.dt*diff_ryr_O_p
        rk4_k2_ryr_R = self.dt*diff_ryr_R
        rk4_k2_ryr_R_p = self.dt*diff_ryr_R_p

        # RK4 stage 3
        sv_intermediate_C0 = self.C0+rk4_k2_C0/2
        sv_intermediate_C1 = self.C1+rk4_k2_C1/2
        sv_intermediate_C2 = self.C2+rk4_k2_C2/2
        sv_intermediate_I = self.I+rk4_k2_I/2
        sv_intermediate_O = self.O+rk4_k2_O/2
        sv_intermediate_buffers_Csqn = self.buffers_Csqn+rk4_k2_buffers_Csqn/2
        sv_intermediate_casr = self.casr+rk4_k2_casr/2
        sv_intermediate_ryr_CaRI = self.ryr_CaRI+rk4_k2_ryr_CaRI/2
        sv_intermediate_ryr_CaRI_p = self.ryr_CaRI_p+rk4_k2_ryr_CaRI_p/2
        sv_intermediate_ryr_I = self.ryr_I+rk4_k2_ryr_I/2
        sv_intermediate_ryr_I_p = self.ryr_I_p+rk4_k2_ryr_I_p/2
        sv_intermediate_ryr_O = self.ryr_O+rk4_k2_ryr_O/2
        sv_intermediate_ryr_O_p = self.ryr_O_p+rk4_k2_ryr_O_p/2
        sv_intermediate_ryr_R = self.ryr_R+rk4_k2_ryr_R/2
        sv_intermediate_ryr_R_p = self.ryr_R_p+rk4_k2_ryr_R_p/2
        J_SRCarel_np = (((ks*sv_intermediate_ryr_O)*(sv_intermediate_casr-(self.caj)))+Jrel_ICaLdep)
        J_SRCarel_p = (((ks*sv_intermediate_ryr_O_p)*(sv_intermediate_casr-(self.caj)))+Jrel_ICaLdep)
        Jserca_np = (((((pow(self.Q10SRCaP,Qpow))*Vmax_SRCaP)*Vmax_mult)*((torch.pow((self.Cai/self.Kmf),self.hillSRCaP))-((torch.pow((sv_intermediate_casr/self.Kmr),self.hillSRCaP)))))/((1.+(torch.pow((self.Cai/self.Kmf),self.hillSRCaP)))+(torch.pow((sv_intermediate_casr/self.Kmr),self.hillSRCaP))))
        Jserca_p = (((((pow(self.Q10SRCaP,Qpow))*Vmax_SRCaP)*Vmax_mult)*((torch.pow((self.Cai/Kmf_p),self.hillSRCaP))-((torch.pow((sv_intermediate_casr/self.Kmr),self.hillSRCaP)))))/((1.+(torch.pow((self.Cai/Kmf_p),self.hillSRCaP)))+(torch.pow((sv_intermediate_casr/self.Kmr),self.hillSRCaP))))
        RIcleft = ((((1.-(sv_intermediate_ryr_R))-(sv_intermediate_ryr_O))-(sv_intermediate_ryr_I))-(sv_intermediate_ryr_CaRI))
        RIcleft_p = ((((1.-(sv_intermediate_ryr_R_p))-(sv_intermediate_ryr_O_p))-(sv_intermediate_ryr_I_p))-(sv_intermediate_ryr_CaRI_p))
        d_buffers_Csqn = (((self.kon_csqn*sv_intermediate_casr)*(Bmax_Csqn-(sv_intermediate_buffers_Csqn)))-((self.koff_csqn*sv_intermediate_buffers_Csqn)))
        diff_C0 = ((betaIKr*sv_intermediate_C1)-((alphaIKr*sv_intermediate_C0)))
        diff_C1 = (((alphaIKr*sv_intermediate_C0)+(self.beta_1*sv_intermediate_C2))-(((betaIKr+self.alpha_1)*sv_intermediate_C1)))
        diff_C2 = ((((self.alpha_1*sv_intermediate_C1)+(beta_2*sv_intermediate_O))+(beta_ItoC2*sv_intermediate_I))-((((self.beta_1+alpha_2)+alpha_C2ToI)*sv_intermediate_C2)))
        diff_I = (((alpha_C2ToI*sv_intermediate_C2)+(alpha_i*sv_intermediate_O))-(((beta_ItoC2+beta_i)*sv_intermediate_I)))
        diff_O = (((alpha_2*sv_intermediate_C2)+(beta_i*sv_intermediate_I))-(((beta_2+alpha_i)*sv_intermediate_O)))
        kCaSR = (self.MaxSR-(((self.MaxSR-(self.MinSR))/(1.+(torch.pow((self.ec50SR/sv_intermediate_casr),self.steepnessCaSR))))))
        nonlinearModifier = (0.2144*(torch.exp((1.83*sv_intermediate_casr))))
        J_SRCarel = ((J_SRCarel_p*self.camk_f_RyR)+(J_SRCarel_np*(1.-(self.camk_f_RyR))))
        J_SRleak = (((1.59306e-6*(sv_intermediate_casr-(self.caj)))*nonlinearModifier)*CaMKIILeakMultiplier)
        Jserca = ((Jserca_np*(1.-(phosphorylationTotal)))+(Jserca_p*phosphorylationTotal))
        diff_buffers_Csqn = d_buffers_Csqn
        diff_ryr_CaRI = ((RI_to_CI*RIcleft)-((self.CI_to_RI*sv_intermediate_ryr_CaRI)))
        diff_ryr_CaRI_p = ((RI_to_CI*RIcleft_p)-((self.CI_to_RI*sv_intermediate_ryr_CaRI_p)))
        kiSRCa = (self.kiCa*kCaSR)
        koSRCa = (self.koCa/kCaSR)
        diff_casr = ((Jserca-((((J_SRleak*vmyo)/vsr)+J_SRCarel)))-(d_buffers_Csqn))
        diff_ryr_I = (((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_O)-((self.kim*sv_intermediate_ryr_I)))-(((self.kom*sv_intermediate_ryr_I)-((((self.caTransFactor2*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*RIcleft)))))
        diff_ryr_I_p = (((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_O_p)-((self.kim*sv_intermediate_ryr_I_p)))-(((self.kom*sv_intermediate_ryr_I_p)-((((caTransFactor2p*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*RIcleft_p)))))
        diff_ryr_O = (((((self.caTransFactor2*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*sv_intermediate_ryr_R)-((self.kom*sv_intermediate_ryr_O)))-(((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_O)-((self.kim*sv_intermediate_ryr_I)))))
        diff_ryr_O_p = (((((caTransFactor2p*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*sv_intermediate_ryr_R_p)-((self.kom*sv_intermediate_ryr_O_p)))-(((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_O_p)-((self.kim*sv_intermediate_ryr_I_p)))))
        diff_ryr_R = (((self.kim*RIcleft)-((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_R)))-(((((self.caTransFactor2*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*sv_intermediate_ryr_R)-((self.kom*sv_intermediate_ryr_O)))))
        diff_ryr_R_p = (((self.kim*RIcleft_p)-((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_R_p)))-(((((caTransFactor2p*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*sv_intermediate_ryr_R_p)-((self.kom*sv_intermediate_ryr_O_p)))))
        rk4_k3_C0 = self.dt*diff_C0
        rk4_k3_C1 = self.dt*diff_C1
        rk4_k3_C2 = self.dt*diff_C2
        rk4_k3_I = self.dt*diff_I
        rk4_k3_O = self.dt*diff_O
        rk4_k3_buffers_Csqn = self.dt*diff_buffers_Csqn
        rk4_k3_casr = self.dt*diff_casr
        rk4_k3_ryr_CaRI = self.dt*diff_ryr_CaRI
        rk4_k3_ryr_CaRI_p = self.dt*diff_ryr_CaRI_p
        rk4_k3_ryr_I = self.dt*diff_ryr_I
        rk4_k3_ryr_I_p = self.dt*diff_ryr_I_p
        rk4_k3_ryr_O = self.dt*diff_ryr_O
        rk4_k3_ryr_O_p = self.dt*diff_ryr_O_p
        rk4_k3_ryr_R = self.dt*diff_ryr_R
        rk4_k3_ryr_R_p = self.dt*diff_ryr_R_p

        # RK4 stage 4
        sv_intermediate_C0 = self.C0+rk4_k3_C0
        sv_intermediate_C1 = self.C1+rk4_k3_C1
        sv_intermediate_C2 = self.C2+rk4_k3_C2
        sv_intermediate_I = self.I+rk4_k3_I
        sv_intermediate_O = self.O+rk4_k3_O
        sv_intermediate_buffers_Csqn = self.buffers_Csqn+rk4_k3_buffers_Csqn
        sv_intermediate_casr = self.casr+rk4_k3_casr
        sv_intermediate_ryr_CaRI = self.ryr_CaRI+rk4_k3_ryr_CaRI
        sv_intermediate_ryr_CaRI_p = self.ryr_CaRI_p+rk4_k3_ryr_CaRI_p
        sv_intermediate_ryr_I = self.ryr_I+rk4_k3_ryr_I
        sv_intermediate_ryr_I_p = self.ryr_I_p+rk4_k3_ryr_I_p
        sv_intermediate_ryr_O = self.ryr_O+rk4_k3_ryr_O
        sv_intermediate_ryr_O_p = self.ryr_O_p+rk4_k3_ryr_O_p
        sv_intermediate_ryr_R = self.ryr_R+rk4_k3_ryr_R
        sv_intermediate_ryr_R_p = self.ryr_R_p+rk4_k3_ryr_R_p
        J_SRCarel_np = (((ks*sv_intermediate_ryr_O)*(sv_intermediate_casr-(self.caj)))+Jrel_ICaLdep)
        J_SRCarel_p = (((ks*sv_intermediate_ryr_O_p)*(sv_intermediate_casr-(self.caj)))+Jrel_ICaLdep)
        Jserca_np = (((((pow(self.Q10SRCaP,Qpow))*Vmax_SRCaP)*Vmax_mult)*((torch.pow((self.Cai/self.Kmf),self.hillSRCaP))-((torch.pow((sv_intermediate_casr/self.Kmr),self.hillSRCaP)))))/((1.+(torch.pow((self.Cai/self.Kmf),self.hillSRCaP)))+(torch.pow((sv_intermediate_casr/self.Kmr),self.hillSRCaP))))
        Jserca_p = (((((pow(self.Q10SRCaP,Qpow))*Vmax_SRCaP)*Vmax_mult)*((torch.pow((self.Cai/Kmf_p),self.hillSRCaP))-((torch.pow((sv_intermediate_casr/self.Kmr),self.hillSRCaP)))))/((1.+(torch.pow((self.Cai/Kmf_p),self.hillSRCaP)))+(torch.pow((sv_intermediate_casr/self.Kmr),self.hillSRCaP))))
        RIcleft = ((((1.-(sv_intermediate_ryr_R))-(sv_intermediate_ryr_O))-(sv_intermediate_ryr_I))-(sv_intermediate_ryr_CaRI))
        RIcleft_p = ((((1.-(sv_intermediate_ryr_R_p))-(sv_intermediate_ryr_O_p))-(sv_intermediate_ryr_I_p))-(sv_intermediate_ryr_CaRI_p))
        d_buffers_Csqn = (((self.kon_csqn*sv_intermediate_casr)*(Bmax_Csqn-(sv_intermediate_buffers_Csqn)))-((self.koff_csqn*sv_intermediate_buffers_Csqn)))
        diff_C0 = ((betaIKr*sv_intermediate_C1)-((alphaIKr*sv_intermediate_C0)))
        diff_C1 = (((alphaIKr*sv_intermediate_C0)+(self.beta_1*sv_intermediate_C2))-(((betaIKr+self.alpha_1)*sv_intermediate_C1)))
        diff_C2 = ((((self.alpha_1*sv_intermediate_C1)+(beta_2*sv_intermediate_O))+(beta_ItoC2*sv_intermediate_I))-((((self.beta_1+alpha_2)+alpha_C2ToI)*sv_intermediate_C2)))
        diff_I = (((alpha_C2ToI*sv_intermediate_C2)+(alpha_i*sv_intermediate_O))-(((beta_ItoC2+beta_i)*sv_intermediate_I)))
        diff_O = (((alpha_2*sv_intermediate_C2)+(beta_i*sv_intermediate_I))-(((beta_2+alpha_i)*sv_intermediate_O)))
        kCaSR = (self.MaxSR-(((self.MaxSR-(self.MinSR))/(1.+(torch.pow((self.ec50SR/sv_intermediate_casr),self.steepnessCaSR))))))
        nonlinearModifier = (0.2144*(torch.exp((1.83*sv_intermediate_casr))))
        J_SRCarel = ((J_SRCarel_p*self.camk_f_RyR)+(J_SRCarel_np*(1.-(self.camk_f_RyR))))
        J_SRleak = (((1.59306e-6*(sv_intermediate_casr-(self.caj)))*nonlinearModifier)*CaMKIILeakMultiplier)
        Jserca = ((Jserca_np*(1.-(phosphorylationTotal)))+(Jserca_p*phosphorylationTotal))
        diff_buffers_Csqn = d_buffers_Csqn
        diff_ryr_CaRI = ((RI_to_CI*RIcleft)-((self.CI_to_RI*sv_intermediate_ryr_CaRI)))
        diff_ryr_CaRI_p = ((RI_to_CI*RIcleft_p)-((self.CI_to_RI*sv_intermediate_ryr_CaRI_p)))
        kiSRCa = (self.kiCa*kCaSR)
        koSRCa = (self.koCa/kCaSR)
        diff_casr = ((Jserca-((((J_SRleak*vmyo)/vsr)+J_SRCarel)))-(d_buffers_Csqn))
        diff_ryr_I = (((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_O)-((self.kim*sv_intermediate_ryr_I)))-(((self.kom*sv_intermediate_ryr_I)-((((self.caTransFactor2*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*RIcleft)))))
        diff_ryr_I_p = (((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_O_p)-((self.kim*sv_intermediate_ryr_I_p)))-(((self.kom*sv_intermediate_ryr_I_p)-((((caTransFactor2p*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*RIcleft_p)))))
        diff_ryr_O = (((((self.caTransFactor2*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*sv_intermediate_ryr_R)-((self.kom*sv_intermediate_ryr_O)))-(((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_O)-((self.kim*sv_intermediate_ryr_I)))))
        diff_ryr_O_p = (((((caTransFactor2p*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*sv_intermediate_ryr_R_p)-((self.kom*sv_intermediate_ryr_O_p)))-(((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_O_p)-((self.kim*sv_intermediate_ryr_I_p)))))
        diff_ryr_R = (((self.kim*RIcleft)-((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_R)))-(((((self.caTransFactor2*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*sv_intermediate_ryr_R)-((self.kom*sv_intermediate_ryr_O)))))
        diff_ryr_R_p = (((self.kim*RIcleft_p)-((((kiSRCa*self.caTransFactor)*(torch.pow(self.caj,self.caExpFactor)))*sv_intermediate_ryr_R_p)))-(((((caTransFactor2p*koSRCa)*(torch.pow(self.caj,self.caExpFactor2)))*sv_intermediate_ryr_R_p)-((self.kom*sv_intermediate_ryr_O_p)))))
        rk4_k4_C0 = self.dt*diff_C0
        rk4_k4_C1 = self.dt*diff_C1
        rk4_k4_C2 = self.dt*diff_C2
        rk4_k4_I = self.dt*diff_I
        rk4_k4_O = self.dt*diff_O
        rk4_k4_buffers_Csqn = self.dt*diff_buffers_Csqn
        rk4_k4_casr = self.dt*diff_casr
        rk4_k4_ryr_CaRI = self.dt*diff_ryr_CaRI
        rk4_k4_ryr_CaRI_p = self.dt*diff_ryr_CaRI_p
        rk4_k4_ryr_I = self.dt*diff_ryr_I
        rk4_k4_ryr_I_p = self.dt*diff_ryr_I_p
        rk4_k4_ryr_O = self.dt*diff_ryr_O
        rk4_k4_ryr_O_p = self.dt*diff_ryr_O_p
        rk4_k4_ryr_R = self.dt*diff_ryr_R
        rk4_k4_ryr_R_p = self.dt*diff_ryr_R_p
        C0_new = self.C0+(rk4_k1_C0+2*rk4_k2_C0+2*rk4_k3_C0+rk4_k4_C0)/6
        C1_new = self.C1+(rk4_k1_C1+2*rk4_k2_C1+2*rk4_k3_C1+rk4_k4_C1)/6
        C2_new = self.C2+(rk4_k1_C2+2*rk4_k2_C2+2*rk4_k3_C2+rk4_k4_C2)/6
        I_new = self.I+(rk4_k1_I+2*rk4_k2_I+2*rk4_k3_I+rk4_k4_I)/6
        O_new = self.O+(rk4_k1_O+2*rk4_k2_O+2*rk4_k3_O+rk4_k4_O)/6
        buffers_Csqn_new = self.buffers_Csqn+(rk4_k1_buffers_Csqn+2*rk4_k2_buffers_Csqn+2*rk4_k3_buffers_Csqn+rk4_k4_buffers_Csqn)/6
        casr_new = self.casr+(rk4_k1_casr+2*rk4_k2_casr+2*rk4_k3_casr+rk4_k4_casr)/6
        ryr_CaRI_new = self.ryr_CaRI+(rk4_k1_ryr_CaRI+2*rk4_k2_ryr_CaRI+2*rk4_k3_ryr_CaRI+rk4_k4_ryr_CaRI)/6
        ryr_CaRI_p_new = self.ryr_CaRI_p+(rk4_k1_ryr_CaRI_p+2*rk4_k2_ryr_CaRI_p+2*rk4_k3_ryr_CaRI_p+rk4_k4_ryr_CaRI_p)/6
        ryr_I_new = self.ryr_I+(rk4_k1_ryr_I+2*rk4_k2_ryr_I+2*rk4_k3_ryr_I+rk4_k4_ryr_I)/6
        ryr_I_p_new = self.ryr_I_p+(rk4_k1_ryr_I_p+2*rk4_k2_ryr_I_p+2*rk4_k3_ryr_I_p+rk4_k4_ryr_I_p)/6
        ryr_O_new = self.ryr_O+(rk4_k1_ryr_O+2*rk4_k2_ryr_O+2*rk4_k3_ryr_O+rk4_k4_ryr_O)/6
        ryr_O_p_new = self.ryr_O_p+(rk4_k1_ryr_O_p+2*rk4_k2_ryr_O_p+2*rk4_k3_ryr_O_p+rk4_k4_ryr_O_p)/6
        ryr_R_new = self.ryr_R+(rk4_k1_ryr_R+2*rk4_k2_ryr_R+2*rk4_k3_ryr_R+rk4_k4_ryr_R)/6
        ryr_R_p_new = self.ryr_R_p+(rk4_k1_ryr_R_p+2*rk4_k2_ryr_R_p+2*rk4_k3_ryr_R_p+rk4_k4_ryr_R_p)/6

        # Finish the update
        self.C0 = C0_new
        self.C1 = C1_new
        self.C2 = C2_new
        self.Cai = Cai_new
        self.Cli = Cli_new
        self.I = I_new
        self.Ki = Ki_new
        self.Nai = Nai_new
        self.O = O_new
        self.buffers_CaM = buffers_CaM_new
        self.buffers_Csqn = buffers_Csqn_new
        self.buffers_Myosin_ca = buffers_Myosin_ca_new
        self.buffers_Myosin_mg = buffers_Myosin_mg_new
        self.buffers_NaBj = buffers_NaBj_new
        self.buffers_NaBsl = buffers_NaBsl_new
        self.buffers_SLHj = buffers_SLHj_new
        self.buffers_SLHsl = buffers_SLHsl_new
        self.buffers_SLLj = buffers_SLLj_new
        self.buffers_SLLsl = buffers_SLLsl_new
        self.buffers_SRB = buffers_SRB_new
        self.buffers_TnCHc = buffers_TnCHc_new
        self.buffers_TnCHm = buffers_TnCHm_new
        self.buffers_TnClow = buffers_TnClow_new
        self.caj = caj_new
        self.camk_f_ICaL = camk_f_ICaL_new
        self.camk_f_PLB = camk_f_PLB_new
        self.camk_f_RyR = camk_f_RyR_new
        self.camk_trap = camk_trap_new
        self.casig_serca_trap = casig_serca_trap_new
        self.casl = casl_new
        self.casr = casr_new
        self.contraction_Ca_TRPN = contraction_Ca_TRPN_new
        self.contraction_TmBlocked = contraction_TmBlocked_new
        self.contraction_XS = contraction_XS_new
        self.contraction_XW = contraction_XW_new
        self.contraction_ZETAS = contraction_ZETAS_new
        self.contraction_ZETAW = contraction_ZETAW_new
        self.d = d_new
        self.d_P = d_P_new
        self.fBPf = fBPf_new
        self.fcaBPf = fcaBPf_new
        self.fcaf = fcaf_new
        self.fcaf_P = fcaf_P_new
        self.fcafp = fcafp_new
        self.fcas = fcas_new
        self.fcas_P = fcas_P_new
        self.ff = ff_new
        self.ff_P = ff_P_new
        self.ffp = ffp_new
        self.fs = fs_new
        self.fs_P = fs_P_new
        self.h = h_new
        self.hL = hL_new
        self.hLp = hLp_new
        self.h_P = h_P_new
        self.hp = hp_new
        self.hp_P = hp_P_new
        self.ical_pureCDI_junc = ical_pureCDI_junc_new
        self.ical_pureCDI_sl = ical_pureCDI_sl_new
        self.j = j_new
        self.j_P = j_P_new
        self.jca = jca_new
        self.jp = jp_new
        self.jp_P = jp_P_new
        self.jrel_icaldep_act = jrel_icaldep_act_new
        self.jrel_icaldep_f1 = jrel_icaldep_f1_new
        self.jrel_icaldep_f2 = jrel_icaldep_f2_new
        self.m = m_new
        self.mL = mL_new
        self.m_P = m_P_new
        self.naj = naj_new
        self.nasl = nasl_new
        self.nca = nca_new
        self.nca_i = nca_i_new
        self.ryr_CaRI = ryr_CaRI_new
        self.ryr_CaRI_p = ryr_CaRI_p_new
        self.ryr_I = ryr_I_new
        self.ryr_I_p = ryr_I_p_new
        self.ryr_O = ryr_O_new
        self.ryr_O_p = ryr_O_p_new
        self.ryr_R = ryr_R_new
        self.ryr_R_p = ryr_R_p_new
        self.xs_junc = xs_junc_new
        self.xs_sl = xs_sl_new
        self.xtof = xtof_new
        self.xtof_p = xtof_p_new
        self.xtos = xtos_new
        self.xtos_p = xtos_p_new
        self.ytof = ytof_new
        self.ytof_p = ytof_p_new
        self.ytos = ytos_new
        self.ytos_p = ytos_p_new

        self.Cai = self.Cai * 1e3

        return -Iion


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    dt = 0.01
    dt_imp = float(np.float32(dt))   # limpet keeps the IMP time step in a float
    stimulus = 20
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    ionic = TWorld(cell_type="ENDO", 
                   dt=dt_imp, 
                   device=device, 
                   dtype=torch.float64)
    V = ionic.initialize(n_nodes=1)

    V_list = []
    Cai_list = []
    casr_list = []

    ctime = 0.0
    for _ in range(int(1000/dt)):
        V_list.append([ctime, V.item()])
        Cai_list.append([ctime, ionic.Cai.item()])
        casr_list.append([ctime, ionic.casr.item()])

        if ctime >= 0 and ctime < (0+2.0): 
            V = V + dt * stimulus
        dV = ionic.differentiate(V)
        V = V + dt * dV
        ctime += dt

    plt.figure()
    V_list = np.array(V_list)    
    plt.plot(V_list[:, 0], V_list[:, 1])
    plt.savefig("V_TWorld.png")

    plt.figure()
    Cai_list = np.array(Cai_list)    
    plt.plot(Cai_list[:, 0], Cai_list[:, 1])
    plt.savefig("Cai.png")

    plt.figure()
    casr_list = np.array(casr_list)    
    plt.plot(casr_list[:, 0], casr_list[:, 1])
    plt.savefig("casr.png")
