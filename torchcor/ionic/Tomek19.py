import torch
import torchcor as tc
from math import exp, log, sqrt, expm1
from typing import Optional, List


@torch.jit.script
class Tomek19:
    def __init__(self, 
                 dt: float, 
                 region_ids: Optional[List[int]] = None, 
                 cell_type: str = "ENDO", 
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float64):
        
        self.name = "Tomek19"
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
        self.ACap = (2.*self.Ageo)
        self.A_atp = 2.
        self.Aff = 0.6
        self.Afs = (1.-(self.Aff))
        self.BSLmax = 1.124
        self.BSRmax = 0.047
        self.C1_init = 7.0344e-4
        self.C2_init = 8.5109e-4
        self.C3_init = 0.9981
        self.CaMKo = 0.05
        self.CaMKt_init = 0.0111
        self.Cai_init = 8.1583e-05
        self.Cajsr_half = 1.7
        self.Cajsr_init = 1.5214
        self.Cansr_init = 1.5211
        self.Cao = 1.8
        self.Cass_init = 7.0305e-5
        self.Cli = 24.0
        self.Clo = 150.0
        self.F = 96485.
        self.R = 8314.
        self.T = 310.
        self.zcl = -1.
        self.ECl = (((self.R*self.T)/(self.zcl*self.F))*(log((self.Clo/self.Cli))))
        self.EKshift = 0.
        self.ENDO = 0.
        self.EPI = 1.
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
        self.I_init = 1.3289e-5
        self.Ko = 5.0
        self.Nao = 140.0
        self.Io = ((0.5*(((self.Nao+self.Ko)+self.Clo)+(4.*self.Cao)))/1000.)
        self.Jrel_b = 1.5378
        self.Jrel_np_init = 1.6129e-22
        self.Jrel_p_init = 1.2475e-20
        self.Jup_b = 1.0
        self.KCaon = 1.5e6
        self.KNa1 = 15.
        self.KNa2 = 5.
        self.Kasymm = 12.5
        self.h10_i = ((self.Kasymm+1.)+((self.Nao/self.KNa1)*(1.+(self.Nao/self.KNa2))))
        self.h12_i = (1./self.h10_i)
        self.K1_i = ((self.h12_i*self.Cao)*self.KCaon)
        self.h10_ss = ((self.Kasymm+1.)+((self.Nao/self.KNa1)*(1.+(self.Nao/self.KNa2))))
        self.h12_ss = (1./self.h10_ss)
        self.K1_ss = ((self.h12_ss*self.Cao)*self.KCaon)
        self.K1m = 182.4
        self.K1p = 949.5
        self.KCaoff = 5e3
        self.K2_i = self.KCaoff
        self.K2_ss = self.KCaoff
        self.K2m = 39.4
        self.K2n = 500.
        self.K2p = 687.2
        self.K3m = 79300.
        self.K3p = 1899.
        self.K4m = 40.
        self.K4p = 639.
        self.K5_i = self.KCaoff
        self.K5_ss = self.KCaoff
        self.KKi = 0.5
        self.KKo = 0.3582
        self.KNa3 = 88.12
        self.KNai0 = 9.073
        self.KNao0 = 27.78
        self.KNap = 224.
        self.K_atp = 0.25
        self.K_o_n = 5.
        self.KdClCa = 0.1
        self.Khp = 1.698e-7
        self.Ki_init = 142.3002
        self.KmBSL = 0.0087
        self.KmBSR = 0.00087
        self.KmCaAct = 150e-6
        self.KmCaM = 0.0015
        self.KmCaMK = 0.15
        self.KmCap = 0.0005
        self.Kmcmdn = 0.00238
        self.Kmcsqn = 0.8
        self.Kmgatp = 1.698e-7
        self.Kmn = 0.002
        self.Kmtrpn = 0.0005
        self.Kss_init = 142.3002
        self.KxKur = 292.
        self.MCELL = 2.
        self.MgADP = 0.05
        self.MgATP = 9.8
        self.Nai_init = 12.1025
        self.Nass_init = 12.1029
        self.O_init = 3.7585e-4
        self.PCa_b = 8.3757e-05
        self.PCab = 5.9194e-08
        self.PKNa = 0.01833
        self.PNaK_b = 15.4509
        self.PNab = 1.9239e-09
        self.VShift = 0.
        self.V_init = -88.7638
        self.Vcell = ((((1000.*3.14)*self.rad)*self.rad)*self.L)
        self.Vjsr = (0.0048*self.Vcell)
        self.Vmyo = (0.68*self.Vcell)
        self.Vnsr = (0.0552*self.Vcell)
        self.Vss = (0.02*self.Vcell)
        self.a2 = self.K2p
        self.a4 = (((self.K4p*self.MgATP)/self.Kmgatp)/(1.+(self.MgATP/self.Kmgatp)))
        self.aCaMK = 0.05
        self.aKiK = (pow((self.Ko/self.K_o_n),0.24))
        self.a_init = 9.5098e-4
        self.bt = 4.75
        self.a_rel = (0.5*self.bt)
        self.btp = (1.25*self.bt)
        self.a_relp = (0.5*self.btp)
        self.alpha_1 = 0.154375
        self.ap_init = 4.8454e-4
        self.b1 = (self.K1m*self.MgADP)
        self.bCaMK = 0.00068
        self.bKiK = (1./(1.+((self.A_atp/self.K_atp)*(self.A_atp/self.K_atp))))
        self.beta_1 = 0.1911
        self.cmdnmax_b = 0.05
        self.dielConstant = 74.
        self.constA = (1.82e6*(pow((self.dielConstant*self.T),-1.5)))
        self.csqnmax = 10.
        self.d_init = 8.1084e-9
        self.delta = -0.155
        self.eP = 4.2
        self.fCaf_init = 1.0
        self.fCafp_init = 1.0
        self.fCas_init = 0.9999
        self.fKatp = 0.0
        self.ff_init = 1.0
        self.ffp_init = 1.0
        self.fs_init = 0.939
        self.gKatp = 4.3195
        self.gamma_Cao = (exp((((-self.constA)*4.)*(((sqrt(self.Io))/(1.+(sqrt(self.Io))))-((0.3*self.Io))))))
        self.gamma_Ko = (exp(((-self.constA)*(((sqrt(self.Io))/(1.+(sqrt(self.Io))))-((0.3*self.Io))))))
        self.gamma_Nao = (exp(((-self.constA)*(((sqrt(self.Io))/(1.+(sqrt(self.Io))))-((0.3*self.Io))))))
        self.h11_i = ((self.Nao*self.Nao)/((self.h10_i*self.KNa1)*self.KNa2))
        self.h11_ss = ((self.Nao*self.Nao)/((self.h10_ss*self.KNa1)*self.KNa2))
        self.hL_init = 0.5255
        self.hLp_init = 0.2872
        self.h_init = 0.8286
        self.hp_init = 0.6707
        self.iF_init = 0.9996
        self.iFp_init = 0.9996
        self.iS_init = 0.5936
        self.iSp_init = 0.6538
        self.jCa_init = 1.0
        self.j_init = 0.8284
        self.jp_init = 0.8281
        self.mL_init = 1.629e-4
        self.m_init = 8.0572e-4
        self.nCa_i_init = 0.0012
        self.nCa_ss_init = 6.6462e-4
        self.offset = 0.
        self.qCa = 0.167
        self.qNa = 0.5224
        self.tauCa = 0.2
        self.tauK = 2.0
        self.tauNa = 2.0
        self.thL = 200.
        self.thLp = (3.*self.thL)
        self.tjCa = 75.
        self.trpnmax = 0.07
        self.wCa = 6e4
        self.wNa = 6e4
        self.wNaCa = 5e3
        self.xs1_init = 0.248
        self.xs2_init = 1.7707e-4
        self.zCa = 2.
        self.zK = 1.
        self.zNa = 1.

        # Parameters
        self.celltype = 1. if cell_type == "EPI" else 2. if cell_type == "MCELL" else 0.

        # Cai_TableIndex
        self.BCai_idx = 0
        self.IpCa_idx = 1
        self.Jupnp_idx = 2
        self.Jupp_idx = 3
        self.KsCa_idx = 4
        self.allo_i_idx = 5
        self.Cai_NROWS = 6

        # V_TableIndex
        self.AfCaf_idx = 0
        self.AfCas_idx = 1
        self.AiF_idx = 2
        self.AiS_idx = 3
        self.IClb_idx = 4
        self.K3_i_idx = 5
        self.K3_ss_idx = 6
        self.K3pp_i_idx = 7
        self.K3pp_ss_idx = 8
        self.K8_i_idx = 9
        self.K8_ss_idx = 10
        self.KNai_idx = 11
        self.Vffrt_idx = 12
        self.Vfrt_idx = 13
        self.a3_idx = 14
        self.alpha_idx = 15
        self.alpha_2_idx = 16
        self.alpha_C2ToI_idx = 17
        self.alpha_i_idx = 18
        self.ass_idx = 19
        self.assp_idx = 20
        self.b2_idx = 21
        self.beta_idx = 22
        self.beta_2_idx = 23
        self.beta_ItoC2_idx = 24
        self.beta_i_idx = 25
        self.dss_idx = 26
        self.fCass_idx = 27
        self.fss_idx = 28
        self.hCa_idx = 29
        self.hLss_idx = 30
        self.hLssp_idx = 31
        self.hNa_idx = 32
        self.hss_idx = 33
        self.hssp_idx = 34
        self.iss_idx = 35
        self.jCass_idx = 36
        self.jss_idx = 37
        self.mLss_idx = 38
        self.mss_idx = 39
        self.ta_idx = 40
        self.td_idx = 41
        self.tfCaf_idx = 42
        self.tfCafp_idx = 43
        self.tfCas_idx = 44
        self.tff_idx = 45
        self.tffp_idx = 46
        self.tfs_idx = 47
        self.th_idx = 48
        self.tiF_idx = 49
        self.tiFp_idx = 50
        self.tiS_idx = 51
        self.tiSp_idx = 52
        self.tj_idx = 53
        self.tjp_idx = 54
        self.tm_idx = 55
        self.tmL_idx = 56
        self.txs1_idx = 57
        self.txs2_idx = 58
        self.xKb_idx = 59
        self.xs1ss_idx = 60
        self.xs2ss_idx = 61
        self.V_NROWS = 62

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

        # 42 states variables
        self.C1 = torch.tensor([self.C1_init])
        self.C2 = torch.tensor([self.C2_init])
        self.C3 = torch.tensor([self.C3_init])
        self.CaMKt = torch.tensor([self.CaMKt_init])
        self.Cai = torch.tensor([self.Cai_init])
        self.Cajsr = torch.tensor([self.Cajsr_init])
        self.Cansr = torch.tensor([self.Cansr_init])
        self.Cass = torch.tensor([self.Cass_init])
        self.I = torch.tensor([self.I_init])
        self.Jrel_np = torch.tensor([self.Jrel_np_init])
        self.Jrel_p = torch.tensor([self.Jrel_p_init])
        self.Ki = torch.tensor([self.Ki_init])
        self.Kss = torch.tensor([self.Kss_init])
        self.Nai = torch.tensor([self.Nai_init])
        self.Nass = torch.tensor([self.Nass_init])
        self.O = torch.tensor([self.O_init])
        self.a = torch.tensor([self.a_init])
        self.ap = torch.tensor([self.ap_init])
        self.d = torch.tensor([self.d_init])
        self.fCaf = torch.tensor([self.fCaf_init])
        self.fCafp = torch.tensor([self.fCafp_init])
        self.fCas = torch.tensor([self.fCas_init])
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
        self.jCa = torch.tensor([self.jCa_init])
        self.jp = torch.tensor([self.jp_init])
        self.m = torch.tensor([self.m_init])
        self.mL = torch.tensor([self.mL_init])
        self.nCa_i = torch.tensor([self.nCa_i_init])
        self.nCa_ss = torch.tensor([self.nCa_ss_init])
        self.xs1 = torch.tensor([self.xs1_init])
        self.xs2 = torch.tensor([self.xs2_init])
        
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
        GK1 = ((self.GK1_b*1.2) if (self.celltype==1.) else ((self.GK1_b*1.3) if (self.celltype==2.) else self.GK1_b))
        GKb = ((self.GKb_b*0.6) if (self.celltype==1.) else self.GKb_b)
        GKr = ((self.GKr_b*1.3) if (self.celltype==1.) else ((self.GKr_b*0.8) if (self.celltype==2.) else self.GKr_b))
        GKs = ((self.GKs_b*1.4) if (self.celltype==1.) else self.GKs_b)
        GNaL = ((self.GNaL_b*0.6) if (self.celltype==1.) else self.GNaL_b)
        Gncx = ((self.Gncx_b*1.1) if (self.celltype==1.) else ((self.Gncx_b*1.4) if (self.celltype==2.) else self.Gncx_b))
        Gto = ((self.Gto_b*2.) if (self.celltype==1.) else ((self.Gto_b*2.) if (self.celltype==2.) else self.Gto_b))
        PCa = ((self.PCa_b*1.2) if (self.celltype==1.) else ((self.PCa_b*2.) if (self.celltype==2.) else self.PCa_b))
        PNaK = ((self.PNaK_b*0.9) if (self.celltype==1.) else ((self.PNaK_b*0.7) if (self.celltype==2.) else self.PNaK_b))
        cmdnmax = ((self.cmdnmax_b*1.3) if (self.celltype==1.) else self.cmdnmax_b)
        upSCale = (1.3 if (self.celltype==1.) else 1.)
        PCaK = (3.574e-4*PCa)
        PCaNa = (0.00125*PCa)
        PCap = (1.1*PCa)
        PCaKp = (3.574e-4*PCap)
        PCaNap = (0.00125*PCap)

        # construct the Cai lookup table
        Cai = torch.arange(self.Cai_T_mn_ind, self.Cai_T_mx_ind + 1, device=self.device, dtype=self.dtype) * self.Cai_T_res
        Cai_tab = torch.zeros((Cai.shape[0], self.Cai_NROWS)).to(self.device).to(self.dtype)

        Cai_tab[:, self.BCai_idx] = (1./((1.+((cmdnmax*self.Kmcmdn)/((self.Kmcmdn+Cai)*(self.Kmcmdn+Cai))))+((self.trpnmax*self.Kmtrpn)/((self.Kmtrpn+Cai)*(self.Kmtrpn+Cai)))))
        Cai_tab[:, self.IpCa_idx] = ((self.GpCa*Cai)/(self.KmCap+Cai))
        Cai_tab[:, self.Jupnp_idx] = (((upSCale*0.005425)*Cai)/(Cai+0.00092))
        Cai_tab[:, self.Jupp_idx] = ((((upSCale*2.75)*0.005425)*Cai)/((Cai+0.00092)-(0.00017)))
        Cai_tab[:, self.KsCa_idx] = (1.+(0.6/(1.+(torch.pow((3.8e-5/Cai),1.4)))))
        Cai_tab[:, self.allo_i_idx] = (1./(1.+((self.KmCaAct/Cai)*(self.KmCaAct/Cai))))

        self.Cai_tab = Cai_tab

        # construct the V lookup table
        V = torch.arange(self.V_T_mn_ind, self.V_T_mx_ind + 1, device=self.device, dtype=self.dtype) * self.V_T_res
        V_tab = torch.zeros((V.shape[0], self.V_NROWS)).to(self.device).to(self.dtype)

        V_tab[:, self.AfCaf_idx] = (0.3+(0.6/(1.+(torch.exp(((V-(10.))/10.))))))
        V_tab[:, self.AiF_idx] = (1./(1.+(torch.exp((((V+self.EKshift)-(213.6))/151.2)))))
        V_tab[:, self.IClb_idx] = (self.GClb*(V-(self.ECl)))
        V_tab[:, self.Vffrt_idx] = (((V*self.F)*self.F)/(self.R*self.T))
        V_tab[:, self.Vfrt_idx] = ((V*self.F)/(self.R*self.T))
        ah = (torch.where((V>=-40.), 0., (0.057*(torch.exp(((-(V+80.))/6.8))))))
        aj = (torch.where((V>=-40.), 0., ((((-2.5428e4*(torch.exp((0.2444*V))))-((6.948e-6*(torch.exp((-0.04391*V))))))*(V+37.78))/(1.+(torch.exp((0.311*(V+79.23))))))))
        V_tab[:, self.ass_idx] = (1./(1.+(torch.exp(((-((V+self.EKshift)-(14.34)))/14.82)))))
        V_tab[:, self.assp_idx] = (1./(1.+(torch.exp(((-((V+self.EKshift)-(24.34)))/14.82)))))
        bh = (torch.where((V>=-40.), (0.77/(0.13*(1.+(torch.exp(((-(V+10.66))/11.1)))))), ((2.7*(torch.exp((0.079*V))))+(3.1e5*(torch.exp((0.3485*V)))))))
        bj = (torch.where((V>=-40.), ((0.6*(torch.exp((0.057*V))))/(1.+(torch.exp((-0.1*(V+32.)))))), ((0.02424*(torch.exp((-0.01052*V))))/(1.+(torch.exp((-0.1378*(V+40.14))))))))
        delta_epi = ((1.-((0.95/(1.+(torch.exp((((V+self.EKshift)+70.)/5.))))))) if (self.celltype==1.) else torch.full_like(V, 1.))
        V_tab[:, self.dss_idx] = (torch.where((V>=31.4978), 1., (1.0763*(torch.exp((-1.0070*(torch.exp((-0.0829*V)))))))))
        dti_deVelop = (1.354+(1.e-4/((torch.exp((((V+self.EKshift)-(167.4))/15.89)))+(torch.exp(((-((V+self.EKshift)-(12.23)))/0.2154))))))
        dti_recoVer = (1.-((0.5/(1.+(torch.exp((((V+self.EKshift)+70.0)/20.0)))))))
        V_tab[:, self.fss_idx] = (1./(1.+(torch.exp(((V+19.58)/3.696)))))
        V_tab[:, self.hLss_idx] = (1./(1.+(torch.exp(((V+87.61)/7.488)))))
        V_tab[:, self.hLssp_idx] = (1./(1.+(torch.exp(((V+93.81)/7.488)))))
        V_tab[:, self.hss_idx] = (1./((1.+(torch.exp(((V+71.55)/7.43))))*(1.+(torch.exp(((V+71.55)/7.43))))))
        V_tab[:, self.hssp_idx] = (1./((1.+(torch.exp(((V+77.55)/7.43))))*(1.+(torch.exp(((V+77.55)/7.43))))))
        V_tab[:, self.iss_idx] = (1./(1.+(torch.exp((((V+self.EKshift)+43.94)/5.711)))))
        V_tab[:, self.jCass_idx] = (1.0/(1.0+(torch.exp(((V+18.08)/2.7916)))))
        V_tab[:, self.mLss_idx] = (1./(1.+(torch.exp(((-(V+42.85))/5.264)))))
        V_tab[:, self.mss_idx] = (1./((1.+(torch.exp(((-(V+56.86))/9.03))))*(1.+(torch.exp(((-(V+56.86))/9.03))))))
        V_tab[:, self.ta_idx] = (1.0515/((1./(1.2089*(1.+(torch.exp(((-((V+self.EKshift)-(18.4099)))/29.3814))))))+(3.5/(1.+(torch.exp((((V+self.EKshift)+100.)/29.3814)))))))
        V_tab[:, self.td_idx] = ((self.offset+0.6)+(1./((torch.exp((-0.05*((V+self.VShift)+6.))))+(torch.exp((0.09*((V+self.VShift)+14.)))))))
        V_tab[:, self.tfCaf_idx] = (7.+(1./((0.04*(torch.exp(((-(V-(4.)))/7.))))+(0.04*(torch.exp(((V-(4.))/7.)))))))
        V_tab[:, self.tfCas_idx] = (100.+(1./((0.00012*(torch.exp(((-V)/3.))))+(0.00012*(torch.exp((V/7.)))))))
        V_tab[:, self.tff_idx] = (7.+(1./((0.0045*(torch.exp(((-(V+20.))/10.))))+(0.0045*(torch.exp(((V+20.)/10.)))))))
        V_tab[:, self.tfs_idx] = (1000.+(1./((0.000035*(torch.exp(((-(V+5.))/4.))))+(0.000035*(torch.exp(((V+5.)/6.)))))))
        tiF_b = (4.562+(1./((0.3933*(torch.exp(((-((V+self.EKshift)+100.))/100.))))+(0.08004*(torch.exp((((V+self.EKshift)+50.)/16.59)))))))
        tiS_b = (23.62+(1./((0.001416*(torch.exp(((-((V+self.EKshift)+96.52))/59.05))))+(1.78e-8*(torch.exp((((V+self.EKshift)+114.1)/8.079)))))))
        V_tab[:, self.tm_idx] = ((0.1292*(torch.exp((-(((V+45.79)/15.54)*((V+45.79)/15.54))))))+(0.06487*(torch.exp((-(((V-(4.823))/51.12)*((V-(4.823))/51.12)))))))
        V_tab[:, self.tmL_idx] = ((0.1292*(torch.exp((-(((V+45.79)/15.54)*((V+45.79)/15.54))))))+(0.06487*(torch.exp((-(((V-(4.823))/51.12)*((V-(4.823))/51.12)))))))
        V_tab[:, self.txs1_idx] = (817.3+(1./((2.326e-4*(torch.exp(((V+48.28)/17.8))))+(0.001292*(torch.exp(((-(V+210.))/230.)))))))
        V_tab[:, self.txs2_idx] = (1./((0.01*(torch.exp(((V-(50.))/20.))))+(0.0193*(torch.exp(((-(V+66.54))/31.))))))
        V_tab[:, self.xKb_idx] = (1./(1.+(torch.exp(((-(V-(10.8968)))/23.9871)))))
        V_tab[:, self.xs1ss_idx] = (1./(1.+(torch.exp(((-(V+11.6))/8.932)))))
        V_tab[:, self.AfCas_idx] = (1.-(V_tab[:, self.AfCaf_idx]))
        V_tab[:, self.AiS_idx] = (1.-(V_tab[:, self.AiF_idx]))
        V_tab[:, self.KNai_idx] = (self.KNai0*(torch.exp(((self.delta*V_tab[:, self.Vfrt_idx])/3.))))
        KNao = (self.KNao0*(torch.exp((((1.-(self.delta))*V_tab[:, self.Vfrt_idx])/3.))))
        V_tab[:, self.alpha_idx] = (0.1161*(torch.exp((0.2990*V_tab[:, self.Vfrt_idx]))))
        V_tab[:, self.alpha_2_idx] = (0.0578*(torch.exp((0.9710*V_tab[:, self.Vfrt_idx]))))
        V_tab[:, self.alpha_C2ToI_idx] = (0.52e-4*(torch.exp((1.525*V_tab[:, self.Vfrt_idx]))))
        V_tab[:, self.alpha_i_idx] = (0.2533*(torch.exp((0.5953*V_tab[:, self.Vfrt_idx]))))
        V_tab[:, self.beta_idx] = (0.2442*(torch.exp((-1.604*V_tab[:, self.Vfrt_idx]))))
        V_tab[:, self.beta_2_idx] = (0.349e-3*(torch.exp((-1.062*V_tab[:, self.Vfrt_idx]))))
        V_tab[:, self.beta_i_idx] = (0.06525*(torch.exp((-0.8209*V_tab[:, self.Vfrt_idx]))))
        V_tab[:, self.fCass_idx] = V_tab[:, self.fss_idx]
        V_tab[:, self.hCa_idx] = (torch.exp((self.qCa*V_tab[:, self.Vfrt_idx])))
        V_tab[:, self.hNa_idx] = (torch.exp((self.qNa*V_tab[:, self.Vfrt_idx])))
        V_tab[:, self.jss_idx] = V_tab[:, self.hss_idx]
        V_tab[:, self.tfCafp_idx] = (2.5*V_tab[:, self.tfCaf_idx])
        V_tab[:, self.tffp_idx] = (2.5*V_tab[:, self.tff_idx])
        V_tab[:, self.th_idx] = (1./(ah+bh))
        V_tab[:, self.tiF_idx] = (tiF_b*delta_epi)
        V_tab[:, self.tiS_idx] = (tiS_b*delta_epi)
        V_tab[:, self.tj_idx] = (1./(aj+bj))
        V_tab[:, self.xs2ss_idx] = V_tab[:, self.xs1ss_idx]
        V_tab[:, self.a3_idx] = ((self.K3p*((self.Ko/self.KKo)*(self.Ko/self.KKo)))/(((((1.+(self.Nao/KNao))*(1.+(self.Nao/KNao)))*(1.+(self.Nao/KNao)))+((1.+(self.Ko/self.KKo))*(1.+(self.Ko/self.KKo))))-(1.)))
        V_tab[:, self.b2_idx] = ((self.K2m*(((self.Nao/KNao)*(self.Nao/KNao))*(self.Nao/KNao)))/(((((1.+(self.Nao/KNao))*(1.+(self.Nao/KNao)))*(1.+(self.Nao/KNao)))+((1.+(self.Ko/self.KKo))*(1.+(self.Ko/self.KKo))))-(1.)))
        V_tab[:, self.beta_ItoC2_idx] = (((V_tab[:, self.beta_2_idx]*V_tab[:, self.beta_i_idx])*V_tab[:, self.alpha_C2ToI_idx])/(V_tab[:, self.alpha_2_idx]*V_tab[:, self.alpha_i_idx]))
        h7_i = (1.+((self.Nao/self.KNa3)*(1.+(1./V_tab[:, self.hNa_idx]))))
        h7_ss = (1.+((self.Nao/self.KNa3)*(1.+(1./V_tab[:, self.hNa_idx]))))
        V_tab[:, self.tiFp_idx] = ((dti_deVelop*dti_recoVer)*V_tab[:, self.tiF_idx])
        V_tab[:, self.tiSp_idx] = ((dti_deVelop*dti_recoVer)*V_tab[:, self.tiS_idx])
        V_tab[:, self.tjp_idx] = (1.46*V_tab[:, self.tj_idx])
        h8_i = (self.Nao/((self.KNa3*V_tab[:, self.hNa_idx])*h7_i))
        h8_ss = (self.Nao/((self.KNa3*V_tab[:, self.hNa_idx])*h7_ss))
        h9_i = (1./h7_i)
        h9_ss = (1./h7_ss)
        K3p_i = (h9_i*self.wCa)
        K3p_ss = (h9_ss*self.wCa)
        V_tab[:, self.K3pp_i_idx] = (h8_i*self.wNaCa)
        V_tab[:, self.K3pp_ss_idx] = (h8_ss*self.wNaCa)
        V_tab[:, self.K8_i_idx] = ((h8_i*self.h11_i)*self.wNa)
        V_tab[:, self.K8_ss_idx] = ((h8_ss*self.h11_ss)*self.wNa)
        V_tab[:, self.K3_i_idx] = (K3p_i+V_tab[:, self.K3pp_i_idx])
        V_tab[:, self.K3_ss_idx] = (K3p_ss+V_tab[:, self.K3pp_ss_idx])

        self.V_tab = V_tab

    def initialize(self, n_nodes: int):
        self.construct_tables()
        
        V = torch.full((n_nodes,), self.V_init, device=self.device, dtype=self.dtype)

        self.C1 = torch.full((n_nodes,), self.C1_init, device=self.device, dtype=self.dtype)
        self.C2 = torch.full((n_nodes,), self.C2_init, device=self.device, dtype=self.dtype)
        self.C3 = torch.full((n_nodes,), self.C3_init, device=self.device, dtype=self.dtype)
        self.CaMKt = torch.full((n_nodes,), self.CaMKt_init, device=self.device, dtype=self.dtype)
        self.Cai = torch.full((n_nodes,), self.Cai_init, device=self.device, dtype=self.dtype)
        self.Cai = self.Cai * 1e3
        self.Cajsr = torch.full((n_nodes,), self.Cajsr_init, device=self.device, dtype=self.dtype)
        self.Cansr = torch.full((n_nodes,), self.Cansr_init, device=self.device, dtype=self.dtype)
        self.Cass = torch.full((n_nodes,), self.Cass_init, device=self.device, dtype=self.dtype)
        self.I = torch.full((n_nodes,), self.I_init, device=self.device, dtype=self.dtype)
        self.Jrel_np = torch.full((n_nodes,), self.Jrel_np_init, device=self.device, dtype=self.dtype)
        self.Jrel_p = torch.full((n_nodes,), self.Jrel_p_init, device=self.device, dtype=self.dtype)
        self.Ki = torch.full((n_nodes,), self.Ki_init, device=self.device, dtype=self.dtype)
        self.Kss = torch.full((n_nodes,), self.Kss_init, device=self.device, dtype=self.dtype)
        self.Nai = torch.full((n_nodes,), self.Nai_init, device=self.device, dtype=self.dtype)
        self.Nass = torch.full((n_nodes,), self.Nass_init, device=self.device, dtype=self.dtype)
        self.O = torch.full((n_nodes,), self.O_init, device=self.device, dtype=self.dtype)
        self.a = torch.full((n_nodes,), self.a_init, device=self.device, dtype=self.dtype)
        self.ap = torch.full((n_nodes,), self.ap_init, device=self.device, dtype=self.dtype)
        self.d = torch.full((n_nodes,), self.d_init, device=self.device, dtype=self.dtype)
        self.fCaf = torch.full((n_nodes,), self.fCaf_init, device=self.device, dtype=self.dtype)
        self.fCafp = torch.full((n_nodes,), self.fCafp_init, device=self.device, dtype=self.dtype)
        self.fCas = torch.full((n_nodes,), self.fCas_init, device=self.device, dtype=self.dtype)
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
        self.jCa = torch.full((n_nodes,), self.jCa_init, device=self.device, dtype=self.dtype)
        self.jp = torch.full((n_nodes,), self.jp_init, device=self.device, dtype=self.dtype)
        self.m = torch.full((n_nodes,), self.m_init, device=self.device, dtype=self.dtype)
        self.mL = torch.full((n_nodes,), self.mL_init, device=self.device, dtype=self.dtype)
        self.nCa_i = torch.full((n_nodes,), self.nCa_i_init, device=self.device, dtype=self.dtype)
        self.nCa_ss = torch.full((n_nodes,), self.nCa_ss_init, device=self.device, dtype=self.dtype)
        self.xs1 = torch.full((n_nodes,), self.xs1_init, device=self.device, dtype=self.dtype)
        self.xs2 = torch.full((n_nodes,), self.xs2_init, device=self.device, dtype=self.dtype)

        return V

    def differentiate(self, V):
        self.Cai = self.Cai * 1e-3

        # Define the constants that depend on the parameters.
        GK1 = ((self.GK1_b*1.2) if (self.celltype==1.) else ((self.GK1_b*1.3) if (self.celltype==2.) else self.GK1_b))
        GKb = ((self.GKb_b*0.6) if (self.celltype==1.) else self.GKb_b)
        GKr = ((self.GKr_b*1.3) if (self.celltype==1.) else ((self.GKr_b*0.8) if (self.celltype==2.) else self.GKr_b))
        GKs = ((self.GKs_b*1.4) if (self.celltype==1.) else self.GKs_b)
        GNaL = ((self.GNaL_b*0.6) if (self.celltype==1.) else self.GNaL_b)
        Gncx = ((self.Gncx_b*1.1) if (self.celltype==1.) else ((self.Gncx_b*1.4) if (self.celltype==2.) else self.Gncx_b))
        Gto = ((self.Gto_b*2.) if (self.celltype==1.) else ((self.Gto_b*2.) if (self.celltype==2.) else self.Gto_b))
        PCa = ((self.PCa_b*1.2) if (self.celltype==1.) else ((self.PCa_b*2.) if (self.celltype==2.) else self.PCa_b))
        PNaK = ((self.PNaK_b*0.9) if (self.celltype==1.) else ((self.PNaK_b*0.7) if (self.celltype==2.) else self.PNaK_b))
        cmdnmax = ((self.cmdnmax_b*1.3) if (self.celltype==1.) else self.cmdnmax_b)
        upSCale = (1.3 if (self.celltype==1.) else 1.)
        PCaK = (3.574e-4*PCa)
        PCaNa = (0.00125*PCa)
        PCap = (1.1*PCa)
        PCaKp = (3.574e-4*PCap)
        PCaNap = (0.00125*PCap)

        Cai_row = self.interpolate(self.Cai, self.Cai_tab, self.Cai_T_mn, self.Cai_T_mx, self.Cai_T_res, self.Cai_T_step, self.Cai_T_mn_ind, self.Cai_T_mx_ind)
        V_row = self.interpolate(V, self.V_tab, self.V_T_mn, self.V_T_mx, self.V_T_res, self.V_T_step, self.V_T_mn_ind, self.V_T_mx_ind)
        # GHK terms are 0/0 at V=0 (and underflow below ~1e-20); nudge Vfrt/Vffrt off exact zero, keeping Vffrt = F*Vfrt
        Vfrt_safe = V_row[:, self.Vfrt_idx]
        Vffrt_safe = V_row[:, self.Vffrt_idx]
        _at_zero = (torch.abs(Vfrt_safe) < 1e-20)
        Vfrt_safe = torch.where(_at_zero, torch.full_like(Vfrt_safe, 1e-12), Vfrt_safe)
        Vffrt_safe = torch.where(_at_zero, torch.full_like(Vffrt_safe, 1e-12 * self.F), Vffrt_safe)

        # Compute storevars and external modvars
        CaMKb = ((self.CaMKo*(1.-(self.CaMKt)))/(1.+(self.KmCaM/self.Cass)))
        EK = (((self.R*self.T)/(self.zK*self.F))*(torch.log((self.Ko/self.Ki))))
        EKs = (((self.R*self.T)/(self.zK*self.F))*(torch.log(((self.Ko+(self.PKNa*self.Nao))/(self.Ki+(self.PKNa*self.Nai))))))
        ENa = (((self.R*self.T)/(self.zNa*self.F))*(torch.log((self.Nao/self.Nai))))
        IClCa_junc = (((self.Fjunc*self.GClCa)/(1.+(self.KdClCa/self.Cass)))*(V-(self.ECl)))
        IClCa_sl = ((((1.-(self.Fjunc))*self.GClCa)/(1.+(self.KdClCa/self.Cai)))*(V-(self.ECl)))
        Ii = ((0.5*(((self.Nai+self.Ki)+self.Cli)+(4.*self.Cai)))/1000.)
        Iss = ((0.5*(((self.Nass+self.Kss)+self.Cli)+(4.*self.Cass)))/1000.)
        P = (self.eP/(((1.+(self.H/self.Khp))+(self.Nai/self.KNap))+(self.Ki/self.KxKur)))
        allo_ss = (1./(1.+((self.KmCaAct/self.Cass)*(self.KmCaAct/self.Cass))))
        f = ((self.Aff*self.ff)+(self.Afs*self.fs))
        fp = ((self.Aff*self.ffp)+(self.Afs*self.fs))
        h4_i = (1.+((self.Nai/self.KNa1)*(1.+(self.Nai/self.KNa2))))
        h4_ss = (1.+((self.Nass/self.KNa1)*(1.+(self.Nass/self.KNa2))))
        CaMKa = (CaMKb+self.CaMKt)
        IClCa = (IClCa_junc+IClCa_sl)
        IKATP = ((((self.fKatp*self.gKatp)*self.aKiK)*self.bKiK)*(V-(EK)))
        IKb = ((GKb*V_row[:, self.xKb_idx])*(V-(EK)))
        IKr = (((GKr*(sqrt((self.Ko/5.))))*self.O)*(V-(EK)))
        IKs = ((((GKs*Cai_row[:, self.KsCa_idx])*self.xs1)*self.xs2)*(V-(EKs)))
        INab = (((self.PNab*Vffrt_safe)*((self.Nai*(torch.exp(Vfrt_safe)))-(self.Nao)))/((torch.expm1(Vfrt_safe))))
        aK1 = (4.094/(1.+(torch.exp((0.1217*((V-(EK))-(49.934)))))))
        b3 = (((self.K3m*P)*self.H)/(1.+(self.MgATP/self.Kmgatp)))
        bK1 = (((15.72*(torch.exp((0.0674*((V-(EK))-(3.257))))))+(torch.exp((0.0618*((V-(EK))-(594.31))))))/(1.+(torch.exp((-0.1629*((V-(EK))+14.207))))))
        gamma_Cai = (torch.exp((((-self.constA)*4.)*(((torch.sqrt(Ii))/(1.+(torch.sqrt(Ii))))-((0.3*Ii))))))
        gamma_Cass = (torch.exp((((-self.constA)*4.)*(((torch.sqrt(Iss))/(1.+(torch.sqrt(Iss))))-((0.3*Iss))))))
        gamma_Ki = (torch.exp(((-self.constA)*(((torch.sqrt(Ii))/(1.+(torch.sqrt(Ii))))-((0.3*Ii))))))
        gamma_Kss = (torch.exp(((-self.constA)*(((torch.sqrt(Iss))/(1.+(torch.sqrt(Iss))))-((0.3*Iss))))))
        gamma_Nai = (torch.exp(((-self.constA)*(((torch.sqrt(Ii))/(1.+(torch.sqrt(Ii))))-((0.3*Ii))))))
        gamma_Nass = (torch.exp(((-self.constA)*(((torch.sqrt(Iss))/(1.+(torch.sqrt(Iss))))-((0.3*Iss))))))
        h5_i = ((self.Nai*self.Nai)/((h4_i*self.KNa1)*self.KNa2))
        h5_ss = ((self.Nass*self.Nass)/((h4_ss*self.KNa1)*self.KNa2))
        h6_i = (1./h4_i)
        h6_ss = (1./h4_ss)
        ICab = ((((self.PCab*4.)*Vffrt_safe)*(((gamma_Cai*self.Cai)*(torch.exp((2.*Vfrt_safe))))-((self.gamma_Cao*self.Cao))))/((torch.expm1((2.*Vfrt_safe)))))
        K1ss = (aK1/(aK1+bK1))
        K6_i = ((h6_i*self.Cai)*self.KCaon)
        K6_ss = ((h6_ss*self.Cass)*self.KCaon)
        PhiCaK_i = ((Vffrt_safe*(((gamma_Ki*self.Ki)*(torch.exp(Vfrt_safe)))-((self.gamma_Ko*self.Ko))))/((torch.expm1(Vfrt_safe))))
        PhiCaK_ss = ((Vffrt_safe*(((gamma_Kss*self.Kss)*(torch.exp(Vfrt_safe)))-((self.gamma_Ko*self.Ko))))/((torch.expm1(Vfrt_safe))))
        PhiCaL_i = (((4.*Vffrt_safe)*(((gamma_Cai*self.Cai)*(torch.exp((2.*Vfrt_safe))))-((self.gamma_Cao*self.Cao))))/((torch.expm1((2.*Vfrt_safe)))))
        PhiCaL_ss = (((4.*Vffrt_safe)*(((gamma_Cass*self.Cass)*(torch.exp((2.*Vfrt_safe))))-((self.gamma_Cao*self.Cao))))/((torch.expm1((2.*Vfrt_safe)))))
        PhiCaNa_i = ((Vffrt_safe*(((gamma_Nai*self.Nai)*(torch.exp(Vfrt_safe)))-((self.gamma_Nao*self.Nao))))/((torch.expm1(Vfrt_safe))))
        PhiCaNa_ss = ((Vffrt_safe*(((gamma_Nass*self.Nass)*(torch.exp(Vfrt_safe)))-((self.gamma_Nao*self.Nao))))/((torch.expm1(Vfrt_safe))))
        a1 = ((self.K1p*(((self.Nai/V_row[:, self.KNai_idx])*(self.Nai/V_row[:, self.KNai_idx]))*(self.Nai/V_row[:, self.KNai_idx])))/(((((1.+(self.Nai/V_row[:, self.KNai_idx]))*(1.+(self.Nai/V_row[:, self.KNai_idx])))*(1.+(self.Nai/V_row[:, self.KNai_idx])))+((1.+(self.Ki/self.KKi))*(1.+(self.Ki/self.KKi))))-(1.)))
        b4 = ((self.K4m*((self.Ki/self.KKi)*(self.Ki/self.KKi)))/(((((1.+(self.Nai/V_row[:, self.KNai_idx]))*(1.+(self.Nai/V_row[:, self.KNai_idx])))*(1.+(self.Nai/V_row[:, self.KNai_idx])))+((1.+(self.Ki/self.KKi))*(1.+(self.Ki/self.KKi))))-(1.)))
        fCa = ((V_row[:, self.AfCaf_idx]*self.fCaf)+(V_row[:, self.AfCas_idx]*self.fCas))
        fCap = ((V_row[:, self.AfCaf_idx]*self.fCafp)+(V_row[:, self.AfCas_idx]*self.fCas))
        fICaLp = (1./(1.+(self.KmCaMK/CaMKa)))
        fINaLp = (1./(1.+(self.KmCaMK/CaMKa)))
        fINap = (1./(1.+(self.KmCaMK/CaMKa)))
        fItop = (1./(1.+(self.KmCaMK/CaMKa)))
        h1_i = (1.+((self.Nai/self.KNa3)*(1.+V_row[:, self.hNa_idx])))
        h1_ss = (1.+((self.Nass/self.KNa3)*(1.+V_row[:, self.hNa_idx])))
        i_t = ((V_row[:, self.AiF_idx]*self.iF)+(V_row[:, self.AiS_idx]*self.iS))
        ip = ((V_row[:, self.AiF_idx]*self.iFp)+(V_row[:, self.AiS_idx]*self.iSp))
        ICaK_i = ((1.-(self.ICaL_fractionSS))*((((((1.-(fICaLp))*PCaK)*PhiCaK_i)*self.d)*((f*(1.-(self.nCa_i)))+((self.jCa*fCa)*self.nCa_i)))+((((fICaLp*PCaKp)*PhiCaK_i)*self.d)*((fp*(1.-(self.nCa_i)))+((self.jCa*fCap)*self.nCa_i)))))
        ICaK_ss = (self.ICaL_fractionSS*((((((1.-(fICaLp))*PCaK)*PhiCaK_ss)*self.d)*((f*(1.-(self.nCa_ss)))+((self.jCa*fCa)*self.nCa_ss)))+((((fICaLp*PCaKp)*PhiCaK_ss)*self.d)*((fp*(1.-(self.nCa_ss)))+((self.jCa*fCap)*self.nCa_ss)))))
        ICaL_i = ((1.-(self.ICaL_fractionSS))*((((((1.-(fICaLp))*PCa)*PhiCaL_i)*self.d)*((f*(1.-(self.nCa_i)))+((self.jCa*fCa)*self.nCa_i)))+((((fICaLp*PCap)*PhiCaL_i)*self.d)*((fp*(1.-(self.nCa_i)))+((self.jCa*fCap)*self.nCa_i)))))
        ICaL_ss = (self.ICaL_fractionSS*((((((1.-(fICaLp))*PCa)*PhiCaL_ss)*self.d)*((f*(1.-(self.nCa_ss)))+((self.jCa*fCa)*self.nCa_ss)))+((((fICaLp*PCap)*PhiCaL_ss)*self.d)*((fp*(1.-(self.nCa_ss)))+((self.jCa*fCap)*self.nCa_ss)))))
        ICaNa_i = ((1.-(self.ICaL_fractionSS))*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_i)*self.d)*((f*(1.-(self.nCa_i)))+((self.jCa*fCa)*self.nCa_i)))+((((fICaLp*PCaNap)*PhiCaNa_i)*self.d)*((fp*(1.-(self.nCa_i)))+((self.jCa*fCap)*self.nCa_i)))))
        ICaNa_ss = (self.ICaL_fractionSS*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_ss)*self.d)*((f*(1.-(self.nCa_ss)))+((self.jCa*fCa)*self.nCa_ss)))+((((fICaLp*PCaNap)*PhiCaNa_ss)*self.d)*((fp*(1.-(self.nCa_ss)))+((self.jCa*fCap)*self.nCa_ss)))))
        IK1 = (((GK1*(sqrt((self.Ko/5.))))*K1ss)*(V-(EK)))
        INa = (((self.GNa*(V-(ENa)))*((self.m*self.m)*self.m))*((((1.-(fINap))*self.h)*self.j)+((fINap*self.hp)*self.jp)))
        INaL = (((GNaL*(V-(ENa)))*self.mL)*(((1.-(fINaLp))*self.hL)+(fINaLp*self.hLp)))
        Ito = ((Gto*(V-(EK)))*((((1.-(fItop))*self.a)*i_t)+((fItop*self.ap)*ip)))
        h2_i = ((self.Nai*V_row[:, self.hNa_idx])/(self.KNa3*h1_i))
        h2_ss = ((self.Nass*V_row[:, self.hNa_idx])/(self.KNa3*h1_ss))
        h3_i = (1./h1_i)
        h3_ss = (1./h1_ss)
        x1 = (((((self.a4*a1)*self.a2)+((V_row[:, self.b2_idx]*b4)*b3))+((self.a2*b4)*b3))+((b3*a1)*self.a2))
        x2 = (((((V_row[:, self.b2_idx]*self.b1)*b4)+((a1*self.a2)*V_row[:, self.a3_idx]))+((V_row[:, self.a3_idx]*self.b1)*b4))+((self.a2*V_row[:, self.a3_idx])*b4))
        x3 = (((((self.a2*V_row[:, self.a3_idx])*self.a4)+((b3*V_row[:, self.b2_idx])*self.b1))+((V_row[:, self.b2_idx]*self.b1)*self.a4))+((V_row[:, self.a3_idx]*self.a4)*self.b1))
        x4 = (((((b4*b3)*V_row[:, self.b2_idx])+((V_row[:, self.a3_idx]*self.a4)*a1))+((V_row[:, self.b2_idx]*self.a4)*a1))+((b3*V_row[:, self.b2_idx])*a1))
        E1 = (x1/(((x1+x2)+x3)+x4))
        E2 = (x2/(((x1+x2)+x3)+x4))
        E3 = (x3/(((x1+x2)+x3)+x4))
        E4 = (x4/(((x1+x2)+x3)+x4))
        ICaK = (ICaK_ss+ICaK_i)
        ICaL = (ICaL_ss+ICaL_i)
        ICaNa = (ICaNa_ss+ICaNa_i)
        K4p_i = ((h3_i*self.wCa)/V_row[:, self.hCa_idx])
        K4p_ss = ((h3_ss*self.wCa)/V_row[:, self.hCa_idx])
        K4pp_i = (h2_i*self.wNaCa)
        K4pp_ss = (h2_ss*self.wNaCa)
        K7_i = ((h5_i*h2_i)*self.wNa)
        K7_ss = ((h5_ss*h2_ss)*self.wNa)
        JNaKK = (2.*((E4*self.b1)-((E3*a1))))
        JNaKNa = (3.*((E1*V_row[:, self.a3_idx])-((E2*b3))))
        K4_i = (K4p_i+K4pp_i)
        K4_ss = (K4p_ss+K4pp_ss)
        INaK = (PNaK*((self.zNa*JNaKNa)+(self.zK*JNaKK)))
        x1_i = (((self.K2_i*K4_i)*(K7_i+K6_i))+((self.K5_i*K7_i)*(self.K2_i+V_row[:, self.K3_i_idx])))
        x1_ss = (((self.K2_ss*K4_ss)*(K7_ss+K6_ss))+((self.K5_ss*K7_ss)*(self.K2_ss+V_row[:, self.K3_ss_idx])))
        x2_i = (((self.K1_i*K7_i)*(K4_i+self.K5_i))+((K4_i*K6_i)*(self.K1_i+V_row[:, self.K8_i_idx])))
        x2_ss = (((self.K1_ss*K7_ss)*(K4_ss+self.K5_ss))+((K4_ss*K6_ss)*(self.K1_ss+V_row[:, self.K8_ss_idx])))
        x3_i = (((self.K1_i*V_row[:, self.K3_i_idx])*(K7_i+K6_i))+((V_row[:, self.K8_i_idx]*K6_i)*(self.K2_i+V_row[:, self.K3_i_idx])))
        x3_ss = (((self.K1_ss*V_row[:, self.K3_ss_idx])*(K7_ss+K6_ss))+((V_row[:, self.K8_ss_idx]*K6_ss)*(self.K2_ss+V_row[:, self.K3_ss_idx])))
        x4_i = (((self.K2_i*V_row[:, self.K8_i_idx])*(K4_i+self.K5_i))+((V_row[:, self.K3_i_idx]*self.K5_i)*(self.K1_i+V_row[:, self.K8_i_idx])))
        x4_ss = (((self.K2_ss*V_row[:, self.K8_ss_idx])*(K4_ss+self.K5_ss))+((V_row[:, self.K3_ss_idx]*self.K5_ss)*(self.K1_ss+V_row[:, self.K8_ss_idx])))
        E1_i = (x1_i/(((x1_i+x2_i)+x3_i)+x4_i))
        E1_ss = (x1_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss))
        E2_i = (x2_i/(((x1_i+x2_i)+x3_i)+x4_i))
        E2_ss = (x2_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss))
        E3_i = (x3_i/(((x1_i+x2_i)+x3_i)+x4_i))
        E3_ss = (x3_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss))
        E4_i = (x4_i/(((x1_i+x2_i)+x3_i)+x4_i))
        E4_ss = (x4_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss))
        JncxCa_i = ((E2_i*self.K2_i)-((E1_i*self.K1_i)))
        JncxCa_ss = ((E2_ss*self.K2_ss)-((E1_ss*self.K1_ss)))
        JncxNa_i = (((3.*((E4_i*K7_i)-((E1_i*V_row[:, self.K8_i_idx]))))+(E3_i*K4pp_i))-((E2_i*V_row[:, self.K3pp_i_idx])))
        JncxNa_ss = (((3.*((E4_ss*K7_ss)-((E1_ss*V_row[:, self.K8_ss_idx]))))+(E3_ss*K4pp_ss))-((E2_ss*V_row[:, self.K3pp_ss_idx])))
        INaCa_i = ((((1.-(self.INaCa_fractionSS))*Gncx)*Cai_row[:, self.allo_i_idx])*((self.zNa*JncxNa_i)+(self.zCa*JncxCa_i)))
        INaCa_ss = (((self.INaCa_fractionSS*Gncx)*allo_ss)*((self.zNa*JncxNa_ss)+(self.zCa*JncxCa_ss)))
        Iion = ((((((((((((((((((INa+INaL)+Ito)+ICaL)+ICaNa)+ICaK)+IKr)+IKs)+IK1)+INaCa_i)+INaCa_ss)+INaK)+INab)+IKb)+Cai_row[:, self.IpCa_idx])+ICab)+IClCa)+V_row[:, self.IClb_idx])+IKATP)
        
        # Complete Forward Euler Update
        BCajsr = (1./(1.+((self.csqnmax*self.Kmcsqn)/((self.Kmcsqn+self.Cajsr)*(self.Kmcsqn+self.Cajsr)))))
        BCass = (1./((1.+((self.BSRmax*self.KmBSR)/((self.KmBSR+self.Cass)*(self.KmBSR+self.Cass))))+((self.BSLmax*self.KmBSL)/((self.KmBSL+self.Cass)*(self.KmBSL+self.Cass)))))
        Jdiff = ((self.Cass-(self.Cai))/self.tauCa)
        JdiffK = ((self.Kss-(self.Ki))/self.tauK)
        JdiffNa = ((self.Nass-(self.Nai))/self.tauNa)
        JleaK = ((0.0048825*self.Cansr)/15.)
        Jrel_inf_b = (((-self.a_rel)*ICaL_ss)/(1.+((((((((self.Cajsr_half/self.Cajsr)*(self.Cajsr_half/self.Cajsr))*(self.Cajsr_half/self.Cajsr))*(self.Cajsr_half/self.Cajsr))*(self.Cajsr_half/self.Cajsr))*(self.Cajsr_half/self.Cajsr))*(self.Cajsr_half/self.Cajsr))*(self.Cajsr_half/self.Cajsr))))
        Jrel_infp_b = (((-self.a_relp)*ICaL_ss)/(1.+((((((((self.Cajsr_half/self.Cajsr)*(self.Cajsr_half/self.Cajsr))*(self.Cajsr_half/self.Cajsr))*(self.Cajsr_half/self.Cajsr))*(self.Cajsr_half/self.Cajsr))*(self.Cajsr_half/self.Cajsr))*(self.Cajsr_half/self.Cajsr))*(self.Cajsr_half/self.Cajsr))))
        Jtr = ((self.Cansr-(self.Cajsr))/60.)
        Km2n = self.jCa
        diff_CaMKt = (((self.aCaMK*CaMKb)*(CaMKb+self.CaMKt))-((self.bCaMK*self.CaMKt)))
        fJrelp = (1./(1.+(self.KmCaMK/CaMKa)))
        fJupp = (1./(1.+(self.KmCaMK/CaMKa)))
        tau_rel_b = (self.bt/(1.+(0.0123/self.Cajsr)))
        tau_relp_b = (self.btp/(1.+(0.0123/self.Cajsr)))
        Jrel = (self.Jrel_b*(((1.-(fJrelp))*self.Jrel_np)+(fJrelp*self.Jrel_p)))
        Jrel_inf = ((Jrel_inf_b*1.7) if (self.celltype==2.) else Jrel_inf_b)
        Jrel_infp = ((Jrel_infp_b*1.7) if (self.celltype==2.) else Jrel_infp_b)
        Jup = (self.Jup_b*((((1.-(fJupp))*Cai_row[:, self.Jupnp_idx])+(fJupp*Cai_row[:, self.Jupp_idx]))-(JleaK)))
        anCa_i = (1./((self.K2n/Km2n)+((((1.+(self.Kmn/self.Cai))*(1.+(self.Kmn/self.Cai)))*(1.+(self.Kmn/self.Cai)))*(1.+(self.Kmn/self.Cai)))))
        anCa_ss = (1./((self.K2n/Km2n)+((((1.+(self.Kmn/self.Cass))*(1.+(self.Kmn/self.Cass)))*(1.+(self.Kmn/self.Cass)))*(1.+(self.Kmn/self.Cass)))))
        diff_Ki = ((((-(((((((Ito+IKr)+IKs)+IK1)+IKb)+IKATP)-((2.*INaK)))+ICaK_i))*self.ACap)/(self.F*self.Vmyo))+((JdiffK*self.Vss)/self.Vmyo))
        diff_Kss = ((((-ICaK_ss)*self.ACap)/(self.F*self.Vss))-(JdiffK))
        diff_Nai = ((((-(((((INa+INaL)+(3.*INaCa_i))+ICaNa_i)+(3.*INaK))+INab))*self.ACap)/(self.F*self.Vmyo))+((JdiffNa*self.Vss)/self.Vmyo))
        diff_Nass = ((((-(ICaNa_ss+(3.*INaCa_ss)))*self.ACap)/(self.F*self.Vss))-(JdiffNa))
        diff_a = ((V_row[:, self.ass_idx]-(self.a))/V_row[:, self.ta_idx])
        diff_ap = ((V_row[:, self.assp_idx]-(self.ap))/V_row[:, self.ta_idx])
        diff_d = ((V_row[:, self.dss_idx]-(self.d))/V_row[:, self.td_idx])
        diff_ff = ((V_row[:, self.fss_idx]-(self.ff))/V_row[:, self.tff_idx])
        diff_fs = ((V_row[:, self.fss_idx]-(self.fs))/V_row[:, self.tfs_idx])
        diff_hL = ((V_row[:, self.hLss_idx]-(self.hL))/self.thL)
        diff_hLp = ((V_row[:, self.hLssp_idx]-(self.hLp))/self.thLp)
        diff_jCa = ((V_row[:, self.jCass_idx]-(self.jCa))/self.tjCa)
        diff_m = ((V_row[:, self.mss_idx]-(self.m))/V_row[:, self.tm_idx])
        diff_mL = ((V_row[:, self.mLss_idx]-(self.mL))/V_row[:, self.tmL_idx])
        diff_xs1 = ((V_row[:, self.xs1ss_idx]-(self.xs1))/V_row[:, self.txs1_idx])
        tau_rel = (torch.where((tau_rel_b<0.001), 0.001, tau_rel_b))
        tau_relp = (torch.where((tau_relp_b<0.001), 0.001, tau_relp_b))
        diff_C2 = (((V_row[:, self.alpha_idx]*self.C3)+(self.beta_1*self.C1))-(((V_row[:, self.beta_idx]+self.alpha_1)*self.C2)))
        diff_C3 = ((V_row[:, self.beta_idx]*self.C2)-((V_row[:, self.alpha_idx]*self.C3)))
        diff_Cai = (Cai_row[:, self.BCai_idx]*(((((-(((ICaL_i+Cai_row[:, self.IpCa_idx])+ICab)-((2.*INaCa_i))))*self.ACap)/((2.*self.F)*self.Vmyo))-(((Jup*self.Vnsr)/self.Vmyo)))+((Jdiff*self.Vss)/self.Vmyo)))
        diff_Cajsr = (BCajsr*(Jtr-(Jrel)))
        diff_Cansr = (Jup-(((Jtr*self.Vjsr)/self.Vnsr)))
        diff_Cass = (BCass*(((((-(ICaL_ss-((2.*INaCa_ss))))*self.ACap)/((2.*self.F)*self.Vss))+((Jrel*self.Vjsr)/self.Vss))-(Jdiff)))
        diff_Jrel_np = ((Jrel_inf-(self.Jrel_np))/tau_rel)
        diff_Jrel_p = ((Jrel_infp-(self.Jrel_p))/tau_relp)
        diff_O = (((V_row[:, self.alpha_2_idx]*self.C1)+(V_row[:, self.beta_i_idx]*self.I))-(((V_row[:, self.beta_2_idx]+V_row[:, self.alpha_i_idx])*self.O)))
        diff_fCaf = ((V_row[:, self.fCass_idx]-(self.fCaf))/V_row[:, self.tfCaf_idx])
        diff_fCafp = ((V_row[:, self.fCass_idx]-(self.fCafp))/V_row[:, self.tfCafp_idx])
        diff_fCas = ((V_row[:, self.fCass_idx]-(self.fCas))/V_row[:, self.tfCas_idx])
        diff_ffp = ((V_row[:, self.fss_idx]-(self.ffp))/V_row[:, self.tffp_idx])
        diff_h = ((V_row[:, self.hss_idx]-(self.h))/V_row[:, self.th_idx])
        diff_hp = ((V_row[:, self.hssp_idx]-(self.hp))/V_row[:, self.th_idx])
        diff_iF = ((V_row[:, self.iss_idx]-(self.iF))/V_row[:, self.tiF_idx])
        diff_iS = ((V_row[:, self.iss_idx]-(self.iS))/V_row[:, self.tiS_idx])
        diff_j = ((V_row[:, self.jss_idx]-(self.j))/V_row[:, self.tj_idx])
        diff_nCa_i = ((anCa_i*self.K2n)-((self.nCa_i*Km2n)))
        diff_nCa_ss = ((anCa_ss*self.K2n)-((self.nCa_ss*Km2n)))
        diff_xs2 = ((V_row[:, self.xs2ss_idx]-(self.xs2))/V_row[:, self.txs2_idx])
        diff_C1 = ((((self.alpha_1*self.C2)+(V_row[:, self.beta_2_idx]*self.O))+(V_row[:, self.beta_ItoC2_idx]*self.I))-((((self.beta_1+V_row[:, self.alpha_2_idx])+V_row[:, self.alpha_C2ToI_idx])*self.C1)))
        diff_I = (((V_row[:, self.alpha_C2ToI_idx]*self.C1)+(V_row[:, self.alpha_i_idx]*self.O))-(((V_row[:, self.beta_ItoC2_idx]+V_row[:, self.beta_i_idx])*self.I)))
        diff_iFp = ((V_row[:, self.iss_idx]-(self.iFp))/V_row[:, self.tiFp_idx])
        diff_iSp = ((V_row[:, self.iss_idx]-(self.iSp))/V_row[:, self.tiSp_idx])
        diff_jp = ((V_row[:, self.jss_idx]-(self.jp))/V_row[:, self.tjp_idx])
        C1_new = self.C1+diff_C1*self.dt
        C2_new = self.C2+diff_C2*self.dt
        C3_new = self.C3+diff_C3*self.dt
        CaMKt_new = self.CaMKt+diff_CaMKt*self.dt
        Cai_new = self.Cai+diff_Cai*self.dt
        Cajsr_new = self.Cajsr+diff_Cajsr*self.dt
        Cansr_new = self.Cansr+diff_Cansr*self.dt
        Cass_new = self.Cass+diff_Cass*self.dt
        I_new = self.I+diff_I*self.dt
        Jrel_np_new = self.Jrel_np+diff_Jrel_np*self.dt
        Jrel_p_new = self.Jrel_p+diff_Jrel_p*self.dt
        Ki_new = self.Ki+diff_Ki*self.dt
        Kss_new = self.Kss+diff_Kss*self.dt
        Nai_new = self.Nai+diff_Nai*self.dt
        Nass_new = self.Nass+diff_Nass*self.dt
        O_new = self.O+diff_O*self.dt
        a_new = self.a+diff_a*self.dt
        ap_new = self.ap+diff_ap*self.dt
        d_new = self.d+diff_d*self.dt
        fCaf_new = self.fCaf+diff_fCaf*self.dt
        fCafp_new = self.fCafp+diff_fCafp*self.dt
        fCas_new = self.fCas+diff_fCas*self.dt
        ff_new = self.ff+diff_ff*self.dt
        ffp_new = self.ffp+diff_ffp*self.dt
        fs_new = self.fs+diff_fs*self.dt
        h_new = self.h+diff_h*self.dt
        hL_new = self.hL+diff_hL*self.dt
        hLp_new = self.hLp+diff_hLp*self.dt
        hp_new = self.hp+diff_hp*self.dt
        iF_new = self.iF+diff_iF*self.dt
        iFp_new = self.iFp+diff_iFp*self.dt
        iS_new = self.iS+diff_iS*self.dt
        iSp_new = self.iSp+diff_iSp*self.dt
        j_new = self.j+diff_j*self.dt
        jCa_new = self.jCa+diff_jCa*self.dt
        jp_new = self.jp+diff_jp*self.dt
        m_new = self.m+diff_m*self.dt
        mL_new = self.mL+diff_mL*self.dt
        nCa_i_new = self.nCa_i+diff_nCa_i*self.dt
        nCa_ss_new = self.nCa_ss+diff_nCa_ss*self.dt
        xs1_new = self.xs1+diff_xs1*self.dt
        xs2_new = self.xs2+diff_xs2*self.dt
        
        # Complete Rush Larsen Update

        # Finish the update
        self.C1 = torch.clamp(C1_new, 0, 1)
        self.C2 = torch.clamp(C2_new, 0, 1)
        self.C3 = torch.clamp(C3_new, 0, 1)
        self.CaMKt = torch.clamp(CaMKt_new, min=1e-9)
        self.Cai = torch.clamp(Cai_new, min=1e-9)
        self.Cajsr = torch.clamp(Cajsr_new, min=1e-9)
        self.Cansr = torch.clamp(Cansr_new, min=1e-9)
        self.Cass = torch.clamp(Cass_new, min=1e-9)
        self.I = torch.clamp(I_new, 0, 1)
        self.Jrel_np = torch.clamp(Jrel_np_new, 0, 1)
        self.Jrel_p = torch.clamp(Jrel_p_new, 0, 1)
        self.Ki = torch.clamp(Ki_new, min=1e-9)
        self.Kss = torch.clamp(Kss_new, min=1e-9)
        self.Nai = torch.clamp(Nai_new, min=1e-9)
        self.Nass = torch.clamp(Nass_new, min=1e-9)
        self.O = torch.clamp(O_new, 0, 1)
        self.a = torch.clamp(a_new, 0, 1)
        self.ap = torch.clamp(ap_new, 0, 1)
        self.d = torch.clamp(d_new, 0, 1)
        self.fCaf = torch.clamp(fCaf_new, 0, 1)
        self.fCafp = torch.clamp(fCafp_new, 0, 1)
        self.fCas = torch.clamp(fCas_new, 0, 1)
        self.ff = torch.clamp(ff_new, 0, 1)
        self.ffp = torch.clamp(ffp_new, 0, 1)
        self.fs = torch.clamp(fs_new, 0, 1)
        self.h = torch.clamp(h_new, 0, 1)
        self.hL = torch.clamp(hL_new, 0, 1)
        self.hLp = torch.clamp(hLp_new, 0, 1)
        self.hp = torch.clamp(hp_new, 0, 1)
        self.iF = torch.clamp(iF_new, 0, 1)
        self.iFp = torch.clamp(iFp_new, 0, 1)
        self.iS = torch.clamp(iS_new, 0, 1)
        self.iSp = torch.clamp(iSp_new, 0, 1)
        self.j = torch.clamp(j_new, 0, 1)
        self.jCa = torch.clamp(jCa_new, 0, 1)
        self.jp = torch.clamp(jp_new, 0, 1)
        self.m = torch.clamp(m_new, 0, 1)
        self.mL = torch.clamp(mL_new, 0, 1)
        self.nCa_i = torch.clamp(nCa_i_new, 0, 1)
        self.nCa_ss = torch.clamp(nCa_ss_new, 0, 1)
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
    ionic = Tomek19(cell_type="ENDO", 
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
    plt.savefig("V_ToRORd.png")

    plt.figure()
    Cai_list = np.array(Cai_list)    
    plt.plot(Cai_list[:, 0], Cai_list[:, 1])
    plt.savefig("Cai.png")
