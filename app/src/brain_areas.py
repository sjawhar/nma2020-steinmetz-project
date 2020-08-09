# visual cortex
AREA_VISa = "VISa"
AREA_VISam = "VISam"
AREA_VISl = "VISl"
AREA_VISp = "VISp"
AREA_VISpm = "VISpm"
AREA_VISrl = "VISrl"

# thalamus
AREA_CL = "CL"
AREA_LD = "LD"
AREA_LGd = "LGd"
AREA_LH = "LH"
AREA_LP = "LP"
AREA_MD = "MD"
AREA_MG = "MG"
AREA_PO = "PO"
AREA_POL = "POL"
AREA_PT = "PT"
AREA_RT = "RT"
AREA_SPF = "SPF"
AREA_TH = "TH"
AREA_VAL = "VAL"
AREA_VPL = "VPL"
AREA_VPM = "VPM"

# hippocampal
AREA_CA = "CA"
AREA_CA1 = "CA1"
AREA_CA2 = "CA2"
AREA_CA3 = "CA3"
AREA_DG = "DG"
AREA_SUB = "SUB"
AREA_POST = "POST"

# non-visual cortex
AREA_ACA = "ACA"
AREA_AUD = "AUD"
AREA_COA = "COA"
AREA_DP = "DP"
AREA_ILA = "ILA"
AREA_MOp = "MOp"
AREA_MOs = "MOs"
AREA_OLF = "OLF"
AREA_ORB = "ORB"
AREA_ORBm = "ORBm"
AREA_PIR = "PIR"
AREA_PL = "PL"
AREA_SSp = "SSp"
AREA_SSs = "SSs"
AREA_RSP = "RSP"
AREA_TT = "TT"

# midbrain
AREA_APN = "APN"
AREA_IC = "IC"
AREA_MB = "MB"
AREA_MRN = "MRN"
AREA_NB = "NB"
AREA_PAG = "PAG"
AREA_RN = "RN"
AREA_SCs = "SCs"
AREA_SCm = "SCm"
AREA_SCig = "SCig"
AREA_SCsg = "SCsg"
AREA_ZI = "ZI"

# basal ganglia
AREA_ACB = "ACB"
AREA_CP = "CP"
AREA_GPe = "GPe"
AREA_LS = "LS"
AREA_LSc = "LSc"
AREA_LSr = "LSr"
AREA_MS = "MS"
AREA_OT = "OT"
AREA_SNr = "SNr"

# cortical subplate
AREA_SI = "SI"
AREA_BLA = "BLA"
AREA_BMA = "BMA"
AREA_EP = "EP"
AREA_EPd = "EPd"
AREA_MEA = "MEA"

# Areas selected based on selectivity reported in paper
AREAS_VISUAL = [
    AREA_VISp,
    AREA_VISl,
    AREA_VISpm,
    AREA_VISam,
    AREA_CP,
    AREA_LD,
    AREA_SCs,
]

AREAS_MOTOR = [
    AREA_MOp,
    AREA_MOs,
]

AREAS_ACTION = [
    AREA_SSp,
    AREA_SNr,
    AREA_APN,
    AREA_PAG,
    AREA_ZI,
    *AREAS_MOTOR,
]
