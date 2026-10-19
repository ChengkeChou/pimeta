"""Meta-analysis toolkit replicating metafor, meta and netmeta in pure Python.

Numeric engine: self-contained NumPy/SciPy transcriptions of the R source —no
third-party meta-analysis package (pymare/statsmodels) is used anywhere.

Submodules
----------
pooling : fixed/random-effects pooling (DL/HE/HS/SJ/ML/REML/PM/EB tau^2).
escalc  : effect-size computation from raw study data (RR/OR/RD/AS/PETO,
          MD/SMD, IRR, proportions, correlations).
tau2    : moment tau-squared estimators (DL/HE/HS/SJ/PM/EB).
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("pimeta")
except PackageNotFoundError:  # importing straight from a source tree
    __version__ = "0.0.0+unknown"

from .pooling import (
    DIFF_TYPES,
    RATIO_TYPES,
    HcResult,
    PooledResult,
    PredictResult,
    StudyEffect,
    Tau2Confint,
    confint_tau2,
    hc_rma,
    pool_yv,
    predict_rma,
    result_to_dict,
    run_meta,
)
from .escalc import EscalcResult, escalc
from .tau2 import estimate_tau2
from .metabin import MetabResult, metabin
from .metacont import MetacontResult, metacont
from .metareg import MetaregResult, metareg
from .publication_bias import (
    BeggResult,
    EggerResult,
    RegtestResult,
    TrimfillResult,
    begg_test,
    egger_test,
    regtest,
    trimfill,
)
from .sensitivity import (
    BaujatResult,
    InfluenceResult,
    LeaveOneOutResult,
    baujat,
    cooks_distance,
    hatvalues,
    influence,
    leave1out,
    rstudent,
)
from .netmeta import (
    DecompDesignResult,
    NetcombResult,
    NetcomplexResult,
    NetconnectionResult,
    NetcontribResult,
    NetdistanceResult,
    NetleagueResult,
    NetmetaResult,
    NetmetabinResult,
    NetpairwiseResult,
    NetrankResult,
    NetsplitResult,
    PairwiseData,
    decomp_design,
    discomb,
    netcomb,
    netcomplex,
    netcomplexity,
    netmetareg,
    netmeasures,
    netmatrix,
    NetmatrixResult,
    netcomparison,
    NetcomparisonResult,
    netbind,
    crossnma2netmeta,
    gemtc2netmeta,
    treats,
    vikor,
    VikorResult,
    subgroup,
    NetsubgroupResult,
    nettable,
    NettableResult,
    createC,
    invmat,
    hatmatrix,
    comps,
    combinations,
    NetbindResult,
    netimpact,
    netpath,
    NetmetaregResult,
    NetmeasuresResult,
    NetimpactResult,
    netconnection,
    netcontrib,
    netdistance,
    netleague,
    netmeta,
    netmetabin,
    netpairwise,
    netrank,
    netsplit,
    pairwise,
)
from .cumul import CumulResult, cumul
from .fsn import FsnResult, fsn
from .gosh import GoshResult, gosh
from .metacor import MetacorResult, metacor
from .metainc import MetaincResult, metainc
from .metagen import MetagenResult, backtransf, metagen, transf
from .metamean import MetameanResult, metamean
from .metaprop import MetapropResult, metaprop
from .metarate import MetarateResult, metarate
from .metainf import MetacumResult, MetainfResult, metacum, metainf
from .rma_mv import RmaMvResult, anova_rma, rma_mv, ranef, robust
from .rma_peto_mh import RmaMhResult, RmaPetoResult, rma_mh, rma_peto
from .rma_glmm import RmaGlmmResult, rma_glmm
from .rma_uni import RmaResult, rma
from .selmodel import SelModelResult, selmodel
from .plotting import (
    PlotResult, bubble, drapery, forest, funnel, hasse, labbe, llplot,
    netgraph, netheat, netposet, plot_baujat, plot_gosh, qqnorm_rma, radial,
    rankogram, traffic_light,
    heatplot,
    regplot,)
from .fmt import dfround, fmtp, fmtp2, fmtt, fmtx
from .diagnostics import (
    DeltaMethodResult, PermutestResult, TesResult, VifResult,
    deltamethod, permutest, tes, vif,
)
from .measures import (
    NntResult, RdResult, asin2ir, asin2p, cor2z, logVR2VE, logit2p, nnt,
    or2smd, p2asin, p2logit, rd, smd2or, VE2logVR, z2cor,
)
from .extras import (
    HettestResult, MatregResult, ProfileResult, hettest, matreg, profile_rma,
    simulate_rma, vcalc,
)
from .conv import (
    ConvDeltaResult, ConvFivenumResult, ConvWaldResult, conv_delta,
    conv_fivenum, conv_wald, conv_2x2, Conv2x2Result,
    to_long, to_table, to_wide,
)
from .s3 import (
    coef, vcov, fitted, residuals, logLik, deviance, nobs, formula,
    AIC, BIC, weights, sigma, df_residual, model_matrix, simulate,
    confint, predict, hatvalues_rma, dfbetas_rma, rstandard_rma,
    ranef_rma, summary, summary_text, print_summary, plot,
)
from .report import (
    aggregate_escalc, cbind_escalc, rbind_escalc, emmprep, addpoly,
    points_regplot, reporter, cidprop, cilayout, forest_dims, longarm,
    rob, format_ci, CidpropResult, LongarmResult, ReportResult,
)
from .readers import (REVIEW_COLUMNS, ReviewDataResult, read_cdir,
                      read_mtv, read_rm5)
from .utils import (
    blup, fitstats, aic_rma, bic_rma, se_rma, df_residual_rma,
    model_matrix_rma, as_data_frame_list_rma, cooks_distance_rma,
    sigma_matreg, update_rma, estimates, bldiag, blsplit, vec2mat,
    as_data_frame_confint, as_matrix_list_rma, se,
    as_data_frame_list_anova, as_data_frame_list_confint,
    as_data_frame_vif, inches2units, units2inches,
    jama_labels, ci, getmfopt, setmfopt,
    replmiss, cmicalc, contrmat, pairmat, rcalc, forest_args,
    RcalcResult, BlupResult, FitstatsResult, EstimatesResult,
)
from .meta_compose import (MetaaddResult, MetabindResult, MetacrResult,
                            metaadd, metabind, metacr, metamerge)
from .settings import (
    META_DEFAULTS, NETMETA_DEFAULTS, gs, setvals, settings_meta,
    settings_netmeta,
)
from . import transforms as _transforms
from .transforms import (
    transf_abt, transf_ahw, transf_arcsin, transf_clestod,
    transf_dtobesd, transf_dtocles, transf_dtocles_int, transf_dtocliffd,
    transf_dtolnor_logis, transf_dtolnor_norm, transf_dtomd, transf_dtoovl,
    transf_dtorbis, transf_dtorpb, transf_dtou1, transf_dtou2, transf_dtou3,
    transf_exp_int, transf_exp_mode, transf_iabt, transf_iabt_int,
    transf_iabt_mode, transf_iahw, transf_iahw_int, transf_iahw_mode,
    transf_iarcsin, transf_iarcsin_int, transf_iarcsin_mode, transf_iirft,
    transf_ilogit, transf_ilogit_int, transf_ilogit_mode, transf_ipft,
    transf_ipft_hm, transf_iprobit, transf_iprobit_int, transf_iprobit_mode,
    transf_irft, transf_isqrt, transf_lnortod_logis, transf_lnortod_norm,
    transf_lnortord, transf_lnortorr, transf_lnortortet_digby,
    transf_lnortortet_pearson, transf_logit, transf_pft, transf_probit,
    transf_r2toz, transf_rpbtod, transf_rpbtorbis, transf_rtod, transf_rtorpb,
    transf_rtoz, transf_ztor, transf_ztor_int, transf_ztor_mode, transf_ztor2,
)

__all__ = [
    "StudyEffect",
    "PooledResult",
    "run_meta",
    "pool_yv",
    "result_to_dict",
    "confint_tau2",
    "Tau2Confint",
    "predict_rma",
    "PredictResult",
    "hc_rma",
    "HcResult",
    "escalc",
    "EscalcResult",
    "estimate_tau2",
    "metabin",
    "MetabResult",
    "metacont",
    "MetacontResult",
    "metareg",
    "MetaregResult",
    "egger_test",
    "EggerResult",
    "begg_test",
    "BeggResult",
    "trimfill",
    "TrimfillResult",
    "influence",
    "InfluenceResult",
    "leave1out",
    "LeaveOneOutResult",
    "hatvalues",
    "cooks_distance",
    "rstudent",
    "netmeta",
    "NetmetaResult",
    "netcomb",
    "NetcombResult",
    "NetcomplexResult",
    "discomb",
    "netmetabin",
    "NetmetabinResult",
    "netpairwise",
    "NetpairwiseResult",
    "pairwise",
    "PairwiseData",
    "netrank",
    "NetrankResult",
    "netsplit",
    "NetsplitResult",
    "netcontrib",
    "NetcontribResult",
    "netleague",
    "NetleagueResult",
    "decomp_design",
    "DecompDesignResult",
    "netconnection",
    "NetconnectionResult",
    "netdistance",
    "NetdistanceResult",
    "rma_glmm",
    "RmaGlmmResult",
    "cumul",
    "CumulResult",
    "fsn",
    "FsnResult",
    "gosh",
    "GoshResult",
    "metacor",
    "MetacorResult",
    "metainc",
    "MetaincResult",
    "metamean",
    "MetameanResult",
    "metagen",
    "MetagenResult",
    "transf",
    "backtransf",
    "metaprop",
    "MetapropResult",
    "metarate",
    "MetarateResult",
    "metainf",
    "MetainfResult",
    "metacum",
    "MetacumResult",
    "regtest",
    "RegtestResult",
    "baujat",
    "BaujatResult",
    "rma_mv",
    "RmaMvResult",
    "anova_rma",
    "ranef",
    "robust",
    "rma_peto",
    "RmaPetoResult",
    "rma_mh",
    "RmaMhResult",
    "rma",
    "RmaResult",
    "selmodel",
    "SelModelResult",
    "forest",
    "funnel",
    "netgraph",
    "plot_baujat",
    "plot_gosh",
    "PlotResult",
    "RATIO_TYPES",
    "DIFF_TYPES",
    # fmt / dfround
    "fmtp", "fmtp2", "fmtt", "fmtx", "dfround",
    # diagnostics
    "deltamethod", "DeltaMethodResult",
    "tes", "TesResult",
    "vif", "VifResult",
    "permutest", "PermutestResult",
    # measure conversions (meta package)
    "asin2ir", "asin2p", "p2asin", "cor2z", "z2cor", "logit2p", "p2logit",
    "smd2or", "or2smd", "logVR2VE", "VE2logVR", "rd", "nnt",
    "RdResult", "NntResult",
    # extras
    "simulate_rma", "vcalc", "hettest", "HettestResult",
    "matreg", "MatregResult", "profile_rma", "ProfileResult",
    "netcomplex",
    "netcomplexity",
    "netmetareg", "NetmetaregResult",
    "netmeasures", "NetmeasuresResult",
    "netmatrix", "NetmatrixResult",
    "netcomparison", "NetcomparisonResult",
    "netbind", "NetbindResult",
    "crossnma2netmeta", "gemtc2netmeta",
    "treats", "vikor", "VikorResult", "subgroup", "NetsubgroupResult",
    "nettable", "NettableResult", "createC", "invmat", "hatmatrix",
    "comps", "combinations",
    "netimpact", "NetimpactResult",
    "netpath",
    "heatplot", "regplot",
    # plotting
    "radial", "labbe", "llplot", "bubble", "drapery", "qqnorm_rma",
    "rankogram", "netheat", "hasse", "netposet", "traffic_light",
    # conv / to.*
    "conv_delta", "conv_wald", "conv_fivenum",
    "conv_2x2", "Conv2x2Result",
    "ConvDeltaResult", "ConvWaldResult", "ConvFivenumResult",
    "to_long", "to_wide", "to_table",
    # composite
    "metabind", "metaadd", "MetaaddResult", "metamerge", "MetabindResult",
    "metacr", "MetacrResult",
    # metafor helpers / accessors
    "bldiag", "blsplit", "vec2mat", "replmiss", "cmicalc",
    "contrmat", "pairmat", "rcalc", "RcalcResult",
    "fitstats", "FitstatsResult", "aic_rma", "bic_rma", "se_rma",
    "df_residual_rma", "model_matrix_rma", "as_data_frame_list_rma",
    "cooks_distance_rma", "sigma_matreg", "update_rma", "blup", "BlupResult",
    "as_data_frame_confint", "as_matrix_list_rma", "se",
    "as_data_frame_list_anova", "as_data_frame_list_confint",
    "as_data_frame_vif", "inches2units", "units2inches",
    "jama_labels", "ci", "getmfopt", "setmfopt",
    # meta helpers
    "estimates", "EstimatesResult", "forest_args",
    # Cochrane review readers
    "read_rm5", "read_cdir", "read_mtv", "ReviewDataResult",
    "REVIEW_COLUMNS",
    # reporting / layout helpers
    "aggregate_escalc", "cbind_escalc", "rbind_escalc", "emmprep",
    "addpoly", "points_regplot", "reporter", "ReportResult",
    "cidprop", "CidpropResult", "cilayout", "format_ci",
    "forest_dims", "longarm", "LongarmResult", "rob",
    # S3-style generics
    "coef", "vcov", "fitted", "residuals", "logLik", "deviance",
    "nobs", "formula", "AIC", "BIC", "weights", "sigma",
    "df_residual", "model_matrix", "simulate", "confint", "predict",
    "hatvalues_rma", "dfbetas_rma", "rstandard_rma", "ranef_rma",
    "summary", "summary_text", "print_summary", "plot",
    # settings
    "gs", "setvals", "settings_meta", "settings_netmeta",
    "META_DEFAULTS", "NETMETA_DEFAULTS",
    # transf.*
    "transf_abt", "transf_ahw", "transf_arcsin", "transf_clestod",
    "transf_dtobesd", "transf_dtocles", "transf_dtocles_int", "transf_dtocliffd",
    "transf_dtolnor_logis", "transf_dtolnor_norm", "transf_dtomd", "transf_dtoovl",
    "transf_dtorbis", "transf_dtorpb", "transf_dtou1", "transf_dtou2",
    "transf_dtou3", "transf_exp_int", "transf_exp_mode",
    "transf_iabt", "transf_iabt_int", "transf_iabt_mode", "transf_iahw",
    "transf_iahw_int", "transf_iahw_mode", "transf_iarcsin", "transf_iarcsin_int",
    "transf_iarcsin_mode", "transf_iirft", "transf_ilogit", "transf_ilogit_int",
    "transf_ilogit_mode", "transf_ipft", "transf_ipft_hm", "transf_iprobit",
    "transf_iprobit_int", "transf_iprobit_mode", "transf_irft", "transf_isqrt",
    "transf_lnortod_logis", "transf_lnortod_norm", "transf_lnortord",
    "transf_lnortorr", "transf_lnortortet_digby", "transf_lnortortet_pearson",
    "transf_logit", "transf_pft", "transf_probit", "transf_r2toz",
    "transf_rpbtod", "transf_rpbtorbis", "transf_rtod", "transf_rtorpb",
    "transf_rtoz", "transf_ztor", "transf_ztor_int", "transf_ztor_mode",
    "transf_ztor2",
]
