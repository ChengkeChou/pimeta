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

from .basis import (
    Basis,
    CurveResult,
    NonlinearTestResult,
    WaldResult,
    default_knots,
    fp,
    nonlinear_test,
    predict_curve,
    rcs,
    wald_test,
)
from .bayes import Posterior, ess_bulk, ess_tail, rhat, sample_log_posterior
from .checks import Diagnosis, Diagnostic, diagnose
from .conv import (
    Conv2x2Result,
    ConvDeltaResult,
    ConvFivenumResult,
    ConvWaldResult,
    conv_2x2,
    conv_delta,
    conv_fivenum,
    conv_wald,
    to_long,
    to_table,
    to_wide,
)
from .copas import (
    CopasBayesResult,
    CopasRegion,
    CopasResult,
    bayes_copas,
    copas,
    copas_fit,
    copas_loglik,
    copas_region,
)
from .cumul import CumulResult, cumul
from .data_compat import (
    FormulaSpec,
    as_columns,
    column_names,
    design_matrix,
    install_repr_html,
    lookup,
    model_frame,
    parse_formula,
    resolve,
    resolve_mods,
    to_dataframe,
    to_html,
)
from .diagnostics import (
    DeltaMethodResult,
    PermutestResult,
    TesResult,
    VifResult,
    deltamethod,
    permutest,
    tes,
    vif,
)
from .dose import DosresResult, dose_cov, dosres, greenland_counts
from .dta import (
    DtaCorrectionWarning,
    DtaData,
    HsrocResult,
    ReitsmaResult,
    SrocResult,
    auc,
    escalc_dta,
    hsroc,
    reitsma,
    sroc,
)
from .dynamichet import (
    BayesTauTrendResult,
    TauTrendResult,
    bayes_tau_trend,
    tau_trend,
)
from .escalc import EscalcResult, escalc
from .extras import (
    HettestResult,
    MatregResult,
    ProfileResult,
    hettest,
    matreg,
    profile_rma,
    simulate_rma,
    vcalc,
)
from .fmt import dfround, fmtp, fmtp2, fmtt, fmtx
from .fsn import FsnResult, fsn
from .gosh import GoshResult, gosh
from .ipd import IpdResult, StudyFit, ipd_two_stage, linear_model, logistic_model
from .ipd_curve import IpdCurve, IpdCurveResult, ipd_two_stage_curve
from .levels import level_to_alpha, level_to_percent
from .limitmeta import LimitmetaResult, limitmeta
from .measures import (
    NntResult,
    RdResult,
    VE2logVR,
    asin2ir,
    asin2p,
    cor2z,
    logit2p,
    logVR2VE,
    nnt,
    or2smd,
    p2asin,
    p2logit,
    rd,
    smd2or,
    z2cor,
)
from .meta_compose import (
    MetaaddResult,
    MetabindResult,
    MetacrResult,
    metaadd,
    metabind,
    metacr,
    metamerge,
)
from .metabin import MetabResult, metabin
from .metacont import MetacontResult, metacont
from .metacor import MetacorResult, metacor
from .metagen import MetagenResult, backtransf, metagen, transf
from .metainc import MetaincResult, metainc
from .metainf import MetacumResult, MetainfResult, metacum, metainf
from .metamean import MetameanResult, metamean
from .metaprop import MetapropResult, metaprop
from .metarate import MetarateResult, metarate
from .metareg import MetaregResult, metareg
from .missing import MissingDataWarning, complete_cases, is_missing
from .missing_policy import (
    STRATEGIES,
    MissingPolicy,
    complete_case_policy,
    policy_from_result,
)
from .multiverse import MultiverseResult, multiverse
from .mv import (
    ConfintMvResult,
    PredictionRegion,
    RhoSensitivityResult,
    TauCorResult,
    confint_rma_mv,
    predict_region,
    rho_sensitivity,
    shared_control_corr,
    tau_cor,
)
from .mv_diagnostics import MvProfileResult, profile_rmamv
from .netmeta import (
    DecompDesignResult,
    NetbindResult,
    NetcombResult,
    NetcomparisonResult,
    NetcomplexResult,
    NetconnectionResult,
    NetcontribResult,
    NetdistanceResult,
    NetimpactResult,
    NetleagueResult,
    NetmatrixResult,
    NetmeasuresResult,
    NetmetabinResult,
    NetmetaregResult,
    NetmetaResult,
    NetpairwiseResult,
    NetrankResult,
    NetsplitResult,
    NetsubgroupResult,
    NettableResult,
    PairwiseData,
    VikorResult,
    combinations,
    comps,
    createC,
    crossnma2netmeta,
    decomp_design,
    discomb,
    gemtc2netmeta,
    hatmatrix,
    invmat,
    netbind,
    netcomb,
    netcomparison,
    netcomplex,
    netcomplexity,
    netconnection,
    netcontrib,
    netdistance,
    netimpact,
    netleague,
    netmatrix,
    netmeasures,
    netmeta,
    netmetabin,
    netmetareg,
    netpairwise,
    netpath,
    netrank,
    netsplit,
    nettable,
    pairwise,
    subgroup,
    treats,
    vikor,
)
from .nma.bayes import BayesNmaResult, bayes_nma
from .nma.design import (
    MultiArmStructure,
    TreatmentNetwork,
    consistency_residuals,
    multiarm_structure,
)
from .nma.dose import NmaDoseCurve, NmaDoseResult, nma_dose
from .nma.ipd import IpdNmaResult, ipd_nma
from .nma.timecourse import (
    TimeCourseNmaCurve,
    TimeCourseNmaResult,
    nma_timecourse,
)
from .onestage import OneStageResult, ipd_one_stage
from .plotting import (
    PlotResult,
    bubble,
    drapery,
    forest,
    funnel,
    hasse,
    heatplot,
    labbe,
    llplot,
    netgraph,
    netheat,
    netposet,
    plot_baujat,
    plot_gosh,
    plot_influence,
    plot_mv_profile,
    plot_permutest,
    plot_selprofile,
    plot_specification_curve,
    plot_sroc,
    qqnorm_rma,
    radial,
    rankogram,
    regplot,
    traffic_light,
)
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
from .prediction import PredictionResult, auc_logit, oe_ratio, prediction_meta
from .prediction_joint import PredictionJointResult, prediction_joint
from .publication_bias import (
    BeggResult,
    EggerResult,
    MetabiasResult,
    RegtestResult,
    TrimfillResult,
    begg_test,
    egger_test,
    metabias,
    peters_test,
    regtest,
    trimfill,
)
from .readers import REVIEW_COLUMNS, ReviewDataResult, read_cdir, read_mtv, read_rm5
from .record import (
    RECORD_SCHEMA,
    STORE_SCHEMA,
    AnalysisRecord,
    EstimatesStore,
    RecordError,
    RecordMismatch,
    StoredModel,
    analysis_fingerprint,
    build_record,
    compare_results,
    environment,
    estimates_store,
    fingerprint,
    load_record,
    read_table,
    save_record,
    source_fingerprint,
    write_table,
)
from .report import (
    CidpropResult,
    LongarmResult,
    ReportResult,
    addpoly,
    aggregate_escalc,
    cbind_escalc,
    cidprop,
    cilayout,
    emmprep,
    forest_dims,
    format_ci,
    longarm,
    points_regplot,
    rbind_escalc,
    reporter,
    rob,
)
from .rma_glmm import RmaGlmmResult, rma_glmm
from .rma_list import (
    as_data_frame_rma_list,
    head,
    head_rma,
    slice_rma,
    tail,
    tail_rma,
)
from .rma_ls import (
    RmaLsProfileResult,
    RmaLsResult,
    confint_rma_ls,
    profile_rma_ls,
    rma_ls,
)
from .rma_mv import RmaMvResult, anova_rma, ranef, rma_mv, robust
from .rma_peto_mh import RmaMhResult, RmaPetoResult, rma_mh, rma_peto
from .rma_uni import RmaResult, rma
from .robma import (
    RobmaResult,
    WeightFunction,
    robma,
    step_weight_function,
)
from .s3 import (
    AIC,
    BIC,
    coef,
    confint,
    deviance,
    df_residual,
    dfbetas_rma,
    fitted,
    formula,
    hatvalues_rma,
    logLik,
    model_matrix,
    nobs,
    plot,
    predict,
    print_summary,
    ranef_rma,
    residuals,
    rstandard_rma,
    sigma,
    simulate,
    summary,
    summary_text,
    vcov,
    weights,
)
from .selmodel import SelModelResult, SelProfileResult, profile_selmodel, selmodel
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
from .sensitivity_table import (
    SensitivityResult,
    SensitivityRow,
    sensitivity_table,
)
from .settings import (
    META_DEFAULTS,
    NETMETA_DEFAULTS,
    gs,
    settings_meta,
    settings_netmeta,
    setvals,
)
from .spec import (
    AnalysisSpec,
    Decision,
    RunOutcome,
    Specification,
    execute,
)
from .surv import (
    KmResult,
    ReconstructedIpD,
    RmstResult,
    km_curve,
    km_reconstruct,
    rmst,
)
from .tau2 import estimate_tau2
from .timecourse import TimeCourseCurve, TimeCourseResult, timecourse
from .transforms import (
    transf_abt,
    transf_ahw,
    transf_arcsin,
    transf_clestod,
    transf_dtobesd,
    transf_dtocles,
    transf_dtocles_int,
    transf_dtocliffd,
    transf_dtolnor_logis,
    transf_dtolnor_norm,
    transf_dtomd,
    transf_dtoovl,
    transf_dtorbis,
    transf_dtorpb,
    transf_dtou1,
    transf_dtou2,
    transf_dtou3,
    transf_exp_int,
    transf_exp_mode,
    transf_iabt,
    transf_iabt_int,
    transf_iabt_mode,
    transf_iahw,
    transf_iahw_int,
    transf_iahw_mode,
    transf_iarcsin,
    transf_iarcsin_int,
    transf_iarcsin_mode,
    transf_iirft,
    transf_ilogit,
    transf_ilogit_int,
    transf_ilogit_mode,
    transf_ipft,
    transf_ipft_hm,
    transf_iprobit,
    transf_iprobit_int,
    transf_iprobit_mode,
    transf_irft,
    transf_isqrt,
    transf_lnortod_logis,
    transf_lnortod_norm,
    transf_lnortord,
    transf_lnortorr,
    transf_lnortortet_digby,
    transf_lnortortet_pearson,
    transf_logit,
    transf_pft,
    transf_probit,
    transf_r2toz,
    transf_rpbtod,
    transf_rpbtorbis,
    transf_rtod,
    transf_rtorpb,
    transf_rtoz,
    transf_ztor,
    transf_ztor2,
    transf_ztor_int,
    transf_ztor_mode,
)
from .tvhr import (
    PiecewiseJointResult,
    PiecewiseResult,
    piecewise_exp,
    piecewise_joint,
    poisson_irls,
    split_followup,
)
from .utils import (
    BlupResult,
    EstimatesResult,
    FitstatsResult,
    RcalcResult,
    aic_rma,
    as_data_frame_confint,
    as_data_frame_list_anova,
    as_data_frame_list_confint,
    as_data_frame_list_rma,
    as_data_frame_vif,
    as_matrix_list_rma,
    bic_rma,
    bldiag,
    blsplit,
    blup,
    ci,
    cmicalc,
    contrmat,
    cooks_distance_rma,
    df_residual_rma,
    estimates,
    fitstats,
    forest_args,
    getmfopt,
    inches2units,
    jama_labels,
    model_matrix_rma,
    pairmat,
    rcalc,
    replmiss,
    se,
    se_rma,
    setmfopt,
    sigma_matreg,
    units2inches,
    update_rma,
    vec2mat,
)
from .validation import (
    SCENARIOS,
    Dataset,
    Scenario,
    run_scenario,
    simulate_study,
    summarise_fits,
)

# metafor exposes ``dfbetas``/``rstandard`` as S3 generics that pick up a model
# object; the same short names are handy here.
dfbetas = dfbetas_rma
rstandard = rstandard_rma


# --------------------------------------------------------------------------- #
# R's S3 method names
# --------------------------------------------------------------------------- #
# R spells its methods ``generic.class`` (``print.rma.uni``,
# ``cooks.distance.rma.mv``, ``[.list.rma``, ...).  Those strings can never be
# written as Python *identifiers* (``.`` would be attribute access, ``[``/``$``
# are syntax), so pimeta uses the R-independent names on the left of this map.
# The map lets ``getattr(pimeta, "cooks.distance.rma.mv")`` - which is what a
# mechanical R -> Python translation produces - and `pimeta.r_methods()` resolve
# to the implementing callable.
_R_METHODS = {
    # printing / summaries
    "print.rma.uni": "summary",
    "print.rma.mv": "summary",
    "print.rma.mh": "summary",
    "print.rma.peto": "summary",
    "print.rma.glmm": "summary",
    "print.rma.ls": "summary",
    "print.summary.rma": "print_summary",
    "print.anova.rma": "print_summary",
    "print.confint.rma": "as_data_frame_confint",
    "print.confint.matreg": "as_data_frame_confint",
    "print.list.rma": "print_summary",
    "print.list.confint.rma": "as_data_frame_list_confint",
    "print.list.anova.rma": "as_data_frame_list_anova",
    "print.matreg": "summary",
    "print.escalc": "print_summary",
    "print.regtest": "print_summary",
    "print.ranktest": "print_summary",
    "print.fsn": "print_summary",
    "print.gosh.rma": "print_summary",
    "print.hc.rma.uni": "print_summary",
    "print.hettest": "print_summary",
    "print.infl.rma.uni": "print_summary",
    "print.permutest.rma.uni": "print_summary",
    "print.profile.rma": "print_summary",
    "print.tes": "print_summary",
    "print.vcovmat": "print_summary",
    "print.vif.rma": "as_data_frame_vif",
    "print.deltamethod": "print_summary",
    # diagnostics that live under a different module in pimeta
    "cooks.distance.rma.mv": "cooks_distance",
    "hc.rma.uni": "hc_rma",
    "profile.rma.mv": "profile_rmamv",
    "profile.rma.ls": "profile_rma_ls",
    "profile.rma.uni.selmodel": "profile_selmodel",
    "profile.rma.uni": "profile_rma",
    "qqnorm.rma.mv": "qqnorm_rma",
    "qqnorm.rma.mh": "qqnorm_rma",
    "qqnorm.rma.peto": "qqnorm_rma",
    "qqnorm.rma.glmm": "qqnorm_rma",
    "qqnorm.rma.uni": "qqnorm_rma",
    "confint.rma.ls": "confint_rma_ls",
    "confint.rma.mv": "confint",
    "confint.rma.mh": "confint",
    "confint.rma.peto": "confint",
    "confint.rma.uni": "confint",
    # list-like results ([, head, tail, as.data.frame, $<-)
    "[.list.rma": "slice_rma",
    "[.escalc": "slice_rma",
    "head.list.rma": "head_rma",
    "tail.list.rma": "tail_rma",
    "$<-.list.rma": "update_rma",
    "$<-.escalc": "update_rma",
    "as.data.frame.list.rma": "as_data_frame_list_rma",
    "as.data.frame.confint.rma": "as_data_frame_confint",
    "as.data.frame.list.confint.rma": "as_data_frame_list_confint",
    "as.data.frame.list.anova.rma": "as_data_frame_list_anova",
    "as.data.frame.vif.rma": "as_data_frame_vif",
    # generics
    "ranktest.rma": "begg_test",
    "regtest.rma": "regtest",
    "anova.rma.uni": "anova_rma",
    "anova.rma.mv": "anova_rma",
    "permutest.rma.uni": "permutest",
    "permutest.rma.ls": "permutest",
    "predict.rma.uni": "predict_rma",
    "predict.rma.ls": "predict_rma",
    "blup.rma.uni": "blup",
    "ranef.rma.uni": "ranef",
    "robust.rma.mv": "robust",
}

# Only advertise entries whose target really exists, so the map can never
# silently point at a name that was renamed later.
_R_METHODS = {k: v for k, v in _R_METHODS.items() if v in globals()}


def r_methods() -> dict:
    """Return the ``R S3 method name -> pimeta callable`` map.

    R writes its methods as ``generic.class`` (``print.rma.uni``,
    ``cooks.distance.rma.mv``, ``[.list.rma``).  Those strings are not valid
    Python identifiers, so pimeta exports the R-independent names (``summary``,
    ``cooks_distance``, ``slice_rma``).  This map bridges the two, and
    ``getattr(pimeta, "<r name>")`` resolves through :func:`__getattr__`.
    """
    return dict(_R_METHODS)


def __getattr__(name):
    target = _R_METHODS.get(name)
    if target is not None:
        return globals()[target]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Give every result object a compact HTML table in Jupyter/IPython without
# repeating the mixin in ~40 dataclass definitions.
_repr_html_classes = install_repr_html(globals())

# Ecosystem interoperability lives in ``pimeta.interop``: the sklearn estimator
# protocol, broom-style tidy tables, a statsmodels-shaped result view, frame
# backends and labelled graph/array output.  Importing it here only attaches
# cheap methods; the optional third-party libraries stay lazy.
from . import interop  # noqa: E402

_interop_classes = interop.install_methods(globals())


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
    "Basis",
    "CurveResult",
    "NonlinearTestResult",
    "WaldResult",
    "default_knots",
    "DosresResult",
    "dose_cov",
    "dosres",
    "greenland_counts",
    "IpdResult",
    "StudyFit",
    "ipd_two_stage",
    "IpdCurve",
    "IpdCurveResult",
    "ipd_two_stage_curve",
    "linear_model",
    "logistic_model",
    "OneStageResult",
    "ipd_one_stage",
    "PredictionResult",
    "auc_logit",
    "oe_ratio",
    "prediction_meta",
    "prediction_joint",
    "PredictionJointResult",
    "KmResult",
    "ReconstructedIpD",
    "RmstResult",
    "km_curve",
    "km_reconstruct",
    "rmst",
    # time-varying hazard ratios
    "piecewise_exp",
    "timecourse",
    "TimeCourseResult",
    "TimeCourseCurve",
    "piecewise_joint",
    "PiecewiseJointResult",
    "PiecewiseResult",
    "poisson_irls",
    "split_followup",
    "fp",
    "nonlinear_test",
    "predict_curve",
    "rcs",
    "wald_test",
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
    "TreatmentNetwork",
    "MultiArmStructure",
    "multiarm_structure",
    "consistency_residuals",
    "ipd_nma",
    "IpdNmaResult",
    "nma_dose",
    "NmaDoseResult",
    "NmaDoseCurve",
    "nma_timecourse",
    "TimeCourseNmaResult",
    "TimeCourseNmaCurve",
    "bayes_nma",
    "BayesNmaResult",
    "tau_trend",
    "TauTrendResult",
    "bayes_tau_trend",
    "BayesTauTrendResult",
    "sample_log_posterior",
    "Posterior",
    "rhat",
    "ess_bulk",
    "ess_tail",
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
    "limitmeta",
    "LimitmetaResult",
    "copas",
    "CopasResult",
    "copas_fit",
    "copas_loglik",
    "copas_region",
    "CopasRegion",
    "bayes_copas",
    "CopasBayesResult",
    "diagnose",
    "Diagnostic",
    "Diagnosis",
    "robma",
    "RobmaResult",
    "step_weight_function",
    "WeightFunction",    "level_to_alpha",
    "level_to_percent",
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
    "metabias",
    "MetabiasResult",
    "peters_test",
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
    "SelProfileResult",
    "profile_selmodel",
    "MvProfileResult",
    "profile_rmamv",
    "plot_mv_profile",
    "rma_ls",
    "RmaLsResult",
    "profile_rma_ls",
    "RmaLsProfileResult",
    "confint_rma_ls",
    "slice_rma",
    "head_rma",
    "tail_rma",
    "head",
    "tail",
    "as_data_frame_rma_list",
    "forest",
    "funnel",
    "netgraph",
    "plot_baujat", "plot_specification_curve",
    "plot_gosh",
    "plot_sroc",
    "plot_influence",
    "plot_permutest",
    "plot_selprofile",
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
    # validation: known-truth simulation and its calibration summaries
    "SCENARIOS", "Scenario", "Dataset", "simulate_study", "summarise_fits",
    "run_scenario",
    # sensitivity: one estimand under each assumption that could differ
    "sensitivity_table", "SensitivityResult", "SensitivityRow",
    # the analysis specification: declared decisions and their grid
    "AnalysisSpec", "Decision", "Specification", "RunOutcome", "execute",
    # stability across that grid
    "multiverse", "MultiverseResult",
    # declared missing-data handling
    "MissingPolicy", "STRATEGIES", "complete_case_policy",
    "policy_from_result",
    # multivariate reporting layer
    "shared_control_corr", "tau_cor", "TauCorResult",
    "rho_sensitivity", "RhoSensitivityResult",
    "confint_rma_mv", "ConfintMvResult",
    "predict_region", "PredictionRegion",
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
    # metafor names these generics without the "_rma" suffix once a model is
    # given; the aliases below make ``pimeta.dfbetas(res)`` work like R.
    "dfbetas", "rstandard",
    # diagnostic test accuracy
    "DtaData", "DtaCorrectionWarning", "ReitsmaResult", "HsrocResult", "SrocResult",
    "escalc_dta", "reitsma", "hsroc", "sroc", "auc",
    # settings
    "gs", "setvals", "settings_meta", "settings_netmeta",
    "META_DEFAULTS", "NETMETA_DEFAULTS",
    # ecosystem interoperability
    "interop",
    # standardised, traceable input/output
    "record",
    "RECORD_SCHEMA", "STORE_SCHEMA",
    "AnalysisRecord", "build_record", "save_record", "load_record",
    "compare_results", "environment", "RecordError", "RecordMismatch",
    "fingerprint", "source_fingerprint", "analysis_fingerprint",
    "EstimatesStore", "StoredModel", "estimates_store",
    "read_table", "write_table",
    # data-frame interoperability
    "as_columns", "column_names", "lookup", "resolve", "resolve_mods",
    "FormulaSpec", "parse_formula", "design_matrix", "model_frame",
    "to_dataframe", "to_html", "install_repr_html",
    # missing values
    "MissingDataWarning", "complete_cases", "is_missing",
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
