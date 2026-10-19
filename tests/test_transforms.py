"""R-pinned tests for the ``metafor::transf.*`` family (``transforms.py``).

Every expected value below was produced by the corresponding
``metafor:::transf.*`` call on metafor 5.2-1 / R 4.6.1, with

    pr = c(0.2, 0.5, 0.8)      zz = c(-0.5, 0.0, 0.8)
    n1 = c(50, 60, 70)         n2 = c(55, 65, 75)
    targs = list(tau2 = 0.1)
"""
import numpy as np
import pytest

import pimeta as ma

PR = [0.2, 0.5, 0.8]
ZZ = [-0.5, 0.0, 0.8]
N1 = [50, 60, 70]
N2 = [55, 65, 75]
TARGS = {"tau2": 0.1}

#: name -> (python kwargs, argument, expected, tolerance)
CASES = {
    # plain transforms
    "transf_logit": ({}, PR, [-1.3862943611198906, 0.0, 1.3862943611198908]),
    "transf_probit": ({}, PR, [-0.84162123357291418, 0.0, 0.84162123357291441]),
    "transf_arcsin": ({}, PR, [0.46364760900080609, 0.78539816339744839,
                               1.1071487177940904]),
    "transf_ilogit": ({}, ZZ, [0.37754066879814541, 0.5, 0.6899744811276125]),
    "transf_iprobit": ({}, ZZ, [0.30853753872598694, 0.5, 0.78814460141660336]),
    "transf_iarcsin": ({}, ZZ, [0.0, 0.0, 0.51459976115064443]),
    "transf_iabt": ({}, ZZ, [0.0, 0.0, 0.55067103588277844]),
    "transf_iahw": ({}, ZZ, [0.0, 0.0, 0.99199999999999999]),
    "transf_isqrt": ({}, ZZ, [0.0, 0.0, 0.64000000000000012]),
    "transf_abt": ({}, PR, [0.22314355131420971, 0.69314718055994529,
                            1.6094379124341005]),
    "transf_ahw": ({}, PR, [0.071682233277444163, 0.20629947401590021,
                            0.41519645235742686]),
    "transf_rtoz": ({}, [0.2, 0.5, -0.3],
                    [0.20273255405408219, 0.54930614433405489,
                     -0.30951960420311175]),
    "transf_ztor": ({}, ZZ, [-0.46211715726000974, 0.0, 0.66403677026784891]),
    "transf_ztor2": ({}, ZZ, [0.21355226703407257, 0.0, 0.44094483226775594]),
    "transf_r2toz": ({}, [0.04, 0.25, 0.09],
                     [0.20273255405408219, 0.54930614433405489,
                      0.30951960420311175]),
    # Cohen's d family
    "transf_dtobesd": ({}, ZZ, [0.37873218748183352, 0.5, 0.68569533817705186]),
    "transf_dtocles": ({}, ZZ, [0.36183680491588155, 0.5, 0.7141961775233342]),
    "transf_dtocliffd": ({}, ZZ, [-0.2763263901682369, 0.0,
                                  0.42839235504666839]),
    "transf_dtolnor_logis": ({}, ZZ, [-0.90689968211710903, 0.0,
                                      1.4510394913873743]),
    "transf_dtolnor_norm": ({}, ZZ, [-0.82499999999999996, 0.0,
                                     1.3200000000000001]),
    "transf_dtoovl": ({}, ZZ, [0.8025873486341526, 1.0, 0.68915651677935164]),
    "transf_dtou1": ({}, ZZ, [0.32973202870483387, 0.0, 0.47426483359695731]),
    "transf_dtou2": ({}, ZZ, [0.4012936743170763, 0.5, 0.65542174161032418]),
    "transf_dtou3": ({}, ZZ, [0.30853753872598694, 0.5, 0.78814460141660336]),
    "transf_dtomd": ({"targs": {"sd": 2.0}}, ZZ, [-1.0, 0.0, 1.6000000000000001]),
    "transf_dtorpb": ({"n1i": N1, "n2i": N2}, ZZ,
                      [-0.2444782511820254, 0.0, 0.37342739356004673]),
    "transf_dtorbis": ({"n1i": N1, "n2i": N2}, ZZ,
                       [-0.30660666269913084, 0.0, 0.46818079126606815]),
    # correlation family
    "transf_rtorpb": ({"pi": 0.5}, PR, [0.1595769121605731,
                                        0.3989422804014327,
                                        0.63830764864229239]),
    "transf_rpbtod": ({"n1i": N1, "n2i": N2}, PR,
                      [0.40480073661449328, 1.1463431266870161,
                       2.6497878521450855]),
    "transf_rpbtorbis": ({"pi": 0.5}, PR, [0.25066282746310004,
                                           0.62665706865775006,
                                           1.0026513098524001]),
    "transf_rtod": ({"n1i": N1, "n2i": N2}, PR,
                    [0.32035347331093705, 0.8633587439765591,
                     1.6469719363874762]),
    "transf_clestod": ({}, PR, [-1.1902321628999897, 0.0, 1.1902321628999901]),
    # log-normal -> d / r
    "transf_lnortod_logis": ({}, ZZ, [-0.27566444771089604, 0.0,
                                      0.44106311633743367]),
    "transf_lnortod_norm": ({}, ZZ, [-0.30303030303030304, 0.0,
                                     0.48484848484848492]),
    "transf_lnortord": ({"pc": 0.5}, ZZ, [-0.12245933120185454, 0.0,
                                          0.1899744811276125]),
    "transf_lnortorr": ({"pc": 0.5}, PR, [1.0996679946249559,
                                          1.2449186624037092,
                                          1.379948962255225]),
    "transf_lnortortet_digby": ({}, ZZ, [-0.18533319990813948, 0.0,
                                         0.29131261245159096]),
    # Freeman-Tukey double arcsine
    "transf_pft": ({"ni": [120, 90, 150]}, PR,
                   [0.46672005653507842, 0.78539816339744828,
                    1.1046824966518334]),
    "transf_ipft": ({"ni": [120, 90, 150]}, [0.324, 0.288, 0.324],
                    [0.098069339450539028, 0.076103119404352826,
                     0.098720224274905344]),
    "transf_ipft_hm": ({"targs": {"ni": [120, 90, 150]}},
                       [0.324, 0.288, 0.324],
                       [0.097925101616267218, 0.077079382390004891,
                        0.097925101616267218]),
    "transf_irft": ({"ti": [120, 90]}, [0.324, 0.288],
                    [0.57284678606049377, 0.54178341465989055]),
    "transf_iirft": ({"ti": [120, 90]}, [0.324, 0.288],
                     [0.10085067876255314, 0.077481471660188983]),
    # the "int" (integrated over the random-effects distribution) variants
    "transf_ztor_int": ({"targs": TARGS}, ZZ,
                        [-0.43004754407324886, 0.0, 0.62943004593825058]),
    "transf_ilogit_int": ({"targs": TARGS}, ZZ,
                          [0.38029512313452873, 0.50000000000086176,
                           0.68606247518660002]),
    "transf_iprobit_int": ({"targs": TARGS}, ZZ,
                           [0.31677673768841669, 0.49999999999999994,
                            0.77719973355205696]),
    "transf_iarcsin_int": ({"targs": TARGS}, ZZ,
                           [0.031102370977181722, 0.090634010637158965,
                            0.51129443048982248]),
    "transf_iabt_int": ({"targs": TARGS}, ZZ,
                        [0.12033489209098028, 0.20962323632863514,
                         0.53130215114384183]),
    "transf_iahw_int": ({"targs": TARGS}, ZZ,
                        [0.31808772491991877, 0.50662501167648599,
                         0.90821057571079378]),
    "transf_dtocles_int": ({"targs": TARGS}, ZZ,
                           [0.36503486379136424, 0.50000000000121181,
                            0.70954378983343458]),
    "transf_exp_int": ({"targs": TARGS}, ZZ,
                       [0.63762815162177333, 1.0512710963760241,
                        2.3396468519259912]),
    # the "mode" variants (these optimise, so allow a little slack)
    "transf_exp_mode": ({"targs": TARGS}, ZZ,
                        [0.5488116360940265, 0.90483741803595952,
                         2.0137527074704766]),
    "transf_ilogit_mode": ({"targs": TARGS}, ZZ,
                           [0.37152158386883494, 0.5, 0.69839757577467165]),
    "transf_iprobit_mode": ({"targs": TARGS}, ZZ,
                            [0.2892513587155468, 0.50000000000000011,
                             0.8129837849427195]),
    "transf_ztor_mode": ({"targs": TARGS}, ZZ,
                         [-0.54315949003050934, 5.5511151231257827e-17,
                          0.73878227641344518]),
}
MODE_CASES = {"transf_iarcsin_mode", "transf_iabt_mode", "transf_iahw_mode",
              "transf_ilogit_mode", "transf_iprobit_mode", "transf_ztor_mode"}


@pytest.mark.parametrize("name", sorted(CASES))
def test_transform_matches_metafor(name):
    kwargs, arg, want = CASES[name]
    got = getattr(ma, name)(arg, **kwargs)
    # The *.mode variants maximise numerically; metafor's stats::optimize() uses
    # tol = 1e-4, so only that much agreement is meaningful.
    tol = 1e-4 if name in MODE_CASES else 1e-10
    assert list(np.atleast_1d(np.asarray(got, dtype=float))) == pytest.approx(
        want, abs=tol)


def test_transform_modes_stay_inside_their_bounds():
    """metafor's .mode() maximises on the transform's own support."""
    for name in ("transf_iarcsin_mode", "transf_iabt_mode", "transf_iahw_mode"):
        got = np.atleast_1d(np.asarray(getattr(ma, name)(ZZ, targs=TARGS),
                                       dtype=float))
        assert np.all(got >= 0.0) and np.all(got <= 1.0)


def test_scalar_input_returns_scalar():
    assert isinstance(ma.transf_logit(0.5), float)
    assert ma.transf_logit(0.5) == pytest.approx(0.0)


@pytest.mark.parametrize("pair", [
    ("transf_logit", "transf_ilogit"),
    ("transf_probit", "transf_iprobit"),
    ("transf_arcsin", "transf_iarcsin"),
    ("transf_abt", "transf_iabt"),
    ("transf_ahw", "transf_iahw"),
    ("transf_rtoz", "transf_ztor"),
])
def test_transform_round_trips(pair):
    fwd, inv = pair
    x = np.array(PR) if fwd in ("transf_logit", "transf_probit",
                                "transf_arcsin", "transf_abt",
                                "transf_ahw") else np.array([0.2, 0.5, -0.3])
    y = np.asarray(getattr(ma, fwd)(x), dtype=float)
    back = np.asarray(getattr(ma, inv)(y), dtype=float)
    assert back == pytest.approx(x, abs=1e-9)


def test_transform_validation():
    with pytest.raises(ValueError, match="tau2"):
        ma.transf_ilogit_mode(ZZ)
    with pytest.raises(ValueError, match="tau2"):
        ma.transf_ilogit_int(ZZ)
    with pytest.raises(ValueError, match="standard deviation"):
        ma.transf_dtomd(ZZ)
    with pytest.raises(ValueError, match="single standard deviation"):
        ma.transf_dtomd(ZZ, targs={"sd": [1.0, 2.0]})
    with pytest.raises(ValueError, match="'pi'"):
        ma.transf_rtorpb(PR, pi=1.5)
    with pytest.raises(ValueError, match="length of 'n1i'"):
        ma.transf_dtorpb(ZZ, n1i=[50, 60], n2i=[55, 65, 75])


def test_mode_rejects_a_negative_tau2():
    with pytest.raises(ValueError, match="tau2"):
        ma.transf_ilogit_mode(ZZ, targs={"tau2": -1.0})
