"""Split-out sections of the network meta-analysis implementation.

``pimeta.netmeta`` remains the public entry point and re-exports everything
here, so ``pimeta.netmeta.pairwise`` keeps working exactly as before.  The
package exists so the 7,000-line implementation can be carved into reviewable
pieces without changing a single import path.

Sections already moved:

=========================  ===============================================
:mod:`.core`                incidence matrices, ``invmat``, I^2, Q, DL tau^2
:mod:`.pairwise`            ``netmeta::pairwise`` — arm-level to comparison-level
:mod:`.measures`            ``netmeta::netmeasures`` — direct evidence share
:mod:`.impact`              ``netmeta::netimpact`` — study influence
:mod:`.binary`              ``netmeta::netmetabin`` — binary (2x2) networks
:mod:`.component`           ``netmeta::netcomb`` / ``discomb`` — additive CNMA
:mod:`.contrib`             ``netmeta::netcontrib`` — study contributions
:mod:`.graph`               ``netmeta::netconnection`` / ``netdistance``
=========================  ===============================================

Sections still in ``pimeta/netmeta.py`` (in file order), each a contiguous,
banner-delimited block and therefore movable on its own:

==========================  ==============================================
krahn / decomposition       ``_nma_ruecker`` … ``decomp_design``
core                        ``netmeta``, ``netrank``, ``_nma_tau_ml_reml``
splitting                   ``netsplit``
league                      ``netleague``
pairwise ranking            ``netpairwise``
simplification              ``netcomplexity``, ``netcomplex``
meta-regression             ``netmetareg``
network paths               ``netpath`` and its path helpers
network matrices            ``netmatrix``, ``netcomparison``, ``netbind``
model conversion            ``crossnma2netmeta``, ``gemtc2netmeta``
ranking and graphics        ``netrank`` plots, ``netheat``, ``netposet``, …
==========================  ==============================================
"""

from __future__ import annotations

from .binary import NetmetabinResult, netmetabin
from .component import NetcombResult, discomb, netcomb
from .contrib import NetcontribResult, netcontrib
from .graph import NetconnectionResult, NetdistanceResult, netconnection, netdistance
from .impact import NetimpactResult, netimpact
from .measures import NetmeasuresResult, netmeasures
from .pairwise import PairwiseData, pairwise

__all__ = [
    "NetcombResult",
    "NetconnectionResult",
    "NetcontribResult",
    "NetdistanceResult",
    "NetimpactResult",
    "NetmeasuresResult",
    "NetmetabinResult",
    "PairwiseData",
    "discomb",
    "netcomb",
    "netconnection",
    "netcontrib",
    "netdistance",
    "netimpact",
    "netmeasures",
    "netmetabin",
    "pairwise",
]
