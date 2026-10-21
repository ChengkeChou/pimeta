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
:mod:`.splitting`           ``netmeta::netsplit`` — direct vs indirect evidence
:mod:`.league`              ``netmeta::netleague`` — league table
:mod:`.pairwise_table`      ``netmeta::netpairwise`` — pairwise view
:mod:`.complexity`          ``netcomplexity`` and ``netmeta::netcomplex``
:mod:`.convert`             ``crossnma2netmeta`` / ``gemtc2netmeta``
:mod:`.matrix`              ``netmatrix`` / ``netcomparison`` / ``netbind``
:mod:`.paths`               ``netmeta::netpath`` — path-by-path inconsistency
:mod:`.labels`              ``netmeta::treats`` — label abbreviation
:mod:`.ranking`             ``netmeta::vikor`` — VIKOR ranking
:mod:`.tables`              ``netmeta::subgroup`` / ``netmeta::nettable``
=========================  ===============================================

Every module above only *reads* a fitted object (plus the small helpers in
:mod:`.core`, :mod:`.impact` and :mod:`.component`); none of them touches the
fitting code, which is why they could be moved without a single test change.

Sections still in ``pimeta/netmeta.py`` (in file order), each a contiguous,
banner-delimited block:

==========================  ==============================================
krahn / decomposition       ``_nma_ruecker`` … ``decomp_design``
core                        ``netmeta``, ``netrank``, ``_nma_tau_ml_reml``
heat matrix                 ``netheat_matrix``
meta-regression             ``netmetareg``
public helper aliases       ``createC``, ``invmat``, ``hatmatrix``, ``comps``
==========================  ==============================================
"""

from __future__ import annotations

from .binary import NetmetabinResult, netmetabin
from .complexity import NetcomplexResult, netcomplex, netcomplexity
from .component import NetcombResult, discomb, netcomb
from .contrib import NetcontribResult, netcontrib
from .convert import crossnma2netmeta, gemtc2netmeta
from .graph import NetconnectionResult, NetdistanceResult, netconnection, netdistance
from .impact import NetimpactResult, netimpact
from .labels import treats
from .league import NetleagueResult, netleague
from .matrix import (
    NetbindResult,
    NetcomparisonResult,
    NetmatrixResult,
    netbind,
    netcomparison,
    netmatrix,
)
from .measures import NetmeasuresResult, netmeasures
from .pairwise import PairwiseData, pairwise
from .pairwise_table import NetpairwiseResult, netpairwise
from .paths import netpath
from .ranking import VikorResult, vikor
from .splitting import NetsplitResult, netsplit
from .tables import NetsubgroupResult, NettableResult, nettable, subgroup

__all__ = [
    "NetbindResult",
    "NetcombResult",
    "NetcomplexResult",
    "NetcomparisonResult",
    "NetconnectionResult",
    "NetcontribResult",
    "NetdistanceResult",
    "NetimpactResult",
    "NetleagueResult",
    "NetmatrixResult",
    "NetmeasuresResult",
    "NetmetabinResult",
    "NetpairwiseResult",
    "NetsplitResult",
    "NetsubgroupResult",
    "NettableResult",
    "PairwiseData",
    "VikorResult",
    "crossnma2netmeta",
    "discomb",
    "gemtc2netmeta",
    "netbind",
    "netcomb",
    "netcomparison",
    "netcomplex",
    "netcomplexity",
    "netconnection",
    "netcontrib",
    "netdistance",
    "netimpact",
    "netleague",
    "netmatrix",
    "netmeasures",
    "netmetabin",
    "netpairwise",
    "netpath",
    "netsplit",
    "nettable",
    "pairwise",
    "subgroup",
    "treats",
    "vikor",
]
