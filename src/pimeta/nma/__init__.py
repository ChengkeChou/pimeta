"""Split-out sections of the network meta-analysis implementation.

``pimeta.netmeta`` remains the public entry point and re-exports everything
here, so ``pimeta.netmeta.pairwise`` keeps working exactly as before.  The
package exists so the 7,000-line implementation can be carved into reviewable
pieces without changing a single import path.

Sections already moved:

=========================  ===============================================
:mod:`.pairwise`            ``netmeta::pairwise`` — arm-level to comparison-level
:mod:`.measures`            ``netmeta::netmeasures`` — direct evidence share
:mod:`.impact`              ``netmeta::netimpact`` — study influence
=========================  ===============================================

Sections still in ``pimeta/netmeta.py`` (in file order), each a contiguous,
banner-delimited block and therefore movable on its own:

==========================  ==============================================
helpers                     ``_create_b`` … ``_iv_pool``
krahn / decomposition       ``_nma_ruecker`` … ``decomp_design``
core                        ``netmeta``, ``netrank``, ``_nma_tau_ml_reml``
contributions               ``netcontrib`` and its path/hat-matrix helpers
splitting                   ``netsplit``
league                      ``netleague``
distance / connectivity     ``netdistance``, ``netconnection``
pairwise ranking            ``netpairwise``
combination                 ``netcomb``, ``discomb``
binary networks             ``netmetabin``
simplification              ``netcomplexity``, ``netcomplex``
meta-regression             ``netmetareg``
network paths               ``netpath`` and its path helpers
network matrices            ``netmatrix``, ``netcomparison``, ``netbind``
model conversion            ``crossnma2netmeta``, ``gemtc2netmeta``
ranking and graphics        ``netrank`` plots, ``netheat``, ``netposet``, …
==========================  ==============================================
"""

from __future__ import annotations

from .impact import NetimpactResult, netimpact
from .measures import NetmeasuresResult, netmeasures
from .pairwise import PairwiseData, pairwise

__all__ = [
    "NetimpactResult",
    "NetmeasuresResult",
    "PairwiseData",
    "netimpact",
    "netmeasures",
    "pairwise",
]
