"""Standardised, traceable, reproducible input and output.

A published meta-analysis has to answer more than "what was the estimate?".
A reader may ask *which file* the numbers came from, *which settings* were
used, *which software versions* were involved, and whether re-running the
analysis gives the same answer.  This subpackage makes those answers part of
the output instead of part of somebody's memory.

Modules
-------
==========================  ==========================================================
``record.jsonio``           strict JSON that keeps NaN and infinity intact
``record.fingerprint``      SHA-256 digests: "same file?" and "same numbers?"
``record.table``            symmetric CSV/TSV/JSON/JSONL/NPZ read and write
``record.schema``           the five-block analysis record, ``verify`` and ``replay``
``record.store``            named collection of fits (``estimates store``)
==========================  ==========================================================

The five blocks
---------------
Every record carries ``inputs`` (输入数据), ``analysis`` (分析配置),
``process`` (运算过程), ``environment`` (运行环境) and ``results``
(结果与诊断).  Together they let a third party rebuild the analysis -- and
audit it, too.

Quick start
-----------
::

    import pimeta as ma
    from pimeta.record import save_record, load_record

    fit = ma.rma(df["TE"], vi=df["seTE2"], method="REML", slab=df["study"])
    save_record(
        "m1.json", fit,
        data=df,
        mode="bundle",
        function="rma",
        call={"yi": "TE", "vi": "seTE2", "method": "REML"},
    )

    record = load_record("m1.json")
    record.verify()      # does the bundled data still hash to the same thing?
    record.replay()      # re-fit and compare every number

See Also:
    :doc:`/guide/reproducibility` for the workflow, and
    :doc:`/guide/data-frames` for how ``data=`` resolves column names.
"""

from __future__ import annotations

from .fingerprint import (
    FINGERPRINT_VERSION,
    analysis_fingerprint,
    fingerprint,
    matches,
    source_fingerprint,
)
from .schema import (
    RECORD_SCHEMA,
    AnalysisRecord,
    RecordError,
    RecordMismatch,
    build_record,
    compare_results,
    environment,
    load_record,
    save_record,
)
from .store import STORE_SCHEMA, EstimatesStore, StoredModel, estimates_store
from .table import read_table, write_table

__all__ = [
    # fingerprints
    "FINGERPRINT_VERSION",
    "analysis_fingerprint",
    "fingerprint",
    "matches",
    "source_fingerprint",
    # the record
    "RECORD_SCHEMA",
    "AnalysisRecord",
    "RecordError",
    "RecordMismatch",
    "build_record",
    "compare_results",
    "environment",
    "load_record",
    "save_record",
    # named store
    "STORE_SCHEMA",
    "EstimatesStore",
    "StoredModel",
    "estimates_store",
    # tables
    "read_table",
    "write_table",
]
