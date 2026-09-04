"""Equation core, independent evaluator, and successor services for CS011 blocks G0/N0/N1.

Nothing in this package produces an equilibrium result, a Ramsey optimum, a calibration,
or a welfare result. The result-use ceiling for every output is ``exploratory_only``.
"""

__version__ = "0.1.0"

#: Specification this package implements, and the blocks it covers.
SPEC_ID = "CS011"
SPEC_VERSION = "0.4"
BLOCKS_IMPLEMENTED = ("G0", "N0", "N1")

#: Hard ceiling on how any output of this package may be described. See
#: ``computation/error-handling-and-result-use.md`` in the research workspace.
RESULT_USE_CEILING = "exploratory_only"
