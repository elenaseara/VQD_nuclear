"""Shared algebraic convention, verified against the Qibo ansatz circuit."""

# Each ansatz layer evolves as exp(ANSATZ_DERIVATIVE_PHASE * theta * A),
# where A is the Hermitian Pauli generator used by the nuclear pool.
ANSATZ_DERIVATIVE_PHASE = -1j
