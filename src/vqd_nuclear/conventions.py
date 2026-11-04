"""Convenciones del ansatz. Fuente única.

ANSATZ_DERIVATIVE_PHASE fija la convencion de la que dependen, a la vez:
    |d psi / d theta> = PHASE * T |psi>
    U(theta)          = exp(PHASE * theta * T)
    dE/dtheta         = 2 Re[PHASE * <H psi|T psi>]
    G_mu              = PHASE * [H, A_mu]
"""
ANSATZ_DERIVATIVE_PHASE = -1j