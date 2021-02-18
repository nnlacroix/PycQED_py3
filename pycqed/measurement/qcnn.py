# Import Qutip
from qutip import *
import numpy as np
import time

# Operators N = 7
X1 = tensor(sigmax(), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2))
Z1 = tensor(sigmaz(), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2))
Y1 = tensor(sigmay(), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2))
X2 = tensor(qeye(2), sigmax(), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2))
Z2 = tensor(qeye(2), sigmaz(), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2))
Y2 = tensor(qeye(2), sigmay(), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2))
X3 = tensor(qeye(2), qeye(2), sigmax(), qeye(2), qeye(2), qeye(2), qeye(2))
Z3 = tensor(qeye(2), qeye(2), sigmaz(), qeye(2), qeye(2), qeye(2), qeye(2))
Y3 = tensor(qeye(2), qeye(2), sigmay(), qeye(2), qeye(2), qeye(2), qeye(2))
X4 = tensor(qeye(2), qeye(2), qeye(2), sigmax(), qeye(2), qeye(2), qeye(2))
Z4 = tensor(qeye(2), qeye(2), qeye(2), sigmaz(), qeye(2), qeye(2), qeye(2))
Y4 = tensor(qeye(2), qeye(2), qeye(2), sigmay(), qeye(2), qeye(2), qeye(2))
X5 = tensor(qeye(2), qeye(2), qeye(2), qeye(2), sigmax(), qeye(2), qeye(2))
Z5 = tensor(qeye(2), qeye(2), qeye(2), qeye(2), sigmaz(), qeye(2), qeye(2))
Y5 = tensor(qeye(2), qeye(2), qeye(2), qeye(2), sigmay(), qeye(2), qeye(2))
X6 = tensor(qeye(2), qeye(2), qeye(2), qeye(2), qeye(2), sigmax(), qeye(2))
Z6 = tensor(qeye(2), qeye(2), qeye(2), qeye(2), qeye(2), sigmaz(), qeye(2))
Y6 = tensor(qeye(2), qeye(2), qeye(2), qeye(2), qeye(2), sigmay(), qeye(2))
X7 = tensor(qeye(2), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2), sigmax())
Z7 = tensor(qeye(2), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2), sigmaz())
Y7 = tensor(qeye(2), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2), sigmay())

# Operators N = 3
X1s = tensor(sigmax(), qeye(2), qeye(2))
Z1s = tensor(sigmaz(), qeye(2), qeye(2))
X2s = tensor(qeye(2), sigmax(), qeye(2))
Z2s = tensor(qeye(2), sigmaz(), qeye(2))
X3s = tensor(qeye(2), qeye(2), sigmax())
Z3s = tensor(qeye(2), qeye(2), sigmaz())

# Hamiltonian N = 7
def H_qpr (J, h1, h2):
    H = -J * (Z1*X2*Z3 + Z2*X3*Z4 + Z3*X4*Z5 + Z4*X5*Z6 + Z5*X6*Z7 + X1*Z2 + Z6*X7) \
        -h1 * (X1 + X2 + X3 + X4 + X5 + X6 + X7) \
        -h2 * (X1*X2 + X2*X3 + X3*X4 + X4*X5 + X5*X6 + X6*X7)
    return H

# Hamiltonian N = 3
def H_qpr_s (J, h1, h2):
    H = -J * (Z1s*X2s*Z3s + X1s*Z2s + X3s*Z2s) - h1 * (X1s + X2s + X3s) - h2 * (X1s*X2s + X2s*X3s)
    return H

# String order parameter
SOP = Z1 * X2 * X4 * X6 * Z7
SOP_QCNN = Z1s * X2s * Z3s

# Sigma Z
ZII = tensor(sigmaz(), identity(2), identity(2))
IZI = tensor(identity(2), sigmaz(), identity(2))
IIZ = tensor(identity(2), identity(2), sigmaz())

# Sigma Y
YII = tensor(sigmay(), identity(2), identity(2))
IYI = tensor(identity(2), sigmay(), identity(2))
IIY = tensor(identity(2), identity(2), sigmay())

# Some useful eigenstates
e0 = Qobj([[1],[0]])
e1 = Qobj([[0],[1]])
e11 = tensor(e1,e1)
e000 = tensor(e0,e0,e0)

e0000000 = tensor(e0,e0,e0,e0,e0,e0,e0)

# Matrices describing the ZZ coupling
r11I = tensor(e11*e11.dag(),identity(2))
rI11 = tensor(identity(2),e11*e11.dag())

r11_12 = tensor(e11*e11.dag(),identity(2),identity(2),identity(2),identity(2),identity(2))
r11_23 = tensor(identity(2),e11*e11.dag(),identity(2),identity(2),identity(2),identity(2))
r11_34 = tensor(identity(2),identity(2),e11*e11.dag(),identity(2),identity(2),identity(2))
r11_45 = tensor(identity(2),identity(2),identity(2),e11*e11.dag(),identity(2),identity(2))
r11_56 = tensor(identity(2),identity(2),identity(2),identity(2),e11*e11.dag(),identity(2))
r11_67 = tensor(identity(2),identity(2),identity(2),identity(2),identity(2),e11*e11.dag())


def Q7_mesolve(Ham, H0, T1, T2, psi0, rho0, times, tSTEP):

    # inputs
    # Ham -- sequence of Hamiltonians describing gates/buffer
    # H0 -- the Hamiltonian describing the residual ZZ coupling
    # Angles -- single qubit rotation angles (3x3 matrix)
    # T1 -- array of T1 times of individual qubits
    # T2 -- array of T2 times of individual qubits
    # psi0 -- initial state (must be pure state, serves for determining the ideal target states psit)
    # rho0 -- initial density matrix (here you can enter an imperfect initial state, otherwise set rho0 = psi0*psi0.dag())
    # times -- array contaning the initial time followed by times after each gate sequence
    # tSTEP -- time step

    # target states
    psit = []

    for index in np.arange(0, len(Ham)):
        psit.append(psi0)

    for index in np.arange(0, len(Ham)):
        psit[index] = (-(times[index + 1] - times[index]) * 1j * (Ham[index] - H0)).expm() * psit[index - 1]

    rhot = [(item * item.dag()) for item in psit]

    # jump operators
    Am1 = np.sqrt(1 / T1[0]) * tensor(sigmap(), identity(2), identity(2), identity(2), identity(2), identity(2), identity(2))
    Am2 = np.sqrt(1 / T1[1]) * tensor(identity(2), sigmap(), identity(2), identity(2), identity(2), identity(2), identity(2))
    Am3 = np.sqrt(1 / T1[2]) * tensor(identity(2), identity(2), sigmap(), identity(2), identity(2), identity(2), identity(2))
    Am4 = np.sqrt(1 / T1[3]) * tensor(identity(2), identity(2), identity(2), sigmap(), identity(2), identity(2), identity(2))
    Am5 = np.sqrt(1 / T1[4]) * tensor(identity(2), identity(2), identity(2), identity(2), sigmap(), identity(2), identity(2))
    Am6 = np.sqrt(1 / T1[5]) * tensor(identity(2), identity(2), identity(2), identity(2), identity(2), sigmap(), identity(2))
    Am7 = np.sqrt(1 / T1[6]) * tensor(identity(2), identity(2), identity(2), identity(2), identity(2), identity(2), sigmap())

    Az1 = np.sqrt(1 / 2 / T2[0] - 1 / 4 / T1[0]) * Z1
    Az2 = np.sqrt(1 / 2 / T2[1] - 1 / 4 / T1[1]) * Z2
    Az3 = np.sqrt(1 / 2 / T2[2] - 1 / 4 / T1[2]) * Z3
    Az4 = np.sqrt(1 / 2 / T2[3] - 1 / 4 / T1[3]) * Z4
    Az5 = np.sqrt(1 / 2 / T2[4] - 1 / 4 / T1[4]) * Z5
    Az6 = np.sqrt(1 / 2 / T2[5] - 1 / 4 / T1[5]) * Z6
    Az7 = np.sqrt(1 / 2 / T2[6] - 1 / 4 / T1[6]) * Z7


    jumps = [Am1, Am2, Am3, Am4, Am5, Am6, Am7, Az1, Az2, Az3, Az4, Az5, Az6, Az7]

    # mesolve options
    options = Options()
    options.store_states = True

    # initialization of output arrays
    tval = [times[0]]
    fidelity = [(rhot[0] * rho0).tr()]
    phnum = [(0.5 * (7 - Z1 - Z2 - Z3 - Z4 - Z5 - Z6 - Z7) * rho0).tr()]
    states = [rho0]

    # ME solver
    for index in np.arange(0, len(Ham)):
        resi = mesolve(Ham[index], states[-1], np.arange(times[index], times[index + 1] + tSTEP, tSTEP), jumps,
                       [rhot[index], 0.5 * (7 - Z1 - Z2 - Z3 - Z4 - Z5 - Z6 - Z7)], options=options)

        tval.extend(resi.times[1:])
        fidelity.extend(resi.expect[0][1:])
        phnum.extend(resi.expect[1][1:])
        states.extend(resi.states[1:])

    # returns object res contaning several outputs
    class res:
        def __init__(self):
            self.tval = tval
            self.fidelity = fidelity
            self.phnum = phnum
            self.states = states
            self.target = psit

    return res()

def Q3_mesolve(Ham, H0, T1, T2, psi0, rho0, times, tSTEP):
    # inputs
    # Ham -- sequence of Hamiltonians describing gates/buffer
    # H0 -- the Hamiltonian describing the residual ZZ coupling
    # Angles -- single qubit rotation angles (3x3 matrix)
    # T1 -- array of T1 times of individual qubits
    # T2 -- array of T2 times of individual qubits
    # psi0 -- initial state (must be pure state, serves for determining the ideal target states psit)
    # rho0 -- initial density matrix (here you can enter an imperfect initial state, otherwise set rho0 = psi0*psi0.dag())
    # times -- array contaning the initial time followed by times after each gate sequence
    # tSTEP -- time step

    # target states
    psit = []

    for index in np.arange(0, len(Ham)):
        psit.append(psi0)

    for index in np.arange(0, len(Ham)):
        psit[index] = (-(times[index + 1] - times[index]) * 1j * (Ham[index] - H0)).expm() * psit[index - 1]

    rhot = [(item * item.dag()) for item in psit]

    # jump operators
    Am1 = np.sqrt(1 / T1[0]) * tensor(sigmap(), identity(2), identity(2))
    Am2 = np.sqrt(1 / T1[1]) * tensor(identity(2), sigmap(), identity(2))
    Am3 = np.sqrt(1 / T1[2]) * tensor(identity(2), identity(2), sigmap())
    Az1 = np.sqrt(1 / 2 / T2[0] - 1 / 4 / T1[0]) * ZII
    Az2 = np.sqrt(1 / 2 / T2[1] - 1 / 4 / T1[1]) * IZI
    Az3 = np.sqrt(1 / 2 / T2[2] - 1 / 4 / T1[2]) * IIZ

    jumps = [Am1, Am2, Am3, Az1, Az2, Az3]

    # mesolve options
    options = Options()
    options.store_states = True

    # initialization of output arrays
    tval = [times[0]]
    fidelity = [(rhot[0] * rho0).tr()]
    phnum = [(0.5 * (3 - ZII - IZI - IIZ) * rho0).tr()]
    states = [rho0]

    # ME solver
    for index in np.arange(0, len(Ham)):
        resi = mesolve(Ham[index], states[-1], np.arange(times[index], times[index + 1] + tSTEP, tSTEP), jumps,
                       [rhot[index], 0.5 * (3 - ZII - IZI - IIZ)], options=options)

        tval.extend(resi.times[1:])
        fidelity.extend(resi.expect[0][1:])
        phnum.extend(resi.expect[1][1:])
        states.extend(resi.states[1:])

    # returns object res contaning several outputs
    class res:
        def __init__(self):
            self.tval = tval
            self.fidelity = fidelity
            self.phnum = phnum
            self.states = states
            self.target = psit

    return res()

def Q7_gnd_state_mesolve(T1s, T2s, Jzzs, var):
    # sigle-qubit gate time in ns
    ty = 45
    # two-qubit gate time in ns
    tg = 70
    # buffer time, before and after each two qubit gate
    tb = 25

    # time step in ns
    tSTEP = 1

    # rotation angles [j,k], in gate sequence j and for qubit k
    Angles = np.zeros([7,7])
    Angles[0, 0] = var[0]
    Angles[0, 1] = var[1]
    Angles[0, 2] = var[2]
    Angles[0, 3] = var[3]
    Angles[0, 4] = var[4]
    Angles[0, 5] = var[5]
    Angles[0, 6] = 0
    Angles[1, 0] = var[6]
    Angles[1, 1] = var[7]
    Angles[1, 2] = var[8]
    Angles[1, 3] = var[9]
    Angles[1, 4] = var[10]
    Angles[1, 5] = var[11]
    Angles[1, 6] = var[12]
    Angles[2, 0] = 0
    Angles[2, 1] = var[13]
    Angles[2, 2] = var[14]
    Angles[2, 3] = var[15]
    Angles[2, 4] = var[16]
    Angles[2, 5] = var[17]
    Angles[2, 6] = var[18]


    # rotation angles devided by gate time
    alpha = Angles / ty
    # Hamiltonians
    H0 = Jzzs[0] * r11_12 + Jzzs[1] * r11_23 + Jzzs[2] * r11_34 \
    + Jzzs[3] * r11_45 + Jzzs[4] * r11_56 + Jzzs[5] * r11_67  # residual ZZ coupling
    Hg12 = np.pi / tg * r11_12  # CZ on qubits 1 and 2
    Hg23 = np.pi / tg * r11_23  # CZ on qubits 2 and 3
    Hg34 = np.pi / tg * r11_34  # CZ on qubits 3 and 4
    Hg45 = np.pi / tg * r11_45  # CZ on qubits 4 and 5
    Hg56 = np.pi / tg * r11_56  # CZ on qubits 5 and 6
    Hg67 = np.pi / tg * r11_67  # CZ on qubits 6 and 7

    # single qubit rotations
    def Hy3(alph):
        return 0.5 * alph[0] * Y1 + 0.5 * alph[1] * Y2 + 0.5 * alph[2] * Y3 \
        + 0.5 * alph[3] * Y4 + 0.5 * alph[4] * Y5 + 0.5 * alph[5] * Y6 + 0.5 * alph[6] * Y7

    # Hamiltonian sequence, note the free hamiltonian before and after each two qubit gate corresponding to a buffer time
    Ham = [H0 + Hy3(alpha[0]), H0, H0 + Hg12 + Hg34 + Hg56, H0, H0 + Hy3(alpha[1]),
             H0, H0 + Hg23 + Hg45 + Hg67, H0, H0 + Hy3(alpha[2])]
    times = [0, ty, ty + tb, ty + tb + tg, ty + 2 * tb + tg, 2 * ty + 2 * tb + tg, 2 * ty + 3 * tb + tg,
             2 * ty + 3 * tb + 2 * tg, 2 * ty + 4 * tb + 2 * tg, 3 * ty + 4 * tb + 2 * tg]

    # ideal initial state
    psi0 = e0000000
    # initial density matrix, you can set here some imperfect initial state
    rho0 = psi0 * psi0.dag()
    start_time = time.time()

    # ME solver
    res = Q7_mesolve(Ham, H0, T1s, T2s, psi0, rho0, times, tSTEP)

    return res

def Q3_gnd_state_mesolve(T1s, T2s, Jzzs, var):
    # sigle-qubit gate time in ns
    ty = 45
    # two-qubit gate time in ns
    tg = 70
    # buffer time, before and after each two qubit gate
    tb = 25

    # time step in ns
    tSTEP = 1

    # rotation angles [j,k], in gate sequence j and for qubit k
    Angles = np.zeros([3, 3])
    Angles[0, 0] = var[0]
    Angles[0, 1] = var[1]
    Angles[0, 2] = 0
    Angles[1, 0] = var[2]
    Angles[1, 1] = var[3]
    Angles[1, 2] = var[4]
    Angles[2, 0] = 0
    Angles[2, 1] = var[5]
    Angles[2, 2] = var[6]

    # rotation angles devided by gate time
    alpha = Angles / ty
    # Hamiltonians
    H0 = Jzzs[0] * r11I + Jzzs[1] * rI11  # residual ZZ coupling
    Hg12 = np.pi / tg * r11I  # CZ on qubits 1 and 2
    Hg23 = np.pi / tg * rI11  # CZ on qubits 2 and 3

    # single qubit rotations
    def Hy3(alph):
        return 0.5 * alph[0] * YII + 0.5 * alph[1] * IYI + 0.5 * alph[2] * IIY

    # Hamiltonian sequence, note the free hamiltonian before and after each two qubit gate corresponding to a buffer time
    Ham = [H0 + Hy3(alpha[0]), H0, H0 + Hg12, H0, H0 + Hy3(alpha[1]), H0, H0 + Hg23, H0, H0 + Hy3(alpha[2])]
    times = [0, ty, ty + tb, ty + tb + tg, ty + 2 * tb + tg, 2 * ty + 2 * tb + tg, 2 * ty + 3 * tb + tg,
             2 * ty + 3 * tb + 2 * tg, 2 * ty + 4 * tb + 2 * tg, 3 * ty + 4 * tb + 2 * tg]

    # ideal initial state
    psi0 = e000
    # initial density matrix, you can set here some imperfect initial state
    rho0 = psi0 * psi0.dag()
    start_time = time.time()

    # ME solver
    res = Q3_mesolve(Ham, H0, T1s, T2s, psi0, rho0, times, tSTEP)

    return res

def Q7_cost_function(var):
    global oper, T1s, T2s, Jzzs, E0

    res = Q7_gnd_state_mesolve(T1s, T2s, Jzzs, var=var)
    gnd_state = res.states[-1]
    return expect(oper=oper, state=gnd_state) - E0

def Q3_cost_function(var):
    global oper, T1s, T2s, Jzzs, E0

    res = Q3_gnd_state_mesolve(T1s, T2s, Jzzs, var=var)
    gnd_state = res.states[-1]
    return expect(oper=oper, state=gnd_state) - E0

def set_sim_config(oper_f, T1s_f, T2s_f, Jzzs_f, E0_f):
    global oper, T1s, T2s, Jzzs, E0

    oper = oper_f
    T1s = T1s_f
    T2s = T2s_f
    Jzzs = Jzzs_f
    E0 = E0_f
