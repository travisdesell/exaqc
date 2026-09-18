from pennylane import numpy as np
import pennylane as qp
import matplotlib.pyplot as plt

#### TUNABLE PARAMETERS
MAX_LAYERS = 10
STEPS = 300
STEPSIZE = 0.1
######################
# CURRENTLY INITIALIZED TO 0000 BUT DOCUMENTATION RECOMMENDS 1100 (HARTREE-FOCK STATE) SO MAYBE CHANGE LATER
###############


def plot_energy_history(energy_history):
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^", "x", "*", "v", "<", ">"]
    for layers, energies in enumerate(energy_history, start=1):
        plt.plot(
            range(1, len(energies) + 1),
            energies,
            label=f"Layers: {layers}",
            marker=markers[layers % len(markers)],
            markevery=10,
            linestyle=line_styles[layers % len(line_styles)],
            linewidth=2,
        )

    plt.xlabel("Steps")
    plt.ylabel("Energy")
    plt.title("Energy vs Steps for Different Layers")
    plt.legend()
    plt.show()


def print_progress_bar(iterations, total_iterations):
    progress = iterations / total_iterations
    bar_length = 40
    block = int(round(bar_length * progress))
    text = f"\rProgress {iterations}/{total_iterations}: [{'#' * block + '-' * (bar_length - block)}] {round(progress * 100, 2)}%"
    print(text, end="")


# get hamiltonian for hydrogen molecule (grabbing everything because i dont know exactly what to grab)
H2data = qp.data.load("qchem", molname="H2", bondlength=0.742, basis="STO-3G")[0]
# print(qp.data.list_attributes(data_name="qchem"))


# h2 hamiltonian ( @ is the tensor product, the number in the parentheses is the qubit index)
# -0.09963387941370971 * I(0)
# + 0.17110545123720233 * Z(0)
# + 0.17110545123720225 * Z(1)
# + 0.16859349595532533 * (Z(0) @ Z(1))
# + 0.04533062254573469 * (Y(0) @ X(1) @ X(2) @ Y(3))
# + -0.04533062254573469 * (Y(0) @ Y(1) @ X(2) @ X(3))
# + -0.04533062254573469 * (X(0) @ X(1) @ Y(2) @ Y(3))
# + 0.04533062254573469 * (X(0) @ Y(1) @ Y(2) @ X(3))
# + -0.22250914236600539 * Z(2)
# + 0.12051027989546245 * (Z(0) @ Z(2))
# + -0.22250914236600539 * Z(3)
# + 0.16584090244119712 * (Z(0) @ Z(3))
# + 0.16584090244119712 * (Z(1) @ Z(2))
# + 0.12051027989546245 * (Z(1) @ Z(3))
# + 0.1743207725924201 * (Z(2) @ Z(3))

# H2 hamiltonian
# target is min energy
# initial state is either |0000> or |1100> with the second being hartree fock state whatever that is (see wolfram alpha citation)
# qc is going to be 4 qubits, layers of roation gates and cnot entangles
# objective function is <psi|H|psi> psi is state of system after qc is applied to initial state
# differentiation method is backpropagation
# classical optimization method is gradient descent
# stopping condition is... unknown right now


energy_history = []

H2 = H2data.hamiltonian

for layers in range(1, MAX_LAYERS + 1):
    print_progress_bar(layers, MAX_LAYERS)
    print()
    layer_energy = []
    # create a device with 4 qubits
    dev = qp.device("default.qubit", wires=4)

    # define the quantum circuit
    @qp.qnode(dev, diff_method="backprop")
    def circuit(params):
        # initialize the state to |1100> HF state
        qp.BasisState(
            np.array([1, 1, 0, 0]),
            wires=range(4),
        )

        for layer in range(layers):
            # put xyz rotation on each qubit
            for qubit in range(4):
                qp.RX(params[layer, qubit, 0], wires=qubit)
                qp.RY(params[layer, qubit, 1], wires=qubit)
                qp.RZ(params[layer, qubit, 2], wires=qubit)
            # put cnot on 0-1, 1-2, 2-3
            for qubit in range(3):
                qp.CNOT(wires=[qubit, qubit + 1])
        return qp.expval(H2)

    # initialize parameters randomly
    params = np.random.rand(layers, 4, 3, requires_grad=True)

    # define the objective function
    def objective_function(params):
        return circuit(params)

    # classical optimizer, step size currently arbitrary
    opt = qp.GradientDescentOptimizer(stepsize=STEPSIZE)

    # stop after steps, currently arbitrary
    for step in range(STEPS):
        print_progress_bar(step + 1, STEPS)

        params = opt.step(objective_function, params)

        # calculate the energy of the current state
        energy = circuit(params)
        layer_energy.append(energy)

    # save the energy history for this layer count
    energy_history.append(layer_energy)

###### tabular output of energy history for each layer count for me ####
print("\nEnergy history for each layer count:")
print(f"{'Layers':<10}{'Final Energy':<20}{'Minimum Energy':<20}")
for layers, energies in enumerate(energy_history, start=1):
    final_energy = energies[-1]
    minimum_energy = min(energies)

    print(
        f"{layers:<10}"
        f"{float(final_energy):<20.10f}"
        f"{float(minimum_energy):<20.10f}"
    )

###### LATEX TABLE OF ENERGY HISTORY FOR EACH LAYER COUNT for me ####
print(r"latex table of energy history for each layer count:")

print(r"\begin{tabular}{r r r r r}")
print(r"\hline")
print(r"Layers & Parameters & Final Energy & Minimum Energy & Step of Minimum \\")
print(r"\hline")

for layers, energies in enumerate(energy_history, start=1):
    final_energy = float(energies[-1])
    minimum_energy = float(min(energies))
    min_step = energies.index(min(energies)) + 1
    parameters = layers * 12

    print(
        f"{layers} & "
        f"{parameters} & "
        f"{final_energy:.10f} & "
        f"{minimum_energy:.10f} & "
        f"{min_step} \\\\"
    )

print(r"\hline")
print(r"\end{tabular}")

####### plot of energy history for each layer count ####
plot_energy_history(energy_history)
