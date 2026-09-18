import torch
import pennylane as qml

from mpi4py import MPI
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.circuits.encoder import initialize_encoder
from src.circuits.decoder import initialize_decoder
from src.circuits.pennylane_gate_specifications import (
    pennylane_gate_specifications,
)

from src.evolution.objective import Objective
from src.evolution.steady_state_population import SteadyStatePopulation
from src.evolution.master_worker import master_worker

# ============================================================
# TUNABLE PARAMETERS
# ============================================================

MAX_POPULATION_SIZE = 60
NUMBER_GENOMES = 500

N_QUBITS = 4

STEPS = 300
STEPSIZE = 0.1

MUTATION_STRATEGY = ["uniform", "1", "3"]
PARENT_STRATEGY = ["uniform", "2", "3"]

BINARY_CROSSOVER_RATE = 0.00
N_ARY_CROSSOVER_RATE = 0.20
EXPONENTIAL_CROSSOVER_RATE = 0.10


# ============================================================
# FITNESS COMPARISON
# ============================================================


def compare(
    genome1: CircuitGenome,
    genome2: CircuitGenome,
) -> int:
    """
    Compare two evaluated circuit genomes.

    Lower H2 energy is better.
    """

    energy1 = genome1.fitness["energy"]
    energy2 = genome2.fitness["energy"]

    if energy1 < energy2:
        return -1

    if energy1 > energy2:
        return 1

    return 0


# ============================================================
# H2 PROBLEM
# ============================================================


def get_h2_hamiltonian():
    """
    Load the H2 Hamiltonian once on the MPI master process
    and broadcast it to all worker processes.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    hamiltonian = None

    if rank == 0:
        dataset = qml.data.load(
            "qchem",
            molname="H2",
            bondlength=0.742,
            basis="STO-3G",
        )[0]

        hamiltonian = dataset.hamiltonian

    # Send the Hamiltonian from rank 0 to every process.
    hamiltonian = comm.bcast(
        hamiltonian,
        root=0,
    )

    return hamiltonian


# ============================================================
# H2 OBJECTIVE
# ============================================================


class H2Objective(Objective):
    """
    Train and evaluate one evolved circuit architecture against
    the H2 Hamiltonian.

    Evolution determines the circuit architecture.

    Gradient descent determines the best values of the circuit's
    continuous gate parameters.

    Fitness is the lowest energy found:

        <psi(theta) | H | psi(theta)>
    """

    def __init__(
        self,
        hamiltonian,
        n_qubits: int,
    ):
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits

        self.device = qml.device(
            "default.qubit",
            wires=n_qubits,
        )

        # H2 contains two electrons.
        #
        # In the four-qubit STO-3G representation the
        # Hartree-Fock reference state is:
        #
        #     |1100>
        #
        self.hf_state = torch.tensor(
            [1, 1, 0, 0],
            dtype=torch.int64,
        )

    def __call__(
        self,
        genome: CircuitGenome,
    ) -> None:
        """
        Train one evolved genome and assign its fitness.
        """

        best_energy = self.train_and_evaluate(genome)

        genome.fitness = {
            "energy": best_energy,
        }

    def build_energy_qnode(
        self,
        genome: CircuitGenome,
    ):
        """
        Build the PennyLane circuit used to calculate

            <psi|H|psi>

        for one evolved genome.
        """

        @qml.qnode(
            self.device,
            interface="torch",
            diff_method="backprop",
        )
        def energy_qnode(weights):

            # --------------------------------------------
            # Prepare H2 reference state
            # --------------------------------------------

            qml.BasisState(
                self.hf_state,
                wires=range(self.n_qubits),
            )

            # --------------------------------------------
            # Apply evolved circuit
            # --------------------------------------------

            genome.sort_gates()

            offset = 0

            for gate in genome.gates:

                if gate.enabled:
                    gate.add_to_pennylane_circuit(
                        genome.qubits,
                        weights=weights,
                        offset=offset,
                    )

                # Important:
                #
                # CircuitGenome reserves parameter positions for
                # all gates, including disabled gates.
                offset += len(gate.parameters)

            # --------------------------------------------
            # H2 energy
            # --------------------------------------------

            return qml.expval(self.hamiltonian)

        return energy_qnode

    def train_and_evaluate(
        self,
        genome: CircuitGenome,
    ) -> float:
        """
        Optimize the continuous parameters of one evolved circuit.

        Uses the same classical optimization budget as Experiment 1:
            - gradient descent
            - stepsize = 0.1
            - 300 steps

        Returns:
            Lowest H2 energy found.
        """

        steps = int(genome.hyperparameters["steps"])

        stepsize = float(genome.hyperparameters["stepsize"])

        # Build the quantum circuit for this evolved architecture.
        energy_qnode = self.build_energy_qnode(genome)

        # --------------------------------------------------------
        # Pull the gate parameters out of the genome
        # --------------------------------------------------------

        parameter_values = []

        for value in genome.get_parameters_as_list():

            if torch.is_tensor(value):
                value = value.detach().item()

            parameter_values.append(float(value))

        genome.metadata["energy_history"] = []

        # --------------------------------------------------------
        # Circuit has no trainable parameters
        # --------------------------------------------------------

        if len(parameter_values) == 0:

            weights = torch.tensor(
                [],
                dtype=torch.float64,
                requires_grad=False,
            )

            energy = energy_qnode(weights)

            energy_value = float(energy.detach().item())

            genome.metadata["energy_history"].append(energy_value)

            genome.metadata["best_energy"] = energy_value
            genome.metadata["best_step"] = 0

            return energy_value

        # --------------------------------------------------------
        # Initialize trainable parameters
        # --------------------------------------------------------

        weights = torch.tensor(
            parameter_values,
            dtype=torch.float64,
            requires_grad=True,
        )

        optimizer = torch.optim.SGD(
            [weights],
            lr=stepsize,
        )

        best_energy = float("inf")
        best_weights = weights.detach().clone()
        best_step = 0

        # --------------------------------------------------------
        # Parameter optimization
        # --------------------------------------------------------

        for step in range(steps):

            optimizer.zero_grad()

            # Calculate current energy.
            energy = energy_qnode(weights)

            # Some genomes may contain parameters that do not
            # actually affect the active circuit.
            if not energy.requires_grad:

                energy_value = float(energy.detach().item())

                genome.metadata["energy_history"].append(energy_value)

                if energy_value < best_energy:
                    best_energy = energy_value
                    best_step = step
                    best_weights = weights.detach().clone()

                break

            # Calculate dE/dtheta.
            energy.backward()

            # Update theta.
            optimizer.step()

            # ----------------------------------------------------
            # Measure energy AFTER the gradient descent update.
            #
            # This matches Experiment 1.
            # ----------------------------------------------------

            updated_energy = energy_qnode(weights)

            energy_value = float(updated_energy.detach().item())

            genome.metadata["energy_history"].append(energy_value)

            # ----------------------------------------------------
            # Save best result
            # ----------------------------------------------------

            if energy_value < best_energy:

                best_energy = energy_value
                best_step = step + 1

                best_weights = weights.detach().clone()

        # --------------------------------------------------------
        # Restore best parameters found
        # --------------------------------------------------------

        genome.set_parameters_from_list(best_weights.tolist())

        genome.metadata["best_energy"] = best_energy
        genome.metadata["best_step"] = best_step

        return best_energy


# ============================================================
# TEMPORARY PROFILER
# ============================================================


class NullProfiler:
    """
    Temporary no-op profiler.

    SteadyStatePopulation expects a profiler, but the existing
    artifact-saving code is heavily classification-oriented.

    For the first H2 experiment we disable that machinery.
    """

    def record(self, *args, **kwargs):
        pass

    def plot_single_run(self, *args, **kwargs):
        pass


# ============================================================
# MAIN
# ============================================================


def main():

    # --------------------------------------------------------
    # H2 Hamiltonian
    # --------------------------------------------------------

    hamiltonian = get_h2_hamiltonian()

    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.info(
            "H2 Hamiltonian:\n{}",
            hamiltonian,
        )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    objective = H2Objective(
        hamiltonian=hamiltonian,
        n_qubits=N_QUBITS,
    )

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    population = SteadyStatePopulation(
        max_population_size=MAX_POPULATION_SIZE,
        compare=compare,
        # Disable the existing classification-oriented
        # circuit artifact writer for now.
        out_dir=None,
        profiler=NullProfiler(),
        save_training_plot=False,
    )

    # --------------------------------------------------------
    # Hyperparameters used when evaluating each genome
    # --------------------------------------------------------

    hyperparameters = {
        "steps": STEPS,
        "stepsize": STEPSIZE,
        "quantum_input_mode": "rx",
        "quantum_output_mode": "expval",
    }

    # --------------------------------------------------------
    # Placeholder encoder / decoder
    # --------------------------------------------------------
    #
    # EXAQC currently expects every CircuitGenome to carry an
    # encoder and decoder.
    #
    # They are not actually used by H2Objective.
    # --------------------------------------------------------

    initial_encoder = initialize_encoder(
        target="pennylane",
        encoding_str="identity",
        n_inputs=N_QUBITS,
        n_outputs=N_QUBITS,
        config=None,
        quantum_input_mode="rx",
        n_input_qubits=N_QUBITS,
    )

    initial_decoder = initialize_decoder(
        target="pennylane",
        decoding_str="linear",
        n_inputs=N_QUBITS,
        n_outputs=1,
    )

    # --------------------------------------------------------
    # Evolutionary search
    # --------------------------------------------------------

    master_worker(
        gate_specifications=pennylane_gate_specifications,
        population=population,
        objective=objective,
        initial_encoder=initial_encoder,
        initial_decoder=initial_decoder,
        hyperparameters=hyperparameters,
        mutation_strategy=MUTATION_STRATEGY,
        parent_strategy=PARENT_STRATEGY,
        binary_crossover_rate=(BINARY_CROSSOVER_RATE),
        n_ary_crossover_rate=(N_ARY_CROSSOVER_RATE),
        exponential_crossover_rate=(EXPONENTIAL_CROSSOVER_RATE),
        run_for=NUMBER_GENOMES,
        # All four qubits are part of the H2 state.
        input_registers={
            "q": N_QUBITS,
        },
        output_registers={
            "q": N_QUBITS,
        },
        target="pennylane",
    )

    # --------------------------------------------------------
    # Print final best genome
    # --------------------------------------------------------

    if MPI.COMM_WORLD.Get_rank() == 0:

        best_genome = population.get_best_genome()

        enabled_gates = [gate for gate in best_genome.gates if gate.enabled]

        print()
        print("====================================")
        print("BEST EVOLVED H2 CIRCUIT")
        print("====================================")

        print(f"Genome: {best_genome.genome_number}")

        print(f"Energy: " f"{best_genome.fitness['energy']:.10f}")

        print(f"Best optimization step: " f"{best_genome.metadata['best_step']}")

        print(f"Enabled gates: {len(enabled_gates)}")

        print(f"Trainable parameters: " f"{len(best_genome.get_parameters_as_list())}")

        print()

        for gate in enabled_gates:

            print(
                f"{gate.method_name:<10} "
                f"qubits={gate.qubits} "
                f"params={gate.parameters} "
                f"depth={gate.depth:.4f}"
            )


if __name__ == "__main__":
    main()
