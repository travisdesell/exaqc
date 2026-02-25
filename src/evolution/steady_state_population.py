import bisect
import random

from functools import cmp_to_key
from typing import Callable

from loguru import logger
import os
import torch
import pennylane as qml
import matplotlib.pyplot as plt

from src.circuits.circuit import CircuitGenome
from src.evolution.population_strategy import PopulationStrategy
from src.utils.helpers import genome_to_torch_params


class SteadyStatePopulation(PopulationStrategy):

    def __init__(
        self,
        max_population_size: int,
        compare: Callable[[CircuitGenome, CircuitGenome], int],
        out_dir: str = "artifacts",
    ):
        """
        Creates a steady state population with the specified max population size.  The population
        will be sorted in order by genome fitness. The get parent methods can be called at any
        time to generate random parent selection.  Genomes will be inserted if the population size
        is below the max population size, or if they are better than the least fit genome in the
        population.  If adding a genome would cause the population size to be greater than the
        max population size, the least fit genome will be removed to keep it under the max size.

        Args:
            max_population_size: is the maximum number of genomes that the population will hold.
            compare: a compare function used for sorting genomes. this should return 0 if both
                genomes should be ranked the same, a negative value if the first genome should
                come before the second genome, and a positive number otherwise
        """

        self.max_population_size = max_population_size
        self.compare = compare
        self.out_dir = out_dir

        self.insertions = 0

        # used to store the population, should be kept in sorted order.
        self.population: list[CircuitGenome] = []

    def get_parent(self, **kwargs) -> CircuitGenome:
        """
        Used to get two or more parents to be used in mutation or
        other operations to generate children.

        Args:
            **kwargs: is used to pass additional options to the method to get
                a parent, e.g., specifying if it is for inter or intra-island
                crossover, or to come from a particular island or species.

        Returns:
            A single CircuitGenome from the population. If the population is empty
            it will return None.
        """

        if len(self.population) > 0:
            return random.choice(self.population)
        else:
            return None

    def get_parents(self, n_parents: int = 2, **kwargs) -> list[CircuitGenome]:
        """
        Used to get two or more parents to be used in crossover or
        other operations to generate children.

        Args:
            n_parents: specifies how many parents to return by the method.
            **kwargs: is used to pass additional options to the method to get
                a parent, e.g., specifying if it is for inter or intra-island
                crossover, or to come from a particular island or species.

        Returns:
            A list of unique (non-duplicate) CircuitGenomes. If the size of the population
            is less than n_parents, it will return None.
        """
        if len(self.population) >= n_parents:
            # sort the parents so the most fit is the first parent
            parents = random.sample(self.population, n_parents)
            parents.sort(key=cmp_to_key(self.compare))
            return parents
        else:
            return None

    def insert_genome(self, genome: CircuitGenome, **kwargs) -> bool:
        """
        Inserts a genome back into the population.

        Args:
            genome: is the genome to insert into the population.
            **kwargs: is used to pass additional options to the method for
                inserting the genome, such as an island or species it came from.

        Returns:
            True if it was inserted into the population, False otherwise.
        """

        # TODO: don't add duplicate genomes to the population
        # options:
        # 1. if gate innovation numbers are the same, keep the genome with better fitness
        # 2. if gate innovation numbers are the same but fitness different, keep both

        bisect.insort(
            self.population,
            genome,
            key=cmp_to_key(self.compare),
        )

        self.insertions += 1

        if genome.genome_number == self.population[0].genome_number:
            # this was a new global best genome
            logger.success(
                f"🎯 New best genome {genome.genome_number} "
                f"[insertion {self.insertions}] Population found new GLOBAL best genome with fitness: {genome.fitness}"
            )
            test_metric = genome.fitness.get("test_acc", None)
            if test_metric is None:
                test_metric = genome.fitness.get("test_fidelity")

            try:
                tag = (
                    f"trainloss_{genome.fitness['train_loss']:.4f}_testloss_"
                    f"{genome.fitness['test_loss']:.4f}_testacc_{test_metric:.3f}"
                )
            except Exception:
                tag = (
                    f"best_ep_return_{genome.fitness['best_episode_return']:.4f}_"
                    f"eval_return_mean_{genome.fitness['eval_return_mean']:.4f}"
                )

            self._save_best_circuit(genome, out_dir=self.out_dir, tag=tag)

        if len(self.population) > self.max_population_size:
            # remove the last genome from the population
            del self.population[-1]

    def _save_best_circuit(
        self, genome: CircuitGenome, out_dir: str = "artifacts/", tag: str = ""
    ):
        if genome is None:
            logger.error("Genome cannot be None")
            raise ValueError

        os.makedirs(out_dir, exist_ok=True)

        genome.generate_pennylane_circuit(return_probs=True, input_mode="angle")
        # logger.info(f"genome circuit: {genome.circuit}")

        # --- Text gate list ---
        txt_path = os.path.join(out_dir, f"genome_{genome.genome_number}.txt")
        with open(txt_path, "w") as f:
            genome.sort_gates()
            f.write(f"Genome {genome.genome_number}\n")
            f.write(f"Qubits: {genome.qubits}\n\n")
            for g in genome.gates:
                if getattr(g, "enabled", True):
                    f.write(
                        f"{g.depth:.3f}  {g.method_name}  {g.qubits}  {g.parameters}\n"
                    )

        # --- PennyLane draw ---
        try:
            params = genome_to_torch_params(genome)
            x0 = torch.zeros(len(genome.input_indexes))
            fig, ax = qml.draw_mpl(genome.circuit)(x0, params)
            ax.set_title(f"Genome {genome.genome_number}")
            path = os.path.join(
                out_dir, f"best_genome_{genome.genome_number}_{tag}.png"
            )
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not draw circuit: {e}")
