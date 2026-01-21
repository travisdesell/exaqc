import math
import random

from src.circuits.circuit import CircuitGenome

def binary_crossover(
    child: CircuitGenome,
    p1: CircuitGenome,
    p2: CircuitGenome,
    best_keep_rate: float = 0.75,
    other_keep_rate: float = 0.25,
    c1: float = -1.0,
    c2: float = 0.5,
):
    return None

def n_ary_crossover(
    child: CircuitGenome,
    parents: list[CircuitGenome],
    best_keep_rate: float = 0.75,
    other_keep_rate: float = 0.25,
    c1: float = -1.0,
    c2: float = 0.5,
):
    """
    This recombines 2 or more parent genomes into a new child genome. If a gate appears
    in the best parent and at least one other parent it will always pass to the child. If
    it is only in the best fit parent but not the others, it will be kept with the best_keep_rate,
    if it only appears in non-best fit parent, it will pass to the child at other_keep_rate.

    Args:
        child: an empty CircuitGenome to have gates added to by the parents.
        parents: a list (2 or more) of genomes to recombine.
        best_keep_rate: is how frequently a gate from the best fit parent (but no other
            parents) will be added to the child genome.
        other_keep_rate: is how frequently a gate not in the best fit parent will be
            added to the child genome.
        c1: line search parameter for how far ahead of the more fit weight the
            randomized line search can potentially go
        c2: line search parameter for how far past the less fit weight the
            randomized line search can potentially go
    """

    parents = sorted(parents, key=lambda parent: parent.fitness['fidelity_loss'])
    print("parent fitnesses now:")
    for parent in parents:
        print(f"\t{parent.fitness}")


    # get the random value for the randomized simplex line search
    r = (random.uniform(0.0, 1.0) * (c2 - c1)) + c1

    print(f"r for randomized line search: {r}")

    return None
    weights_avg = None
    weights_std = None

    for node_or_edge in genome.nodes + genome.edges:
        # add the weights from each parent that has the node
        # to the list of recombination weights (which will be
        # in order of parent fitness)
        recombination_weights = []

        for parent in parents:
            # get the weights from the parent node or edge
            if isinstance(node_or_edge, Node):
                if node_or_edge.innovation_number in parent.node_map.keys():
                    recombination_weights.append(
                        parent.node_map[node_or_edge.innovation_number].weights
                    )
                else:
                    if node_or_edge.innovation_number in parent.edge_map.keys():
                        recombination_weights.append(
                            parent.edge_map[node_or_edge.innovation_number].weights
                        )

        weights = node_or_edge.weights

        if len(recombination_weights) == 0:
            # this component (node or edge) came from none of the parents - which can
            # happen if the crossover operation needs to connect a node without any
            # input or output edges.

            if weights_avg is None:
                # only need to get the distribution once
                weights_avg, weights_std = genome.get_weight_distribution(
                    min_weight_std_dev=self.min_weight_std_dev
                )

                for i in range(len(weights)):
                    if weights[i] is None:
                        weights[i] = torch.tensor(
                            (torch.randn(1).item() * weights_std) + weights_avg,
                            requires_grad=True,
                        )
                    print(
                        f"\tweight normal random wtih avg {weights_avg} and std {weights_std} "
                        f"set to: {weights[i]}"
                    )
            else:
                # this component can be initialized by the parental weights
                more_fit_weights = recombination_weights[0]
                other_weights = recombination_weights[1:]

                for i in range(len(weights)):
                    if len(other_weights) == 0:
                        # there are no other weights so just keep the best ones
                        weights[i] = torch.tensor(
                            more_fit_weights[i].detach().clone(), requires_grad=True
                        )
                    else:
                        # get the average of the non-best weights
                        weight_avg = 0.0
                        for j in range(len(other_weights)):
                            weight_avg += other_weights[j][i]
                        weight_avg /= len(other_weights)

                        diff = weight_avg - more_fit_weights[i]
                        line_search_value = (r * diff) + more_fit_weights[i]
                        print(
                            f"\tline search value: {line_search_value}, c1: {self.c1}, c2: {self.c2}, "
                            f"r: {r}, diff: {diff}"
                        )

                        weights[i] = torch.tensor(
                            line_search_value,
                            requires_grad=True,
                        )
