import random

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.circuits.gate import Gate


def exponential_crossover(
    child: CircuitGenome,
    p1: CircuitGenome,
    p2: CircuitGenome,
):
    """
    This recombines 2 parent genomes into a new child genome. A random depth is
    selected and all gates below that depth from p1 are added to the child genome,
    and all gates greater than or equal to that depth from p2 are added to the child.

    Args:
        child: an empty CircuitGenome to have gates added to by the parents.
        p1: The first parent to recombine.
        p2: The second parent to recombine.
    """

    crossover_depth = random.uniform(0, 1.0)

    logger.debug(f"exponential crossover at depth: {crossover_depth}")

    # add all gates in p1 with less depth than the crossover gate
    for gate in p1.gates:
        logger.debug(f"p1 gate {gate.innovation_number} at depth: {gate.depth}")
        if gate.depth < crossover_depth:
            logger.debug("\tadding!")
            child.add_existing_gate(gate)
        else:
            logger.debug("\tnot adding.")

    for gate in p2.gates:
        logger.debug(f"p2 gate {gate.innovation_number} at depth: {gate.depth}")
        if gate.depth >= crossover_depth:
            logger.debug("\tadding!")
            child.add_existing_gate(gate)
        else:
            logger.debug("\tnot adding.")

    return child


def binary_crossover(
    child: CircuitGenome,
    p1: CircuitGenome,
    p2: CircuitGenome,
    best_keep_rate: float = 0.75,
    other_keep_rate: float = 0.25,
    c1: float = -1.0,
    c2: float = 0.5,
):
    """
    This recombines 2 parent genomes into a new child genome. If a gate appears
    in the best parent and the other parent it will always pass to the child. If
    it is only in the best fit parent but not the other, it will be kept with the best_keep_rate,
    if it only appears in other parent, it will pass to the child at other_keep_rate.

    Args:
        child: an empty CircuitGenome to have gates added to by the parents.
        p1: The first parent to recombine. This should be the more fit parent of the two.
        p2: The second parent to recombine.
        best_keep_rate: is how frequently a gate from the best fit parent (but no other
            parents) will be added to the child genome.
        other_keep_rate: is how frequently a gate not in the best fit parent will be
            added to the child genome.
        c1: line search parameter for how far ahead of the more fit weight the
            randomized line search can potentially go
        c2: line search parameter for how far past the less fit weight the
            randomized line search can potentially go
    """

    logger.debug("binary crossover with parents:")
    logger.debug(f"\tp1 fitness: {p1.fitness}")
    logger.debug(f"\tp2 fitness: {p2.fitness}")

    # get the random value for the randomized simplex line search
    r = (random.uniform(0.0, 1.0) * (c2 - c1)) + c1

    best_gates_by_innovation = {}
    for gate in p1.gates:
        best_gates_by_innovation[gate.innovation_number] = gate

    logger.debug(f"best parent has {len(p1.gates)}: {best_gates_by_innovation.keys()}")

    other_gates_by_innovation = {}
    for gate in p2.gates:
        other_gates_by_innovation[gate.innovation_number] = gate
    logger.debug(
        f"other parent has {len(p2.gates)}: {other_gates_by_innovation.keys()}"
    )

    # add gates from the best fit parent to the child
    for i, gate in best_gates_by_innovation.items():
        if i in other_gates_by_innovation.keys():
            logger.debug(f"adding gate innovation {i} because it is in both parents")
            # this gate is in the other circuit, always add it
            child_gate = gate.copy()

            # if the gate has parameters, recombine their
            # values
            if len(child_gate.parameters) > 0:
                other_gate = other_gates_by_innovation[i]

                for p, v in child_gate.parameters.items():
                    # difference between the less fit gate parameter value and the more fit
                    # parameter value
                    diff = other_gate.parameters[p] - v

                    # do randomized line search along that gradient with our r value
                    line_search_value = (r * diff) + v

                    child_gate.parameters[p] = line_search_value

                    logger.debug(
                        f"setting parameter {p} to {line_search_value}, p1 value: {v}, "
                        f"p2 value: {other_gate.parameters[p]}, r was: {r}"
                    )

            child.add_existing_gate(child_gate)

            # remove this gate from the other gates so we don't try to
            # re-add it
            del other_gates_by_innovation[i]

        elif random.uniform(0.0, 1.0) < best_keep_rate:
            # this gate isn't in the other parent but we're randomly
            # selecting to keep it based on the rate. parameters
            # will just be inherited from the parent
            logger.debug(
                f"adding gate innovation {i} because it is only in best parent but randomly selected"
            )
            child_gate = gate.copy()
            child.add_existing_gate(child_gate)

        # gate will not be added otherwise

    for i, gate in other_gates_by_innovation.items():
        if random.uniform(0.0, 1.0) < other_keep_rate:
            # this gate isn't in the best fit parent but we're randomly
            # selecting to keep it based on the rate. parameters
            # will just be inherited from the parent
            logger.debug(
                f"adding gate innovation {i} because it is only in other parent but randomly selected"
            )
            child_gate = gate.copy()
            child.add_existing_gate(child_gate)

    child_gate_innovations = [gate.innovation_number for gate in child.gates]

    logger.debug(f"child genome has {len(child.gates)} gates: {child_gate_innovations}")

    return child


def n_ary_crossover(
    child: CircuitGenome,
    parents: list[CircuitGenome],
    primary_keep_rate: float = 0.75,
    other_keep_rate: float = 0.25,
    c1: float = -1.0,
    c2: float = 0.5,
):
    """
    This recombines 2 or more parent genomes into a new child genome. The first parent circuit is
    considered the primary circuit. If a gate is in the primary parent and at least one other parent
    it will always pass to the child. it is only in the primary parent but not the others, it will be
    kept with the primary_keep_rate, if it only appears in the other parents, it will pass to the child
    at other_keep_rate.

    Args:
        child: an empty CircuitGenome to have gates added to by the parents.
        parents: a list (2 or more) of genomes to recombine, these should be ordered by fitness,
            so the first parent in the list is the primary (best fit) parent.
        primary_keep_rate: is how frequently a gate from the primary parent (but no other
            parents) will be added to the child genome.
        other_keep_rate: is how frequently a gate not in the primary parent will be
            added to the child genome.
        c1: line search parameter for how far ahead of the more primary weight the
            randomized line search can potentially go.
        c2: line search parameter for how far past the average of the other parent weights
            the  randomized line search can potentially go.
    """

    logger.debug("doing n-ary crossover")
    logger.debug("parent fitnesses now:")
    for parent in parents:
        logger.debug(f"\t{parent.fitness}")

    primary = parents[0]
    others = parents[1:]
    logger.debug(f"primary fitnesses: {primary.fitness}")
    for other in others:
        logger.debug(f"other fitness: {parent.fitness}")

    # get the random value for the randomized simplex line search
    r = (random.uniform(0.0, 1.0) * (c2 - c1)) + c1

    logger.debug(f"r for randomized line search: {r}")

    primary_gates_by_innovation: dict[int, Gate] = {}
    for gate in primary.gates:
        primary_gates_by_innovation[gate.innovation_number] = gate

    logger.debug(
        f"primary parent has {len(primary.gates)}: {primary_gates_by_innovation.keys()}"
    )

    # keep a list of the other gates so we can average their
    # parameter values to use in parmeter value recombinations
    other_gates_by_innovation: dict[int, list[Gate]] = {}
    for other_parent in others:
        for gate in other_parent.gates:
            if gate.innovation_number not in other_gates_by_innovation.keys():
                other_gates_by_innovation[gate.innovation_number] = []
            other_gates_by_innovation[gate.innovation_number].append(gate)

    logger.debug(
        f"other parents have {len(other_gates_by_innovation)} gates: {other_gates_by_innovation.keys()}"
    )
    logger.debug("other gate counts by innovation number:")
    for i, gate_list in other_gates_by_innovation.items():
        logger.debug(f"\t{i} - {len(gate_list)}")

    # add gates from the best fit parent to the child
    for i, gate in primary_gates_by_innovation.items():
        if i in other_gates_by_innovation.keys():
            logger.debug(f"adding gate innovation {i} because it is in both parents")
            # this gate is in the other circuit, always add it
            child_gate = gate.copy()

            # if the gate has parameters, recombine their
            # values
            if len(child_gate.parameters) > 0:
                other_gates = other_gates_by_innovation[i]

                for p, v in child_gate.parameters.items():
                    # difference between the less fit gate parameter value and the more fit
                    # parameter value
                    avg_value = 0.0
                    for other_gate in other_gates:
                        avg_value += other_gate.parameters[p]
                    avg_value /= len(other_gates)

                    diff = avg_value - v

                    # do randomized line search along that gradient with our r value
                    line_search_value = (r * diff) + v

                    child_gate.parameters[p] = line_search_value

                    logger.debug(
                        f"setting parameter {p} to {line_search_value}, p1 value: {v}, "
                        f"other gates avg value: {avg_value}, r was: {r}"
                    )

            child.add_existing_gate(child_gate)

            # remove this gate from the other gates so we don't try to
            # re-add it
            del other_gates_by_innovation[i]

        elif random.uniform(0.0, 1.0) < primary_keep_rate:
            # this gate isn't in the other parent but we're randomly
            # selecting to keep it based on the rate. parameters
            # will just be inherited from the parent
            logger.debug(
                f"adding gate innovation {i} because it is only in best parent but randomly selected"
            )
            child_gate = gate.copy()
            child.add_existing_gate(child_gate)

        # gate will not be added otherwise

    for i, gate_list in other_gates_by_innovation.items():
        if random.uniform(0.0, 1.0) < other_keep_rate:
            # this gate isn't in the primary primary but we're randomly
            # selecting to keep it based on the rate.
            logger.debug(
                f"adding gate innovation {i} because it is only in other parent but randomly selected"
            )
            child_gate = gate_list[0].copy()

            # if the gate has parameters and there is more than
            # one parent in the other list, use the average of
            # the other gate parameter values. otherwise weights
            # will be directly inherited from the one parent

            # TODO: we could maybe recombine similar to above
            # by taking one as best but that's probably a bit
            # too complicated for not much benefit
            if len(gate_list) > 1 and len(child_gate.parameters) > 0:
                for p, v in child_gate.parameters.items():
                    # difference between the less fit gate parameter value and the more fit
                    # parameter value
                    avg_value = 0.0

                    # TODO: can probably remove tracking the other values for the print
                    # statement once we know this works right
                    other_values = []
                    for other_gate in gate_list:
                        v = other_gate.parameters[p]
                        other_values.append(v)
                        avg_value += v

                    avg_value /= len(gate_list)

                    child_gate.parameters[p] = avg_value

                    logger.debug(
                        f"setting parameter {p} to avg of gate values {avg_value} - other values: {other_values}"
                    )

            child.add_existing_gate(child_gate)

    child_gate_innovations = [gate.innovation_number for gate in child.gates]

    logger.debug(f"child genome has {len(child.gates)} gates: {child_gate_innovations}")

    return child
