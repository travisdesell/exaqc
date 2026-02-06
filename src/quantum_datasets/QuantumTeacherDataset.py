import pennylane as qml
import torch


class QuantumTeacherDataset:

    @staticmethod
    def make_teacher_qnode(
        n_wires: int = 6,
        input_wires: list = [],
        output_wires: list = [],
        teacher_name: str = "bell_out",
        input_mode: str = "angle",
    ):
        """
        Method to create a baseline qnode

        :param n_wires: number of qubits in total
        :type n_wires: int
        :param teacher_name: the type of quantum circuit needed
        :type teacher_name: str
        :param input_mode: which encoding is preferred
        :type input_mode: str
        """

        def _teacher_identity(in_wires: list, out_wires: list):
            qml.Identity(wires=in_wires + out_wires)

        def _teacher_x_out4(in_wires: list, out_wires: list):
            qml.PauliX(wires=out_wires[0])

        def _half_adder(in_wires: list, out_wires: list):
            """
            Half adder with arbitrary wire indices.

            Expects:
              in_wires  : [A_wire, B_wire]
              out_wires : [SUM_wire, CARRY_wire]

            Computes:
              SUM   = A XOR B  onto out_wires[0]
              CARRY = A AND B  onto out_wires[1]

            Assumes SUM and CARRY wires start in |0>.
            """
            if len(in_wires) < 2:
                raise ValueError("half_adder needs at least 2 input wires: [A, B]")
            if len(out_wires) < 2:
                raise ValueError("half_adder needs at least 2 output wires: [SUM, CARRY]")

            a_w, b_w = in_wires[0], in_wires[1]
            s_w, c_w = out_wires[0], out_wires[1]

            used = [a_w, b_w, s_w, c_w]
            if len(set(used)) != 4:
                raise ValueError(
                    f"half_adder requires 4 distinct wires (A,B,SUM,CARRY). Got {used}"
                )

            qml.CNOT(wires=[a_w, s_w])
            qml.CNOT(wires=[b_w, s_w])
            qml.Toffoli(wires=[a_w, b_w, c_w])


        def _teacher_bell_out(in_wires: list, out_wires: list):
            qml.Hadamard(wires=out_wires[0])
            qml.CNOT(wires=[out_wires[0], out_wires[1]])

        def _teacher_copy_in_to_out(in_wires: list, out_wires: list):
            qml.CNOT(wires=[in_wires[0], out_wires[0]])
            qml.CNOT(wires=[in_wires[1], out_wires[1]])

        def _teacher_parity012_to_out4(in_wires: list, out_wires: list):
            qml.CNOT(wires=[in_wires[0], out_wires[0]])
            qml.CNOT(wires=[in_wires[1], out_wires[0]])
            qml.CNOT(wires=[in_wires[2], out_wires[0]])

        def _teacher_input_controlled_bell(in_wires: list, out_wires: list):
            qml.Hadamard(wires=out_wires[0])
            qml.CNOT(wires=[out_wires[0], out_wires[1]])
            qml.CNOT(wires=[in_wires[0], out_wires[0]])

        def _teacher_2layer_out_block(in_wires: list, out_wires: list):
            # fixed angles (constant teacher)
            qml.RY(0.7, wires=out_wires[0])
            qml.RY(1.1, wires=out_wires[1])
            qml.CNOT(wires=[out_wires[0], out_wires[1]])
            qml.RX(0.4, wires=out_wires[0])
            qml.RX(-0.9, wires=out_wires[1])

        def _teacher_grover(in_wires: list, out_wires: list):
            # 1. Apply Hadamard gates to all wires
            for wire in in_wires + out_wires:
                qml.Hadamard(wires=wire)
            # 2. Apply X gates to all wires (flips |0> to |1> to control on |0>)
            for wire in in_wires + out_wires:
                qml.PauliX(wires=wire)
            # 3. Apply a multi-controlled Z gate (MCZ)
            # PennyLane has a built-in MultiControlledZ gate
            qml.MultiControlledZ(wires=in_wires + out_wires)
            # 4. Apply X gates to all wires (resets the control state)
            for wire in in_wires + out_wires:
                qml.PauliX(wires=wire)
            # 5. Apply Hadamard gates to all wires
            for wire in in_wires + out_wires:
                qml.Hadamard(wires=wire)

        def _apply_teacher_body(teacher_name: str, in_wires: list, out_wires: list):
            """Apply the teacher circuit based on teacher_name."""
            teachers = {
                "identity": _teacher_identity,
                "x_out4": _teacher_x_out4,
                "bell_out": _teacher_bell_out,
                "copy_in_to_out": _teacher_copy_in_to_out,
                "parity012_to_out4": _teacher_parity012_to_out4,
                "input_controlled_bell": _teacher_input_controlled_bell,
                "2layer_out_block": _teacher_2layer_out_block,
                "grover": _teacher_grover,
                "half_adder": _half_adder, 
            }

            if teacher_name not in teachers:
                raise ValueError(f"Unknown teacher_name={teacher_name}")

            teachers[teacher_name](in_wires, out_wires)

        dev = qml.device("default.qubit", wires=n_wires)

        in_wires = input_wires
        out_wires = output_wires

        @qml.qnode(dev, interface="torch")
        def teacher(x: torch.Tensor):
            # ----- input encoding -----
            if input_mode == "basis":
                # expects int tensor length 6
                qml.BasisState(x.to(torch.int64), wires=in_wires)
            elif input_mode == "angle":
                # expects float tensor length 4
                for i, w in enumerate(in_wires):
                    qml.RY(torch.pi * x[i], wires=w)
            else:
                raise ValueError(f"Unknown input_mode={input_mode}")

            # ----- teacher body (allowed gates only) -----
            _apply_teacher_body(teacher_name, in_wires, out_wires)

            return qml.state()

        return teacher

    @staticmethod
    def make_teacher_inputs(
        n_train: int = 64,
        n_test: int = 64,
        input_dim: int = 4,
        seed: int = 0,
        grid: bool = False,
    ):
        g = torch.Generator().manual_seed(seed)

        if grid:
            # structured set (more repeatable) – good for debugging
            # create a small grid and slice
            vals = torch.linspace(0.0, 1.0, steps=5)
            X = torch.cartesian_prod(*([vals] * input_dim)).float()
            X = X[torch.randperm(X.size(0), generator=g)]
            train_x = X[:n_train].clone()
            test_x = X[n_train : n_train + n_test].clone()
        else:
            # random continuous
            train_x = torch.rand((n_train, input_dim), generator=g)
            test_x = torch.rand((n_test, input_dim), generator=g)

        return train_x, test_x
