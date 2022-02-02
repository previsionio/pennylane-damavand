# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Simulator
========

**Simulator:** :mod:`.device1`

.. currentmodule:: damavand.simulator

This Device implements all the :class:`~pennylane.device.Device` methods,
for using Target Framework device/simulator as a PennyLane device.

It can inherit from the abstract FrameworkDevice to reduce
code duplication if needed.


See https://pennylane.readthedocs.io/en/latest/API/overview.html
for an overview of Device methods available.

Classes
-------

----
"""

# we always import NumPy directly
import numpy as np

from pennylane import QubitDevice
from pennylane.ops import QubitStateVector, BasisState

from damavand import Circuit
from mpi4py import MPI

from ._version import __version__


class DamavandQubitSimulator(QubitDevice):
    r"""Damavand Simulator for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.
        additional_option (float): as many additional arguments can be
            added as needed
        specific_option_for_device1 (int): another example
    """
    name = "Damavand Qubit Simulator for PennyLane"
    short_name = "damavand.qubit"
    author = "Michel Nowak"
    version = __version__
    pennylane_requires = __version__

    _operation_map = {
        "QubitStateVector": None,
        "BasisState": None,
        "CNOT": None,
        "S": None,
        "T": None,
        "RX": None,
        "RY": None,
        "RZ": None,
        "PauliX": None,
        "PauliY": None,
        "PauliZ": None,
        "Hadamard": None,
    }

    _observable_map = {
        "PauliX": None,
        "PauliY": None,
        "PauliZ": None,
        "Hadamard": None,
    }

    operations = _operation_map.keys()
    observables = _observable_map.keys()

    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard"}

    _circuits = {}

    def __init__(self, wires, apply_method="multithreading", shots=1000):
        super().__init__(wires, shots=shots)
        self.apply_method = apply_method
        self._circuit = Circuit(wires, apply_method=apply_method)

        real_part_state = self._circuit.get_real_part_state()
        imaginary_part_state = self._circuit.get_real_part_state()
        self._state = [complex(r, i) for r, i in zip(real_part_state, imaginary_part_state)]

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            supports_reversible_diff=False,
            supports_inverse_operations=False,
            supports_analytic_computation=False,
            returns_state=True,
            returns_probs=True,
        )
        return capabilities


    def apply(self, operations, **kwargs):
        rotations = kwargs.get("rotations", [])

        self._circuit.reset()

        # apply the circuit operations
        for i, operation in enumerate(operations):

            if i > 0 and isinstance(operation, (QubitStateVector, BasisState)):
                raise DeviceError(
                    f"Operation {operation.name} cannot be used after other Operations have already been applied "
                    f"on a {self.short_name} device."
                )

            if isinstance(operation, QubitStateVector):
                self._apply_state_vector(operation.parameters[0], operation.wires)
            elif isinstance(operation, BasisState):
                self._apply_basis_state(operation.parameters[0], operation.wires)
            else:
                self._state = self.apply_operation(self._state, operation)

        self._circuit.forward()

        # store the pre-rotated state
        # we are forced to retrieve real and imaginary parts separatly because
        # Rust PyO3 binding does not support Complex<f64> to python cast
        real_part_state = self._circuit.get_real_part_state()
        imaginary_part_state = self._circuit.get_real_part_state()
        self._pre_rotated_state = [complex(r, i) for r, i in zip(real_part_state, imaginary_part_state)]

    def extract_expectation_values(self, samples):
        return self._circuit.extract_expectation_values(samples)

    def apply_operation(self, state, operation):
        """Applies operations to the input state.

        Args:
            state (array[complex]): input state
            operation (~.Operation): operation to apply on the device

        Returns:
            array[complex]: output state
        """
        op_name = operation.name
        wires = operation.wires
        par = operation.data

        if op_name == "PauliX":
            self._circuit.add_pauli_x_gate(wires[0], measure=False)
        elif op_name == "PauliY":
            self._circuit.add_pauli_y_gate(wires[0], measure=False)
        elif op_name == "PauliZ":
            self._circuit.add_pauli_z_gate(wires[0], measure=False)
        elif op_name == "Hadamard":
            self._circuit.add_hadamard_gate(wires[0])
        elif op_name == "S":
            self._circuit.add_s_gate(wires[0])
        elif op_name == "T":
            self._circuit.add_t_gate(wires[0])
        elif op_name == "RX":
            self._circuit.add_rotation_x_gate(wires[0], par[0])
        elif op_name == "RY":
            self._circuit.add_rotation_y_gate(wires[0], par[0])
        elif op_name == "RZ":
            self._circuit.add_rotation_z_gate(wires[0], par[0])
        elif op_name == "Rot":
            self._circuit.add_rotation_z_gate(wires[0], par[0])
            self._circuit.add_rotation_y_gate(wires[0], par[1])
            self._circuit.add_rotation_z_gate(wires[0], par[2])
        elif op_name == "CNOT":
            self._circuit.add_cnot_gate(wires[0], wires[1])
        elif op_name == "Unitary":
            if len(wires) > 0:
                raise ValueError("Unitary operations are only supported on one qubit.")

            real_part_unitary = [u.real for u in operation.matrix]
            imaginary_part_unitary = [u.real for u in operation.matrix]
            self._circuit.add_unitary_gate(wires[0], real_part_unitary, imaginary_part_unitary)

    @property
    def state(self):
        real_part_state = self._circuit.get_real_part_state()
        imaginary_part_state = self._circuit.get_imaginary_part_state()
        return self._convert_to_pennylane_convention(
                [complex(r, i) for r, i in zip(real_part_state, imaginary_part_state)])

    def get_fidelity_between_two_states_with_parameters(self, parameters_1, parameters_2):
        return self._circuit.get_fidelity_between_two_states_with_parameters(
                parameters_1, parameters_2)

    def reset(self):
        self._samples = None
        self._circuit.reset()

    def expval(self, observable, **kwargs):
        if self.shots is None:
            raise ValueError("Analytic not implemented yet")

        # estimate the ev
        return np.squeeze(np.mean(self.sample(observable), axis=0))

    def analytic_probability(self, wires=None):
        measure = self._circuit.measure()
        probas = measure / np.sum(measure)
        return self._convert_to_pennylane_convention(probas)

    def _convert_to_pennylane_convention(self, vector):
        """Reverse the qubit order for a vector of amplitudes.

        Taken from Pennylane-Qulacs plugin:
        https://github.com/PennyLaneAI/pennylane-qulacs/blob/master/pennylane_qulacs/qulacs_device.py

        Args:
            state_vector (iterable[complex]): vector containing the amplitudes
        Returns:
            list[complex]
        """
        vector = np.array(vector)
        N = int(np.log2(len(vector)))
        reversed_state = vector.reshape([2] * N).T.flatten()
        return reversed_state
