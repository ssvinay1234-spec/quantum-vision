"""
model.py: Multi-Class Hybrid Quantum-Classical Neural Network Architecture
Using PyTorch and PennyLane for 10-Class Classification (Digits 0-9).
"""

import math
import torch
import torch.nn as nn
import pennylane as qml

# ---------------------------------------------------------
# Quantum Circuit Specifications
# ---------------------------------------------------------
N_QUBITS = 4
N_LAYERS = 2
N_CLASSES = 10

# Define a 4-qubit quantum simulator device using PennyLane
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    4-qubit Variational Quantum Circuit:
    1. Data Encoding: AngleEmbedding mapping 4 classical features into Y-rotation angles.
    2. Variational Ansatz: BasicEntanglerLayers with 2 layers of parameterized weights.
    3. Measurement: PauliZ expectation values across all 4 qubits.
    """
    # 1. Angle Embedding for classical data encoding using Y-rotations
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")

    # 2. Variational Ansatz with single-qubit rotations and entangling CNOT gates
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))

    # 3. Measurement: PauliZ expectation values across all 4 qubits
    return [qml.expval(qml.PauliZ(wire)) for wire in range(N_QUBITS)]


# ---------------------------------------------------------
# Hybrid Quantum-Classical Neural Network (10 Classes)
# ---------------------------------------------------------
class HybridQNN(nn.Module):
    """
    Multi-Class Hybrid QNN Model:
    - Classical CNN Encoder: 2 Conv2D layers (with ReLU & MaxPool) + Linear
      downprojecting 28x28 grayscale inputs into 4 latent features.
    - Feature Scaling: torch.tanh scaled by pi / 2 to constrain angles to [-pi/2, pi/2].
    - PennyLane TorchLayer: Evaluates the variational circuit with full backpropagation.
    - Classification Head: Linear projection from 4 quantum expectation values to 10 class logits.
    """

    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=N_CLASSES):
        super(HybridQNN, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes

        # 1. Classical CNN Feature Extractor
        # Conv Layer 1: 1 -> 16 channels, 28x28 -> 14x14
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Layer 2: 16 -> 32 channels, 14x14 -> 7x7
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # 32 channels * 7 * 7 = 1568 spatial features
        self.fc_reduction = nn.Linear(32 * 7 * 7, n_qubits)

        # 2. PennyLane Variational Quantum Layer Integration
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # 3. Multi-Class Classification Head (4 quantum expectation values -> 10 classes)
        self.fc_out = nn.Linear(n_qubits, n_classes)

    def extract_features(self, x):
        """
        Passes input through the hybrid network and extracts intermediate telemetry:
        - raw CNN latent features
        - rotation angles in [-pi/2, pi/2]
        - quantum expectation values <Z_i>
        - output logits and 10-class softmax probabilities
        """
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = self.flatten(out)
        raw_features = self.fc_reduction(out)

        # Constrain features into rotation angles [-pi/2, pi/2]
        angles = torch.tanh(raw_features) * (math.pi / 2.0)

        # Execute quantum circuit
        quantum_expvals = self.qlayer(angles)

        # Project to 10 class logits and compute probabilities
        logits = self.fc_out(quantum_expvals)
        probabilities = torch.softmax(logits, dim=-1)

        return {
            "raw_features": raw_features,
            "angles": angles,
            "quantum_expvals": quantum_expvals,
            "logits": logits,
            "probabilities": probabilities,
        }

    def forward(self, x, return_probs=False):
        """
        Forward pass through the Hybrid Multi-Class Network.
        Args:
            x (torch.Tensor): Grayscale images of shape (B, 1, 28, 28).
            return_probs (bool): Return Softmax probabilities if True, else raw logits.
        """
        # Classical feature extraction
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = self.flatten(out)
        out = self.fc_reduction(out)

        # Angle normalization
        angles = torch.tanh(out) * (math.pi / 2.0)

        # Quantum layer execution
        quantum_out = self.qlayer(angles)

        # 10-class projection
        logits = self.fc_out(quantum_out)

        if return_probs:
            return torch.softmax(logits, dim=-1)
        return logits

    def predict_proba(self, x):
        """Convenience evaluation method returning 10-class probabilities."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_probs=True)


def draw_circuit(inputs=None, weights=None, style="pennylane"):
    """
    Renders the quantum circuit schematic using PennyLane's qml.draw_mpl.
    """
    if inputs is None:
        inputs = torch.zeros(N_QUBITS)
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().squeeze()
        if inputs.ndim > 1:
            inputs = inputs[0]

    if weights is None:
        weights = torch.zeros((N_LAYERS, N_QUBITS))
    elif isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu()

    fig, ax = qml.draw_mpl(quantum_circuit, style=style)(inputs, weights)
    return fig
