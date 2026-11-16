# ⚛️ Hybrid Quantum-Classical Multi-Class Image Classifier Dashboard (Digits 0–9)

A full-stack, production-ready Python application implementing a **Variational Quantum-Classical Neural Network (VQC-CNN)** to classify handwritten digits across all 10 classes (**Digits 0 through 9**). Built with **PyTorch**, **PennyLane**, and **Streamlit**.

---

## 🏗️ Architecture Pipeline

```text
Input Image (28x28 Grayscale)
           │
           ▼
┌──────────────────────────────────────────────┐
│       Classical Feature Extractor (CNN)      │
│  - Conv2D (1 -> 16, 3x3) + ReLU + MaxPool2D  │
│  - Conv2D (16 -> 32, 3x3) + ReLU + MaxPool2D │
│  - Flatten (32 * 7 * 7 = 1568 features)      │
│  - Linear Projection (1568 -> 4 features)    │
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│           Feature Angle Normalization        │
│       θ_i = tanh(f_i) * (π / 2)             │
│       Scaled to [-π/2, π/2] radians          │
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│      PennyLane Variational Quantum Circuit   │
│  - 4 Qubits (`default.qubit` simulator)      │
│  - Data Encoding: qml.AngleEmbedding (Ry)    │
│  - Ansatz: qml.BasicEntanglerLayers (2 L)    │
│  - Measurement: 4x PauliZ Expectation Values │
│  - Gradients: End-to-end PyTorch Autograd    │
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│        Multi-Class Classification Head       │
│  - Linear Projection (4 -> 10 classes)       │
│  - Softmax Probability Distribution (0 to 9) │
└──────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
├── app.py                   # Multi-class Streamlit dashboard with live circuit rendering
├── model.py                 # 10-Class HybridQNN architecture & PennyLane QNode definition
├── train.py                 # Multi-class training script on full MNIST (Digits 0–9)
├── requirements.txt         # Pinned dependency requirements
├── model_multiclass.pth     # Trained 10-class model checkpoint
└── README.md                # System documentation
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup

#### **Windows (PowerShell)**:
```powershell
# Create & activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

### 2. Multi-Class Training

Train the model across all 10 digits (0 to 9):

```powershell
python train.py --epochs 5 --subset-train 3000 --subset-test 1000 --batch-size 32
```

- Trained weights are saved to `model_multiclass.pth`.
- To train on the entire MNIST training set (60,000 samples), pass `--subset-train 0`.

---

### 3. Launch Dashboard

Run the Streamlit application:

```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🌟 Dashboard Features

1. **Full 10-Class Classification (Digits 0–9):**
   - **Preset Digits (0 to 9):** Choose any digit from the dropdown to test an authentic sample immediately.
   - **Custom Upload:** Upload custom grayscale PNG/JPG digit images.
   - **Color Inversion Toggle:** Inverts black ink on white paper to white-on-black MNIST format.
2. **Intermediate Quantum State Telemetry:**
   - 4 classical latent features from the CNN.
   - 4 encoded rotation angles $\theta_0, \theta_1, \theta_2, \theta_3 \in [-\pi/2, \pi/2]$.
   - 4 quantum expectation values $\langle Z_i \rangle \in [-1, 1]$.
3. **Full Probability Distribution:**
   - Highlights the predicted class and confidence percentage.
   - Displays all 10 class probabilities as interactive progress bars.
4. **Dynamic Quantum Circuit Schematic:**
   - Matplotlib visualization rendered via PennyLane's `qml.draw_mpl` showing wires, entangling CNOT gates, and measurement meters.
