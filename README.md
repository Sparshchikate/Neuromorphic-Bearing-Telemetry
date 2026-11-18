# Adaptive Event-Driven Neuromorphic Telemetry

Official implementation of **"Adaptive Event-Driven Neuromorphic Telemetry: Energy-Sparse Spiking Neural Networks with Dynamic Threshold Encoding for Industrial Bearing Diagnostics"** (IC-AIDSA 2026).

## Architecture Overview

The framework couples a causal dynamic-threshold delta modulator with a 2-layer Recurrent Leaky Integrate-and-Fire (rLIF) Spiking Neural Network for low-power industrial bearing fault diagnostics on 12 kHz Case Western Reserve University (CWRU) vibration data.

1. **Causal Dynamic-Threshold Delta Modulator (`delta_modulator.py`)**
   - Tracks trailing rolling population standard deviation $\sigma[k]$ over $W = 1024$ past samples ($85.33\text{ ms}$ at $12\text{ kHz}$).
   - Adapts trigger boundary dynamically: $\Delta[k] = \Delta_{\text{base}} + \alpha \cdot \sigma[k]$ ($\Delta_{\text{base}} = 0.05\text{ g}$, $\alpha = 1.20$).
   - Emits ternary events $s[k] \in \{-1, 0, +1\}$ rectified into dual binary spike channels $(s^+[k], s^-[k])$.

2. **Recurrent LIF Spiking Classifier (`rlif_model.py`)**
   - Topology: $2 \rightarrow 128 \rightarrow 64 \rightarrow 10$ with intra-layer recurrent weights $V^{(1)}, V^{(2)}$.
   - Neuron parameters: decay $\beta = 0.85$, threshold $U_{\text{th}} = 1.00\text{ V}$, subtractive reset.
   - Trained via Backpropagation Through Time (BPTT) with a fast-sigmoid surrogate gradient ($\gamma = 2.0$).

3. **Leakage-Free CWRU Evaluation (`data_loader.py`, `train_and_evaluate.py`)**
   - Evaluates 10 classes (Normal + Inner Race, Ball, and Outer Race defects at 7, 14, and 21 mils) across 4 motor loads (`0, 1, 2, 3 HP`).
   - Uses 5-fold cross-validation partitioned strictly at the recording-file level.

## Benchmark Summary (CWRU 12 kHz, 5-Fold CV)

| Architecture | Mean Accuracy | Synaptic Sparsity | Est. Inference Energy | Battery Autonomy Model |
| :--- | :---: | :---: | :---: | :---: |
| Standard 1D-CNN | $99.35\% \pm 0.11\%$ | $0.0\%$ | $14.80\ \mu\text{J}$ | ~17.5 days (raw stream) |
| Quantized 8-Bit CNN | $97.48\% \pm 0.18\%$ | $0.0\%$ | $3.92\ \mu\text{J}$ | ~75 days (duty-cycled) |
| Feedforward SNN | $94.60\% \pm 0.32\%$ | $78.4\%$ | $0.91\ \mu\text{J}$ | — |
| **Proposed Adaptive rLIF** | **$99.12\% \pm 0.14\%$** | **$98.73\%$** | **$0.48\ \mu\text{J}$** | **3.40 yrs (analytical proj.)** |

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt