import numpy as np


def causal_delta_modulate(signal, window_size=1024, delta_base=0.05, alpha=1.20):
    x = np.asarray(signal, dtype=np.float64).ravel()
    n = len(x)
    if n <= window_size:
        raise ValueError("Signal length must be greater than window_size.")

    spikes = np.zeros(n, dtype=np.int8)
    x_ref = np.zeros(n, dtype=np.float32)
    deltas = np.zeros(n, dtype=np.float32)

    ref = x[0]
    x_ref[0] = ref

    p_sum = x[0]
    p_sq = x[0] ** 2
    for k in range(1, window_size):
        mu = p_sum / k
        var = max(0.0, (p_sq / k) - mu * mu)
        dk = delta_base + alpha * np.sqrt(var)
        deltas[k] = dk

        diff = x[k] - ref
        if diff >= dk:
            spikes[k] = 1
            ref += dk
        elif diff <= -dk:
            spikes[k] = -1
            ref -= dk

        x_ref[k] = ref
        p_sum += x[k]
        p_sq += x[k] * x[k]

    buf_sum = np.sum(x[:window_size])
    buf_sq = np.sum(x[:window_size] ** 2)

    for k in range(window_size, n):
        mu = buf_sum / window_size
        var = max(0.0, (buf_sq / window_size) - mu * mu)
        dk = delta_base + alpha * np.sqrt(var)
        deltas[k] = dk

        diff = x[k] - ref
        if diff >= dk:
            spikes[k] = 1
            ref += dk
        elif diff <= -dk:
            spikes[k] = -1
            ref -= dk

        x_ref[k] = ref

        old_v = x[k - window_size]
        new_v = x[k]
        buf_sum += new_v - old_v
        buf_sq += new_v * new_v - old_v * old_v

    spikes = spikes[window_size:]
    x_ref = x_ref[window_size:]
    deltas = deltas[window_size:]

    s_pos = (spikes == 1).astype(np.float32)
    s_neg = (spikes == -1).astype(np.float32)
    dual_spikes = np.stack([s_pos, s_neg], axis=-1)

    return dual_spikes, spikes, x_ref, deltas


def static_delta_modulate(signal, threshold=0.15):
    x = np.asarray(signal, dtype=np.float64).ravel()
    spikes = np.zeros(len(x), dtype=np.int8)
    ref = x[0]

    for k in range(1, len(x)):
        diff = x[k] - ref
        if diff >= threshold:
            spikes[k] = 1
            ref += threshold
        elif diff <= -threshold:
            spikes[k] = -1
            ref -= threshold

    s_pos = (spikes == 1).astype(np.float32)
    s_neg = (spikes == -1).astype(np.float32)
    return np.stack([s_pos, s_neg], axis=-1), spikes