import numpy as np

def marginal_calibration_eeg(generated: np.ndarray, real: np.ndarray) -> np.ndarray:
    """
    Perform marginal calibration for EEG-like data of shape (B, L, C).

    Parameters:
        generated (np.ndarray): Generated EEG samples, shape (B, L, C)
        real (np.ndarray): Real EEG samples, shape (B', L, C)

    Returns:
        np.ndarray: Calibrated generated EEG samples, shape (B, L, C)
    """
    B, L, C = generated.shape
    calibrated = np.zeros_like(generated)

    for c in range(C):         # For each channel
        for l in range(L):     # For each time step
            gen_col = generated[:, l, c]
            real_col = real[:, l, c]

            # ECDF from generated
            gen_ranks = np.argsort(np.argsort(gen_col))
            gen_cdf = gen_ranks / (B - 1)

            # Inverse CDF from real data
            sorted_real = np.sort(real_col)
            real_quantiles = np.interp(gen_cdf, np.linspace(0, 1, len(sorted_real)), sorted_real)

            calibrated[:, l, c] = real_quantiles

    return calibrated
