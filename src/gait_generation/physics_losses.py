import torch

def compute_magnitude(windows):
    return torch.sqrt((windows ** 2).sum(dim=-1) + 1e-8)

def smoothness_loss(recon_windows):
    a_mag = compute_magnitude(recon_windows)
    d2 = a_mag[:, 2:] - 2 * a_mag[:, 1:-1] + a_mag[:, :-2]
    return (d2 ** 2).mean()

def distribution_loss(input_windows, recon_windows):
    mag_in = compute_magnitude(input_windows)
    mag_out = compute_magnitude(recon_windows)
    mean_diff = (mag_in.mean(dim=1) - mag_out.mean(dim=1)).abs().mean()
    std_diff = (mag_in.std(dim=1) - mag_out.std(dim=1)).abs().mean()
    return mean_diff + std_diff

def spectral_loss(input_windows, recon_windows, fs: float = 100.0):
    mag_in = compute_magnitude(input_windows)
    mag_out = compute_magnitude(recon_windows)

    mag_in = mag_in - mag_in.mean(dim=1, keepdim=True)
    mag_out = mag_out - mag_out.mean(dim=1, keepdim=True)
    hann = torch.hann_window(mag_in.shape[1], device=mag_in.device)
    mag_in = mag_in * hann
    mag_out = mag_out * hann

    fft_in = torch.fft.rfft(mag_in, dim=1)
    fft_out = torch.fft.rfft(mag_out, dim=1)

    psd_in = (fft_in.abs() ** 2)
    psd_out = (fft_out.abs() ** 2)

    psd_in = psd_in / (psd_in.sum(dim=1, keepdim=True) + 1e-8)
    psd_out = psd_out / (psd_out.sum(dim=1, keepdim=True) + 1e-8)

    return torch.mean((psd_in - psd_out) ** 2)

