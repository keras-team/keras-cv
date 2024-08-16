class SD3LatentFormat:
    """Latents are slightly shifted from center.

    This class must be called after VAE Decode to correct for the shift.
    """

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor
