"""Native materials multimode capture extension boundary."""


def capture_kernel(
    initial,
    final,
    force_initial,
    force_final,
    coupling,
    temperature,
    energies,
    width,
    masses=None,
    queries=(),
):
    """Implement the physical kernel and native-input response in the Harbor instruction."""
    raise NotImplementedError("Implement native multimode quantum capture kernel")
