# Multimode defect transition kernel

This application extends `pymatgen-analysis-defects` 2026.10.16 with synthetic
stable harmonic models. Source: https://github.com/materialsproject/pymatgen-analysis-defects,
commit `71b3c7ab543c0e7833f4e6416d51a3fc626c8b26`.
The upstream license is in `environment/src/UPSTREAM-LICENSE`.

Implement `vibronic_kernel.capture_kernel` under `/app/src`; `instruction.md`
defines the spectrum, native structures, derivatives and tolerances. Runtime
and verifier dependencies are installed from pinned wheels. VASP and external
materials databases are unnecessary.

Run with Harbor and Docker on Linux/amd64. Only `/app/src` is transferred to the
separate verifier. The root verifier runs each candidate call under a distinct
unprivileged UID in a dependency-only chroot, using Docker's standard
SYS_CHROOT, MKNOD, SETUID and SETGID capabilities. Private scratch is writable;
tests, solutions and reward files are inaccessible. Each scientific call has
180 seconds, 2 CPU and 4 GiB.
