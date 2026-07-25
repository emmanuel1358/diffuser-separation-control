"""Clear the executable-stack flag on JAX shared objects, at image build time.

Run by base/tpu/Dockerfile during the TPU base build, after the jax[tpu] and
JAX-ecosystem installs so it patches the final on-disk jaxlib and libtpu .so.

jaxlib's xla_extension.so ships with its PT_GNU_STACK program header marked
executable (RWX). Under the hardened/sandboxed grading runtime the dynamic
loader refuses to enable an executable stack, so `import jax` fails with "cannot
enable executable stack as shared object requires: Invalid argument"
(jax-ml/jax#26781) -- which breaks the long-running grading process even though a
fresh agent process can import jax. Clearing the X bit (RWX -> RW) on
PT_GNU_STACK makes the import succeed everywhere. JAX does not need an executable
stack; the flag is a packaging artifact, so this is the same change `execstack
-c` would make (execstack is not in the base image).

Best-effort and idempotent: it patches what it can, prints per-.so status, and
never fails the build.
"""

import glob
import os
import struct
import sysconfig

PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def _site_dirs() -> list[str]:
    dirs = set()
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        path = paths.get(key)
        if path:
            dirs.add(path)
    for minor in ("3.12", "3.13"):
        dirs.add(f"/usr/local/lib/python{minor}/site-packages")
    return [d for d in dirs if os.path.isdir(d)]


def _find_sos() -> list[str]:
    sos: list[str] = []
    for directory in _site_dirs():
        for pkg in ("jaxlib", "libtpu", "jax"):
            sos.extend(
                glob.glob(os.path.join(directory, pkg, "**", "*.so"), recursive=True)
            )
    return sorted(set(sos))


def _clear_execstack(path: str) -> str:
    with open(path, "rb") as handle:
        data = bytearray(handle.read())
    if data[:4] != b"\x7fELF":
        return "not-elf"
    if data[4] != 2:  # EI_CLASS: 2 == ELF64
        return "not-elf64"
    endian = "<" if data[5] == 1 else ">"  # EI_DATA: 1 == little-endian
    e_phoff = struct.unpack_from(endian + "Q", data, 0x20)[0]
    e_phentsize = struct.unpack_from(endian + "H", data, 0x36)[0]
    e_phnum = struct.unpack_from(endian + "H", data, 0x38)[0]
    changed = False
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        if off + 8 > len(data):
            break
        p_type = struct.unpack_from(endian + "I", data, off)[0]
        if p_type == PT_GNU_STACK:
            p_flags = struct.unpack_from(endian + "I", data, off + 4)[0]
            if p_flags & PF_X:
                struct.pack_into(endian + "I", data, off + 4, p_flags & ~PF_X)
                changed = True
    if not changed:
        return "already-ok"
    with open(path, "wb") as handle:
        handle.write(data)
    return "cleared"


def main() -> None:
    sos = _find_sos()
    print(f"[clear_execstack] checking {len(sos)} JAX shared object(s)")
    for so in sos:
        try:
            status = _clear_execstack(so)
        except Exception as exc:  # noqa: BLE001 - best-effort, never fail the build
            status = f"error: {type(exc).__name__}: {exc}"
        print(f"[clear_execstack]   {status:11s} {so}")
    print("[clear_execstack] done")


if __name__ == "__main__":
    main()
