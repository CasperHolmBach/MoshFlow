"""Start-up file for Visual Studio (F5): launches the MoshFlow GUI.

Kept compatible with Python 3.8+ so pressing F5 under an older interpreter
(such as Visual Studio's bundled Python 3.9) either hands off to a local
virtual environment or prints a clear setup guide instead of a syntax error.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

MIN_PYTHON = (3, 10)
REQUIRED = (
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("PySide6", "PySide6"),
)
REEXEC_GUARD = "MOSHFLOW_REEXEC"
ROOT = Path(__file__).resolve().parent


def _missing_packages():
    missing = []
    for module_name, package_name in REQUIRED:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def _venv_pythons(root):
    rel_paths = (Path("Scripts") / "python.exe", Path("bin") / "python")
    try:
        current = Path(sys.executable).resolve()
    except OSError:
        current = None
    found = []
    for folder in (".venv", "venv", "env"):
        for rel in rel_paths:
            candidate = root / folder / rel
            if not candidate.is_file():
                continue
            try:
                if current is not None and candidate.resolve() == current:
                    continue
            except OSError:
                pass
            found.append(candidate)
    return found


def _python_is_ready(executable):
    check = (
        "import importlib.util, sys; "
        "ok = sys.version_info >= (3, 10) and all("
        "importlib.util.find_spec(m) is not None for m in ('numpy', 'cv2', 'PySide6')); "
        "sys.exit(0 if ok else 1)"
    )
    try:
        result = subprocess.run(
            [str(executable), "-c", check],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _find_ready_venv(root):
    for candidate in _venv_pythons(root):
        if _python_is_ready(candidate):
            return candidate
    return None


def _reexec(executable, args):
    env = os.environ.copy()
    env[REEXEC_GUARD] = "1"
    return subprocess.call([str(executable), str(Path(__file__).resolve())] + list(args), env=env)


def _print_setup_guide(version_too_old, missing):
    version_text = "{}.{}.{}".format(*sys.version_info[:3])
    print("MoshFlow could not start with the active Python environment.", file=sys.stderr)
    print("  Active Python: {} ({})".format(version_text, sys.executable), file=sys.stderr)
    if version_too_old:
        print(
            "  Problem: MoshFlow requires Python {}.{} or newer (3.12 recommended).".format(*MIN_PYTHON),
            file=sys.stderr,
        )
    if missing:
        print("  Missing packages: {}".format(", ".join(missing)), file=sys.stderr)
    print("", file=sys.stderr)
    print("How to set up in Visual Studio:", file=sys.stderr)
    print("  1. In Solution Explorer, right-click Python Environments -> Add Environment...", file=sys.stderr)
    if version_too_old:
        print(
            "  2. Pick a Python 3.10+ base interpreter (do not use Visual Studio's bundled Python 3.9)",
            file=sys.stderr,
        )
        print("     and install from requirements.txt, or choose Existing environment -> .venv.", file=sys.stderr)
    else:
        print("  2. Right-click the active environment -> Install from requirements.txt", file=sys.stderr)
        print("     (or choose Add Environment -> Existing environment -> .venv).", file=sys.stderr)
    print("  3. Activate the environment and press F5 again.", file=sys.stderr)
    print("", file=sys.stderr)
    print("Or from a terminal in the project folder:", file=sys.stderr)
    if version_too_old:
        print("  py -3.12 -m venv .venv", file=sys.stderr)
        print("  .venv\\Scripts\\python -m pip install -r requirements.txt", file=sys.stderr)
        print("  .venv\\Scripts\\python run.py", file=sys.stderr)
    else:
        print('  "{}" -m pip install -r requirements.txt'.format(sys.executable), file=sys.stderr)


def _offer_install(root):
    requirements = root / "requirements.txt"
    if not requirements.is_file() or not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        answer = input("\nInstall packages from requirements.txt into this Python now? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if answer not in ("y", "yes"):
        return False
    code = subprocess.call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
    return code == 0 and not _missing_packages()


def launch(args=None):
    args = sys.argv[1:] if args is None else list(args)
    version_too_old = sys.version_info < MIN_PYTHON
    missing = _missing_packages()

    if version_too_old or missing:
        if os.environ.get(REEXEC_GUARD) != "1":
            venv_python = _find_ready_venv(ROOT)
            if venv_python is not None:
                print("MoshFlow: switching to local virtual environment ({})".format(venv_python))
                return _reexec(venv_python, args)
        _print_setup_guide(version_too_old, missing)
        if not version_too_old and _offer_install(ROOT):
            pass
        else:
            return 1

    from moshflow.__main__ import main

    return main()


if __name__ == "__main__":
    sys.exit(launch())
