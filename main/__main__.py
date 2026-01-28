# main/__main__.py
# Entry point for running PyRo-FOAMS as a module: python -m main
"""
PyRo-FOAMS - Pore/Foam Image Analysis Toolkit

Usage:
    python -m main              # Launch preprocessing GUI (default)
    python -m main --help       # Show help
    python -m main processing   # Launch processing GUI directly
    python -m main stereology   # Launch stereology GUI directly
"""
from __future__ import annotations

import os
import sys

# Support both "python -m main" (relative imports work) and direct script execution
_RUNNING_AS_PACKAGE = __package__ is not None

if not _RUNNING_AS_PACKAGE:
    # Add parent directory to path so we can import main.gui.*
    _here = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_here)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)


def _import_preprocessing():
    """Import preprocessing module, handling both package and script modes."""
    if _RUNNING_AS_PACKAGE:
        from .gui import preprocessing
    else:
        from main.gui import preprocessing
    return preprocessing


def main():
    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("Commands:")
        print("  (default)     Launch preprocessing GUI")
        print("  processing    Launch processing GUI")
        print("  stereology    Launch stereology GUI")
        print("  postprocess   Launch mask editor GUI")
        return 0

    cmd = args[0].lower() if args else ""

    if cmd == "processing":
        # Processing requires images to be passed in; launch preprocessing instead
        print("Note: Processing GUI requires images. Launching preprocessing first.")
        preprocessing = _import_preprocessing()
        preprocessing.main()
    elif cmd == "stereology":
        print("Note: Stereology GUI requires labels. Launching preprocessing first.")
        preprocessing = _import_preprocessing()
        preprocessing.main()
    elif cmd == "postprocess":
        print("Note: Postprocessing GUI requires images/masks. Launching preprocessing first.")
        preprocessing = _import_preprocessing()
        preprocessing.main()
    else:
        # Default: launch preprocessing GUI
        preprocessing = _import_preprocessing()
        preprocessing.main()

    return 0


if __name__ == "__main__":
    sys.exit(main())
