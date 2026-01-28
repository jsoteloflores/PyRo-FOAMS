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
import sys


def main():
    args = sys.argv[1:]
    
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("Commands:")
        print("  (default)     Launch preprocessing GUI")
        print("  processing    Launch processing GUI")
        print("  stereology    Launch stereology GUI")
        print("  postprocess   Launch mask editor GUI")
        return 0
    
    cmd = args[0].lower()
    
    if cmd == "processing":
        # Processing requires images to be passed in; launch preprocessing instead
        print("Note: Processing GUI requires images. Launching preprocessing first.")
        from . import preprocessinggui
        preprocessinggui.main()
    elif cmd == "stereology":
        print("Note: Stereology GUI requires labels. Launching preprocessing first.")
        from . import preprocessinggui
        preprocessinggui.main()
    elif cmd == "postprocess":
        print("Note: Postprocessing GUI requires images/masks. Launching preprocessing first.")
        from . import preprocessinggui
        preprocessinggui.main()
    else:
        # Default: launch preprocessing GUI
        from . import preprocessinggui
        preprocessinggui.main()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
