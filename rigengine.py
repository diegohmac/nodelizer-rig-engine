# The rig engine's entry point. The app runs one step per process, for example:
#     python.exe -I -B -u -X utf8 <engine folder>\rigengine.py read --model skintokens ...
# See PROTOCOL.md. (Not named run.py: the Windows kit's Python treats any script called run.py as
# UniRig's and imports PyTorch at start up.)
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if __name__ == "__main__":
    from engine.cli import main

    main(sys.argv[1:])
