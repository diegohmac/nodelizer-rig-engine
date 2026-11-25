# The pinned inputs of a Mac root (darwin-arm64) beyond build/sources.py: the Python it runs on. The
# packages are pinned, each with its SHA-256, in requirements-mac.txt (made by lock.sh from
# requirements-mac.in at the Windows pins). Standard library only.

PYTHON = {
    "name": "cpython",
    "version": "3.11.18",
    "release": "20261110",
    "publisher": "Astral (python-build-standalone), built from the Python Software Foundation's source",
    "file": "cpython-3.11.18+20261110-aarch64-apple-darwin-install_only_stripped.tar.gz",
    "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20261110/"
           "cpython-3.11.18%2B20261110-aarch64-apple-darwin-install_only_stripped.tar.gz",
    # from the release's own SHA256SUMS
    "sha256": "3afb0567b2a2b30b3a7341bd143cb4e8a69f3d8e95d4641218eaa45f0fabbd07",
    "licence": "MIT (python-build-standalone) and the PSF licence (Python)",
}
# The oldest macOS a Mac root runs on: ENGINE.json's `minimum_macos`, which the release manifest can
# repeat so the app checks it before downloading.
# It is the newest LC_BUILD_VERSION minos among the root's Mach-O files (scipy's wheels for macOS 14 and
# PyTorch's libomp.dylib say 14.0; Python and Blender's module say 11.0 and 11.2). The build fails if any
# file needs more (build/mac/macho.py).
MIN_MACOS = "14.0"
# Below the root: the executable the app starts (ENGINE.json's `python`).
EXECUTABLE = "python/bin/python3.11"
SITE_PACKAGES = "python/lib/python3.11/site-packages"

# Installed by the lock's resolution but never imported by SkinTokens' three steps (measured with
# python -X importtime on read, rig and write, a made and a kept skeleton): left out of the root. Each
# name is a package folder or module in site-packages, with its .dist-info. The first group is a
# network client the engine never uses (fsspec's and huggingface_hub's downloads).
NOT_SHIPPED = {
    "aiohttp": "fsspec's HTTP files", "aiohappyeyeballs": "aiohttp's", "aiosignal": "aiohttp's",
    "frozenlist": "aiohttp's", "multidict": "aiohttp's", "propcache": "aiohttp's", "yarl": "aiohttp's",
    "attrs": "aiohttp's", "attr": "attrs' second package",
    "hf_xet": "huggingface_hub's download backend",
    "pytorch_lightning": "the old name of lightning, which SkinTokens imports instead",
}

# Python's own tools that a root never runs: no package is installed into it after the build.
PYTHON_TOOLS_REMOVED = ("pip", "setuptools", "wheel", "_distutils_hack", "distutils-precedence.pth")

# Parts of Python's own archive that no step imports or starts (none is in any step's -X importtime list,
# and nothing in the root links to the libraries), removed by path below the root, each with why. What
# stays of the standard library is what a real import may need (sysconfig, pydoc and _sysconfigdata are
# imported). The links in python/bin, python/lib/pkgconfig and python/share go with what they point at,
# so the root holds no links.
PYTHON_PARTS_REMOVED = {
    "python/lib/python3.11/ensurepip": "pip and setuptools as wheels, to install them; nothing installs",
    "python/lib/python3.11/venv": "makes virtual environments, with ensurepip; never run",
    "python/lib/python3.11/idlelib": "IDLE, the editor; needs Tk",
    "python/lib/python3.11/tkinter": "Tk windows; nothing draws",
    "python/lib/python3.11/turtledemo": "Tk demos",
    "python/lib/python3.11/turtle.py": "Tk turtle graphics",
    "python/lib/python3.11/lib-dynload/_tkinter.cpython-311-darwin.so": "tkinter's library, linking Tcl and Tk",
    "python/lib/libtcl9.0.dylib": "Tcl, for tkinter only",
    "python/lib/libtcl9tk9.0.dylib": "Tk, for tkinter only",
    "python/lib/tcl9": "Tcl's modules",
    "python/lib/tcl9.0": "Tcl's library",
    "python/lib/tk9.0": "Tk's library",
    "python/lib/itcl4.3.8": "a Tcl extension, with its own libraries",
    "python/lib/thread3.0.6": "a Tcl extension, with its own libraries",
    "python/lib/python3.11/lib2to3": "converts Python 2 code; never run",
    "python/lib/python3.11/config-3.11-darwin": "the files for building C extensions against this Python",
    "python/lib/libpython3.11.dylib": "Python as a library for embedding; the executable carries its own and "
                                      "nothing links it",
    "python/include": "C headers, for building extensions",
    "python/lib/pkgconfig": "pkg-config files, for building against Python",
    "python/share": "manual pages",
    "python/bin/2to3": "a link to 2to3-3.11",
    "python/bin/2to3-3.11": "lib2to3's script",
    "python/bin/idle3": "a link to idle3.11",
    "python/bin/idle3.11": "IDLE's script",
    "python/bin/pydoc3": "a link to pydoc3.11",
    "python/bin/pydoc3.11": "pydoc's script",
    "python/bin/python": "a link to python3.11; the app starts python3.11 itself",
    "python/bin/python3": "a link to python3.11; the app starts python3.11 itself",
    "python/bin/python3-config": "a link to python3.11-config",
    "python/bin/python3.11-config": "prints build flags, for building extensions",
}
