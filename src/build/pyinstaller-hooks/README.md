# Custom PyInstaller Hooks
See [the PyInstaller
documentation](https://pyinstaller.readthedocs.io/en/stable/hooks.html) for
background information.

We use a quite old version of PyInstaller: 3.4. See the main README as to why.
This leave us with a fair deal of packages that do not work without new hooks.
Some hooks are already comitted to PyInstaller upstream, but is not in version
3.4, some are comitted, but not updated and some are not comitted.

## Verbatim copys from newer versions of PyInstaller

- `hook-astor.py`
- `hook-pywt.py`
- `hook-skimage.io.py`
- `hook-tensorflow_core.py`
- `hook-tensorflow.py`

## Hooks with changes from upstream

- `hook-astropy.py`
- `hook-sklearn.metrics.cluster.py`

## Custom written hooks

- `hook-networkx.py`
- `hook-pomegranate.py`
- `hook-sklearn.cluster.py`
- `hook-sklearn.neighbors.py`
- `hook-sklearn.py`
- `hook-sklearn.tree.py`
