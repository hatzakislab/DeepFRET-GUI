from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('pomegranate')
hiddenimports += collect_submodules('pomegranate.utils')
hiddenimports += collect_submodules('pomegranate.distributions')
hiddenimports += collect_submodules('networkx')
