import sys
from main import AppContext
from widgets.base_window import PreferencesWindow

if __name__ == "__main__":
    ctxt = AppContext()
    ctxt.load_resources()

    PreferencesWindow_ = PreferencesWindow()
    PreferencesWindow_.show()

    exit_code = ctxt.run()
    sys.exit(exit_code)
