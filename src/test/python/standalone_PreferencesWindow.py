import sys
from main import PreferencesWindow, AppContext

if __name__ == "__main__":
    ctxt = AppContext()
    ctxt.load_resources()

    PreferencesWindow_ = PreferencesWindow()
    PreferencesWindow_.show()

    exit_code = ctxt.run()
    sys.exit(exit_code)
