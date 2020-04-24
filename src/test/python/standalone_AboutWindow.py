import sys
from main import AppContext
from widgets.base_window import AboutWindow

if __name__ == "__main__":
    ctxt = AppContext()
    ctxt.load_resources()

    AboutWindow_ = AboutWindow()
    AboutWindow_.show()

    exit_code = ctxt.run()
    sys.exit(exit_code)
