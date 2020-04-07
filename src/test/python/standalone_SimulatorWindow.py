import sys
from main import SimulatorWindow, AppContext

if __name__ == "__main__":
    ctxt = AppContext()
    ctxt.load_resources()

    SimulatorWindow_ = SimulatorWindow()
    SimulatorWindow_.show()

    exit_code = ctxt.run()
    sys.exit(exit_code)
