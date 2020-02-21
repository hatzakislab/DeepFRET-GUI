from main import TraceWindow

filename = '../resources/traces/fiddler_3dim_0.txt'
_trace = TraceWindow.loadTraceFromAscii(None, filename)
print(_trace)