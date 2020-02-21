from unittest import TestCase

from numpy import ndarray

from src.main.python.main import TraceWindow

class TestTraceWindow(TestCase):
    def test_loadTraceFromAscii_DeepFRET(self):
        filename = '../resources/traces/fiddler_3dim_0.txt'
        _trace = TraceWindow.loadTraceFromAscii(filename)
        self.assertIsInstance(_trace.acc.int, ndarray)
        self.assertIsInstance(_trace.grn.int, ndarray)
        self.assertIsInstance(_trace.red.int, ndarray)
        self.assertEqual(_trace.get_bleaches(), (57, None, 57))


