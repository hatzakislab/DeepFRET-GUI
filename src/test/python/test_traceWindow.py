import os
from unittest import TestCase
import numpy as np

from src.main.python.main import TraceWindow

class TestTraceWindow(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.file_path = 'temp.txt'


    def test_loadTraceFromAscii_DeepFRET(self):
        filename = '../resources/traces/fiddler_3dim_0.txt'
        _trace = TraceWindow.loadTraceFromAscii(filename)
        self.assertIsInstance(_trace.acc.int, np.ndarray)
        self.assertIsInstance(_trace.grn.int, np.ndarray)
        self.assertIsInstance(_trace.red.int, np.ndarray)
        self.assertEqual(_trace.get_bleaches(), (57, None, 57))

    def test_save_and_load_trace(self):
        self.addCleanup(os.remove, self.file_path)
        filename = '../resources/traces/fiddler_3dim_0.txt'
        _trace = TraceWindow.loadTraceFromAscii(filename)
        _trace.tracename = self.file_path
        _trace.export_trace_to_txt()
        _trace2 = TraceWindow.loadTraceFromAscii(self.file_path)

        self.assertTrue(np.allclose(_trace.grn.int, _trace2.grn.int))
        self.assertTrue(np.allclose(_trace.first_bleach, _trace2.first_bleach))


    def test_reducing_trace_save_and_load(self):
        self.addCleanup(os.remove, self.file_path)
        filename = '../resources/traces/fiddler_3dim_0.txt'
        _trace = TraceWindow.loadTraceFromAscii(filename)
        _trace.acc.int[:] = None
        _trace.acc.bg[:] = None
        _trace.stoi[:] = None
        _trace.tracename = self.file_path
        _trace.export_trace_to_txt()
        _trace2 = TraceWindow.loadTraceFromAscii(self.file_path)

        assert np.allclose(_trace.first_bleach, _trace2.first_bleach)
        assert np.allclose(_trace.grn.int, _trace2.grn.int)
        assert np.allclose(_trace.grn.bg, _trace2.grn.bg)
        assert np.allclose(_trace.red.int, _trace2.red.int)
        assert np.allclose(_trace.red.bg, _trace2.red.bg)
        np.testing.assert_array_almost_equal(_trace.acc.int, _trace2.acc.int)
        np.testing.assert_array_almost_equal(_trace.acc.bg, _trace2.acc.bg)
        np.testing.assert_array_almost_equal(_trace.stoi, _trace2.stoi)
