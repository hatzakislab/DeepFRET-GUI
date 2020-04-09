import os
from unittest import TestCase
import numpy as np

from lib.container import TraceContainer


class TestTraceContainer(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.file_path = "temp.txt"

    def test_load_trace_from_ascii(self):
        filename = "../resources/traces/fiddler_3dim_0.txt"
        trace = TraceContainer(filename)
        self.assertTrue(trace.load_successful)
        self.assertIsInstance(trace.acc.int, np.ndarray)
        self.assertIsInstance(trace.grn.int, np.ndarray)
        self.assertIsInstance(trace.red.int, np.ndarray)
        self.assertEqual(
            (57, 57, 57), trace.get_bleaches()
        )  # expect bleach to propagate all channels

    def test_load_trace_from_ascii_specified_bleach(self):
        filename = "../resources/traces/TraceNoAA.txt"
        trace = TraceContainer(filename)
        self.assertTrue(trace.load_successful)
        self.assertIsInstance(trace.acc.int, np.ndarray)
        self.assertIsInstance(trace.grn.int, np.ndarray)
        self.assertIsInstance(trace.red.int, np.ndarray)
        self.assertEqual(
            (57, None, 57), trace.get_bleaches()
        )  # expect bleach to propagate all channels

    def test_load_trace_from_dat(self):
        filename = "../resources/traces/kinsoftTrace.dat"
        trace = TraceContainer(filename)
        self.assertTrue(trace.load_successful)
        self.assertIsInstance(trace.acc.int, np.ndarray)
        self.assertIsInstance(trace.grn.int, np.ndarray)
        self.assertIsInstance(trace.red.int, np.ndarray)
        self.assertTrue(np.isnan(trace.red.int[0]))

    def test_save_and_load_trace_dat(self):
        self.addCleanup(os.remove, self.file_path)
        filename = "../resources/traces/kinsoftTrace.dat"
        trace = TraceContainer(filename)
        trace.tracename = self.file_path
        trace.export_trace_to_txt()
        trace2 = TraceContainer(self.file_path)
        np.testing.assert_allclose(trace.grn.int, trace2.grn.int, rtol=1e-06)
        np.testing.assert_allclose(trace.acc.int, trace2.acc.int, rtol=1e-06)
        np.testing.assert_allclose(trace.red.int, trace2.red.int, rtol=1e-06)
        self.assertEqual(trace.first_bleach, trace2.first_bleach)
        self.assertEqual(trace.n, trace2.n)

    def test_save_and_load_trace(self):
        self.addCleanup(os.remove, self.file_path)
        filename = "../resources/traces/fiddler_3dim_0.txt"
        trace = TraceContainer(filename)
        trace.tracename = self.file_path
        trace.export_trace_to_txt()
        trace2 = TraceContainer(self.file_path)
        np.testing.assert_array_almost_equal(trace.grn.int, trace2.grn.int)
        self.assertEqual(trace.first_bleach, trace2.first_bleach)

    def test_reducing_trace_save_and_load(self):
        self.addCleanup(os.remove, self.file_path)
        filename = "../resources/traces/fiddler_3dim_0.txt"
        trace = TraceContainer(filename)
        trace.red.int[:] = None
        trace.red.bg[:] = None
        trace.stoi[:] = None
        trace.tracename = self.file_path
        trace.export_trace_to_txt()
        trace2 = TraceContainer(self.file_path)

        self.assertEqual(trace.first_bleach, trace2.first_bleach)

        np.testing.assert_allclose(trace.grn.int, trace2.grn.int)
        np.testing.assert_allclose(trace.grn.bg, trace2.grn.bg)
        np.testing.assert_allclose(trace.acc.int, trace2.acc.int)
        np.testing.assert_allclose(trace.acc.bg, trace2.acc.bg)
        np.testing.assert_allclose(trace.red.int, trace2.red.int)
        np.testing.assert_allclose(trace.red.bg, trace2.red.bg)
        np.testing.assert_allclose(trace.stoi, trace2.stoi)

    def test_reducing_trace_calculating_fret(self):
        self.addCleanup(os.remove, self.file_path)
        filename = "../resources/traces/fiddler_3dim_0.txt"
        trace = TraceContainer(filename)
        trace.red.int[:] = None
        trace.red.bg[:] = None
        trace.stoi[:] = None
        df = trace.get_export_df()
        trace.fret[:] = None
        trace.savename = self.file_path
        trace.export_trace_to_txt(keep_nan_columns=False)
        trace2 = TraceContainer(self.file_path)
        df2 = trace2.get_export_df()

        np.testing.assert_array_almost_equal(df["E"], df2["E"])
