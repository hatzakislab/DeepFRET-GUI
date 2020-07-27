import os
from unittest import TestCase
import numpy as np

from lib.container import TraceContainer


class TestTraceContainer(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.file_dir = "../resources/temp"
        cls.file_name = "temp.txt"
        cls.file_path = os.path.join(cls.file_dir, cls.file_name)

        os.mkdir(cls.file_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        for root, dirs, files in os.walk(cls.file_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        os.rmdir(cls.file_dir)

    def test_load_trace_from_ascii(self):
        filename = "../resources/traces/Trace.txt"
        trace = TraceContainer(filename, loaded_from_ascii=True)
        self.assertTrue(trace.load_successful)
        self.assertIsInstance(trace.acc.int, np.ndarray)
        self.assertIsInstance(trace.grn.int, np.ndarray)
        self.assertIsInstance(trace.red.int, np.ndarray)
        self.assertEqual(
            (28, 28, 28), trace.get_bleaches()
        )  # expect bleach to propagate all channels

    def test_load_trace_from_ascii_specified_bleach(self):
        filename = "../resources/traces/TraceNoAA.txt"
        trace = TraceContainer(filename, loaded_from_ascii=True)
        self.assertTrue(trace.load_successful)
        self.assertIsInstance(trace.acc.int, np.ndarray)
        self.assertIsInstance(trace.grn.int, np.ndarray)
        self.assertIsInstance(trace.red.int, np.ndarray)
        self.assertEqual(
            (57, 57, 57), trace.get_bleaches()
        )  # expect bleach to propagate all channels

    def test_load_trace_from_dat(self):
        filename = "../resources/traces/kinsoftTrace.dat"
        trace = TraceContainer(filename, loaded_from_ascii=True)

        self.assertTrue(trace.load_successful)
        self.assertIsInstance(trace.acc.int, np.ndarray)
        self.assertIsInstance(trace.grn.int, np.ndarray)
        self.assertIsInstance(trace.red.int, np.ndarray)
        self.assertTrue(np.isnan(trace.red.int[0]))

    def test_save_and_load_trace_dat(self):
        self.addCleanup(os.remove, self.file_path)
        filename = "../resources/traces/kinsoftTrace.dat"
        trace = TraceContainer(filename, loaded_from_ascii=True)
        trace.tracename = self.file_name
        trace.export_trace_to_txt(dir_to_join=self.file_dir)
        trace2 = TraceContainer(self.file_path, loaded_from_ascii=True)
        self.assertTrue(trace2.load_successful)
        self.assertEqual(type(trace.grn.int), type(trace2.grn.int))
        self.assertEqual(type(trace.acc.int), type(trace2.acc.int))
        self.assertEqual(type(trace.red.int), type(trace2.red.int))
        np.testing.assert_allclose(trace.grn.int, trace2.grn.int, rtol=1e-06)
        np.testing.assert_allclose(trace.acc.int, trace2.acc.int, rtol=1e-06)
        np.testing.assert_allclose(trace.red.int, trace2.red.int, rtol=1e-06)
        self.assertEqual(trace.first_bleach, trace2.first_bleach)
        self.assertEqual(trace.n, trace2.n)

    def test_reassignment_of_tracename(self):
        filename = "../resources/traces/Trace.txt"
        trace = TraceContainer(filename, loaded_from_ascii=True)
        trace.tracename = self.file_name
        self.assertEqual(trace.get_tracename(), self.file_name)
        self.assertEqual(
            trace.get_savename(dir_to_join=self.file_dir), self.file_path
        )

    def test_save_to_right_path(self):
        filename = "../resources/traces/Trace.txt"
        trace = TraceContainer(filename, loaded_from_ascii=True)
        trace.tracename = self.file_name
        trace.export_trace_to_txt(dir_to_join=self.file_dir)
        self.assertTrue(os.path.exists(self.file_path))

    def test_save_and_load_trace(self):
        self.addCleanup(os.remove, self.file_path)
        filename = "../resources/traces/Trace.txt"
        trace = TraceContainer(filename, loaded_from_ascii=True)
        trace.tracename = self.file_name
        trace.export_trace_to_txt(dir_to_join=self.file_dir)
        trace2 = TraceContainer(self.file_path, loaded_from_ascii=True)
        self.assertTrue(trace2.load_successful)
        np.testing.assert_array_almost_equal(trace.grn.int, trace2.grn.int)
        self.assertEqual(trace.first_bleach, trace2.first_bleach)

    def test_reducing_trace_save_and_load(self):
        self.addCleanup(os.remove, self.file_path)
        filename = "../resources/traces/Trace.txt"
        trace = TraceContainer(filename, loaded_from_ascii=True)
        trace.red.int[:] = None
        trace.red.bg[:] = None
        trace.stoi[:] = None
        trace.tracename = self.file_name
        trace.export_trace_to_txt(dir_to_join=self.file_dir)
        trace2 = TraceContainer(self.file_path, loaded_from_ascii=True)
        self.assertTrue(trace2.load_successful)
        self.assertEqual(trace.first_bleach, trace2.first_bleach)

        np.testing.assert_allclose(trace.grn.int, trace2.grn.int)
        np.testing.assert_allclose(trace.grn.bg, trace2.grn.bg)
        np.testing.assert_allclose(trace.acc.int, trace2.acc.int)
        np.testing.assert_allclose(trace.acc.bg, trace2.acc.bg)
        np.testing.assert_allclose(trace.red.int, trace2.red.int)
        np.testing.assert_allclose(trace.red.bg, trace2.red.bg)
        np.testing.assert_allclose(trace.stoi, trace2.stoi)

    def test_loading_nonALEX_iSMS_trace(self):
        filename = "../resources/traces/iSMSNoALEX.txt"
        trace = TraceContainer(filename, loaded_from_ascii=True)
        self.assertTrue(trace.load_successful)

    def test_reducing_trace_calculating_fret(self):
        self.addCleanup(os.remove, self.file_path)
        filename = "../resources/traces/Trace.txt"
        trace = TraceContainer(filename, loaded_from_ascii=True)
        trace.red.int[:] = None
        trace.red.bg[:] = None
        trace.stoi[:] = None
        df = trace.get_export_df()
        trace.fret[:] = None
        trace.tracename = self.file_name
        trace.export_trace_to_txt(
            dir_to_join=self.file_dir, keep_nan_columns=False
        )
        trace2 = TraceContainer(self.file_path, loaded_from_ascii=True)
        self.assertTrue(trace2.load_successful)
        df2 = trace2.get_export_df()

        np.testing.assert_array_almost_equal(df["E"], df2["E"])
