import sys
from main import AppContext, gvars
from widgets.video_window import VideoWindow


class SetUp(VideoWindow):
    def __init__(self):
        super(SetUp, self).__init__()
        self.ui.spotsGrnSpinBox.setValue(100)
        self.ui.spotsRedSpinBox.setValue(100)

    def setFile(self, path, **kwargs):
        self.data.load_video_data(path=path, name="", alex=True, **kwargs)
        self.currName = self.data.currName

    def testAlexQuadTIFF(self, **kwargs):
        """
        Loads and tests ALEX Quad view TIFF
        """
        self.setFile(**kwargs)
        self.refreshPlot()
        self.show()

    def testAlexDualFITS(self, **kwargs):
        """
        Loads and tests ALEX dual view FITS
        """
        self.setFile(**kwargs)
        self.refreshPlot()
        self.show()

    def testALEXDualInterleavedTIFF(self, **kwargs):
        """
        Loads and tests ALEX Dual cam view TIFF with interleaved video.
        The channel order is assumed to be

        Dexc-Aem -> Aexc-Aem -> Dexc-Dem -> Aexc-Dem (blank)

        Otherwise it might be impossible to auto-detect...
        """
        self.setFile(**kwargs)
        self.refreshPlot()
        self.show()


if __name__ == "__main__":
    ctxt = AppContext()
    ctxt.load_resources()

    cls = SetUp()

    # cls.testAlexQuadTIFF(
    #     path="../resources/movies/Test_Quad_2c.tif",
    #     donor_is_left=True,
    #     donor_is_first=True,
    # )
    #
    # cls.testAlexDualFITS(
    #     path="../resources/movies/Antibody_RNAP_KG7_22degrees_667.fits",
    #     donor_is_left = True,
    #     donor_is_first = False,
    # )

    cls.testALEXDualInterleavedTIFF(
        path="/Users/Joh/Desktop/077_078_Combined_20200304.tif"
    )

    exit_code = ctxt.run()
    sys.exit(exit_code)
