import sys
from main import MainWindow, AppContext, gvars


class SetUp(MainWindow):
    def __init__(self):
        super(SetUp, self).__init__()
        self.ui.spotsGrnSpinBox.setValue(100)
        self.ui.spotsRedSpinBox.setValue(100)

    def setFile(self, path, **kwargs):
        self.data.load_video_data(path=path, name="", **kwargs)
        self.currName = self.data.currName

    def setupAlexQuadTIFF(self, **kwargs):
        """
        Loads and tests ALEX Quad view TIFF
        """
        self.setFile(**kwargs)
        self.refreshPlot()
        self.show()

    def setupAlexDualFITS(self, **kwargs):
        """
        Loads and tests ALEX dual view FITS
        """
        self.setFile(**kwargs)
        self.refreshPlot()
        self.show()

    def setupALEXDualInterleavedTIFF(self, **kwargs):
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

    # cls.setupAlexQuadTIFF(
    #     path="../resources/movies/Test_Quad_2c.tif",
    #     donor_is_left=True,
    #     donor_is_first=True,
    # )
    #
    # cls.setupAlexDualFITS(
    #     path="../resources/movies/Antibody_RNAP_KG7_22degrees_667.fits",
    #     donor_is_left = True,
    #     donor_is_first = False,
    # )

    cls.setupALEXDualInterleavedTIFF(
        path="/Users/Joh/Desktop/077_078_Combined_20200304.tif"
    )

    exit_code = ctxt.run()
    sys.exit(exit_code)
