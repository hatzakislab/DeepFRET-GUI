import sys
from main import MainWindow, AppContext, gvars


class SetUp(MainWindow):
    def __init__(self):
        super(SetUp, self).__init__()
        self.ui.spotsGrnSpinBox.setValue(100)
        self.ui.spotsRedSpinBox.setValue(100)

    def setFile(self, path, img_mode, donor_is_left, donor_is_first):
        self.img_mode = img_mode
        self.data.load_video_data(
            path=path,
            name="",
            setup=img_mode,
            donor_is_left=donor_is_left,
            donor_is_first=donor_is_first,
        )
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


if __name__ == "__main__":
    ctxt = AppContext()
    ctxt.load_resources()

    cls = SetUp()

    cls.setupAlexQuadTIFF(
        path="../resources/movies/Test_Quad_2c.tif",
        img_mode=gvars.key_imgMode2Color,
        donor_is_left=True,
        donor_is_first=True,
    )

    # cls.setupAlexDualFITS(
    #     path="../resources/movies/Antibody_RNAP_KG7_22degrees_667.fits",
    #     img_mode=gvars.key_imgMode2Color,
    #     donor_is_left = True,
    #     donor_is_first = False,
    # )

    exit_code = ctxt.run()
    sys.exit(exit_code)
