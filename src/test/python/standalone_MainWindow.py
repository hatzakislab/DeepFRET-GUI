import sys
from main import MainWindow, AppContext, gvars


class SetUp(MainWindow):
    def __init__(self):
        super(SetUp, self).__init__()

        self.setConfig(gvars.key_contrastBoxHiGrnVal, 100)
        self.setConfig(gvars.key_contrastBoxHiRedVal, 100)

    def setFile(self, path, imgmode):
        self.setConfig(gvars.key_imgMode, imgmode)
        self.data.load_img(path=path, setup=imgmode, name="")
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
        path="../resources/movies/Test_Quad_2c-mini.tif",
        imgmode=gvars.key_imgMode2Color,
    )

    cls.setupAlexDualFITS(
        path="../resources/movies/Antibody_RNAP_KG7_22degrees_667.fits",
        imgmode=gvars.key_imgMode2Color,
    )

    exit_code = ctxt.run()
    sys.exit(exit_code)
