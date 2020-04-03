import sys
from main import MainWindow, AppContext, gvars

def setUp(movie_file_path, imgmode):
    """
    Set up the essentials for the window to launch
    """
    mock_movie_file = movie_file_path
    MainWindow_ = MainWindow()
    MainWindow_.data.load_img(mock_movie_file, name = mock_movie_file, setup = imgmode)
    MainWindow_.currName = MainWindow_.data.currName
    return MainWindow_

if __name__ == "__main__":
    ctxt = AppContext()
    ctxt.load_resources()

    # Set the config file imgmode
    ctxt.config[gvars.key_imgMode] = gvars.key_imgMode2Color

    # Initialize mainwindow with the right file and imgmode
    MainWindow_ = setUp(movie_file_path = "../resources/movies/Test_Quad_2c-mini.tif",
                        imgmode = gvars.key_imgMode2Color)

    MainWindow_.displaySpotsSingle("green")
    MainWindow_.displaySpotsSingle("red")
    MainWindow_.show()

    exit_code = ctxt.run()
    sys.exit(exit_code)