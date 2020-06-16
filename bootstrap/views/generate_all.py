from ..lib.options import Options
from ..run import main
from .factory import factory
from glob import glob
import os
from .plotly import PlotlyAll

def generate(path_opts=None):
    opt = Options(path_yaml=path_opts)
    exp_dir = opt['exp.dir']
    view = PlotlyAll(exp_dir, fname="view_all.html")
    view.generate()

if __name__ == '__main__':
    main(run=generate)
