
# Plotting specifics.
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import scipy.stats as stats

import cphxai.src.utils.utilities_plot as up
import cphxai.src.utils.utilities_calc as uc


# Source code: https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html.

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default."""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default."""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()

            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, angles=None):
            labels_with_newlines = [l.replace(' ', '\n') for l in labels]
            _lines, texts = self.set_thetagrids(np.degrees(theta), labels_with_newlines)
            half = (len(texts) - 1) // 2
            for t in texts[1:half]:
                t.set_horizontalalignment('right')
            for t in texts[-half + 1:]:
                t.set_horizontalalignment('left')

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped."""
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)

                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def beginFinalPlot(YpredTrain, YpredTest, Ytrain, Ytest, testIndices, years, yearsObs, YpredObs, trainIndices, yearsall, sis, accuracy, obsyearstart,
                   variq, directoryfigure, savename, modelType):

    """
    Plot prediction of year
    """

    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})

    fig = plt.figure()
    ax = plt.subplot(111)

    up.adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both', length=4, width=2, which='major', color='black')

    train_output_rs = YpredTrain.reshape(len(trainIndices),
                                         len(years))
    test_output_rs = YpredTest.reshape(len(testIndices),
                                       len(years))

    xs_test = (np.arange(np.shape(test_output_rs)[1]) + yearsall[sis].min())


    for i in range(0, train_output_rs.shape[0]):
        if i == train_output_rs.shape[0] - 1:
            p3 = plt.plot(xs_test, train_output_rs[i, :], 'o',
                          markersize=4, color='lightgray', clip_on=False,
                          alpha=0.4, markeredgecolor='k', markeredgewidth=0.4,
                          label=r'\textbf{Training Data}')
        else:
            p3 = plt.plot(xs_test, train_output_rs[i, :], 'o',
                          markersize=4, color='lightgray', clip_on=False,
                          alpha=0.4, markeredgecolor='k', markeredgewidth=0.4)
    for i in range(0, test_output_rs.shape[0]):
        if i == test_output_rs.shape[0] - 1:
            p4 = plt.plot(xs_test, test_output_rs[i, :], 'o',
                          markersize=4, color='dimgrey', clip_on=False, alpha=0.3,
                          markeredgecolor='dimgrey', markeredgewidth=0.4,
                          label=r'\textbf{Testing Data}' )
        else:
            p4 = plt.plot(xs_test, test_output_rs[i, :], 'o',
                          markersize=4, color='dimgrey', clip_on=False, alpha=0.3,
                          markeredgecolor='dimgrey', markeredgewidth=0.4)


    iy = np.where(yearsObs >= obsyearstart)[0]
    linereg = stats.linregress(yearsObs[iy], YpredObs[iy])
    ylineobs = linereg.slope * yearsObs[iy] + linereg.intercept
    plt.plot(yearsObs[iy], YpredObs[iy], 'x', color='deepskyblue',
             label=r'\textbf{Reanalysis}', clip_on=False)
    plt.plot(yearsObs[iy], ylineobs, '-',
             color='blue', linewidth=2, clip_on=False)

    plt.xlabel(r'\textbf{YEAR}', fontsize=10, color='dimgrey')
    plt.ylabel(r'\textbf{PREDICTED YEAR}', fontsize=10, color='dimgrey')
    plt.plot(np.arange(yearsall[sis].min(), yearsall[sis].max() + 1, 1),
             np.arange(yearsall[sis].min(), yearsall[sis].max() + 1, 1), '-',
             color='black', linewidth=2, clip_on=False)

    plt.xticks(np.arange(yearsall[sis].min(), 2101, 20), map(str, np.arange(yearsall[sis].min(), 2101, 20)),
               size=8)
    plt.yticks(np.arange(yearsall[sis].min(), 2101, 20), map(str, np.arange(yearsall[sis].min(), 2101, 20)),
               size=8)
    plt.xlim([yearsall[sis].min(), yearsall[sis].max()])
    plt.ylim([yearsall[sis].min(), yearsall[sis].max()])

    plt.text(yearsall[sis].max()-50, yearsall[sis].min(), r'{%s \textbf{RMSE} Test = %s}' % (modelType, np.round(uc.rmse(YpredTest[:, ],
                                                                  Ytest[:, 0]),
                                                         decimals=1)),
              color='k',
              fontsize=11)


    leg = plt.legend(shadow=False, fontsize=11, loc='upper left',
                     bbox_to_anchor=(-0.01, 1), fancybox=True, ncol=1, frameon=False,
                     handlelength=1, handletextpad=0.5)
    savefigName = modelType + '_' + variq + '_scatterPred_' + savename

    plt.savefig(
        directoryfigure + savefigName + '.png' ,
        dpi=600)
    print(np.round(np.corrcoef(yearsObs, YpredObs)[0, 1], 2))
    return