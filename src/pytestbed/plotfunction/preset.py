import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray
from pykato.log import setup_logger
from pykato.plotfunction.gridspec_layout import GridSpec_Layout
from pykato.plotfunction.preset import Complex_ImageGrid_TwoColorbars_Preset, Complex_Imshow_TwoColorbars_Preset, Histogram_Colorbar_Preset, Imshow_Colorbar_Preset, monkeypatch_Axes_mask_image, monkeypatch_AxesImage_cmap_name, monkeypatch_AxesImage_cmap_norm
from tqdm import tqdm

logger = setup_logger("preset", terminator="\n")


def capture_data_plot_pdf(output: str, captures_list: list[np.ndarray], suptitles_list: list[str] | list[None] | None = None, capture_titles_list: list[str] | list[None] | None = None, capture_stats_list: list[str] | list[None] | None = None, capture_cmap_norm: Normalize | None = None, capture_cmap_name: str = "hot", capture_cmap_units="adu"):
    """Save a multi-page PDF of all frames from captures_list, one page per frame.

    Args:
        output: str
            File path for the output PDF.
        captures_list: list[np.ndarray]
            List of capture images.
        suptitles_list: list[str] | list[None] | None
            Per-frame suptitles, or None for no suptitles.
        capture_titles_list: list[str] | list[None] | None
            Per-frame capture axis titles, or None for default.
        capture_stats_list: list[str] | list[None] | None
            Per-frame stat annotations, or None.
        capture_cmap_norm: Normalize | None
            Colormap normalization, or None for default full 16-bit range [0, 65535].
    """
    if capture_cmap_norm is None:
        capture_cmap_norm = Normalize(0, 2**16 - 1)

    nframes = len(captures_list)
    nones = [None] * nframes

    if suptitles_list is None:
        suptitles_list = nones

    if capture_titles_list is None:
        capture_titles_list = nones

    if capture_stats_list is None:
        capture_stats_list = nones

    diagram = Image_Plot_Preset(np.zeros(captures_list[0].shape), capture_cmap_norm, capture_cmap_name)

    diagram.get_imshow_axes().set_title("Source", size=10)
    diagram.get_imshow_axes().set_xlabel("px", size=10)
    diagram.get_imshow_axes().set_ylabel("px", size=10)
    diagram.get_imshow_axes().axhline(captures_list[0].shape[0] / 2 - 0.5, alpha=0.25, linewidth=0.5, color="white")
    diagram.get_imshow_axes().axvline(captures_list[0].shape[1] / 2 - 0.5, alpha=0.25, linewidth=0.5, color="white")
    diagram.get_cbar_axes().set_title(capture_cmap_units, size=10)

    stat_textbox = diagram.get_imshow_axes().text(0.95, 0.95, "", fontfamily="monospace", horizontalalignment="right", verticalalignment="top", size=8, bbox=dict(boxstyle="round", facecolor="white", pad=0.25, linewidth=0.75), transform=diagram.get_imshow_axes().transAxes)
    stat_textbox.set_visible(False)

    vmin, vmax = float(captures_list[0].min()), float(captures_list[0].max())
    diagram.cbar_min_line.set_ydata([vmin, vmin])
    diagram.cbar_max_line.set_ydata([vmax, vmax])

    progressbar = tqdm(total=nframes, desc=f"Generating PDF : {output}")

    with PdfPages(output) as pdf:
        for capture, suptitle, capture_title, capture_stat in zip(captures_list, suptitles_list, capture_titles_list, capture_stats_list):
            diagram.get_image().set_data(capture)
            vmin, vmax = float(capture.min()), float(capture.max())
            diagram.cbar_min_line.set_ydata([vmin, vmin])
            diagram.cbar_max_line.set_ydata([vmax, vmax])
            if suptitle is not None:
                diagram.suptitle(suptitle, y=0.975)
            if capture_title is not None:
                diagram.get_imshow_axes().set_title(capture_title, size=10)
            if capture_stat is not None:
                stat_textbox.set_text(capture_stat)
                stat_textbox.set_visible(True)
            else:
                stat_textbox.set_visible(False)
            pdf.savefig(diagram)
            progressbar.update(1)

    progressbar.close()

    logger.info("Saved: %s", output)


def Wavefront_Plot_Preset(wavefront: NDArray[np.complex128], abs_min: float | None = None, abs_max: float | None = None, figure: Figure | None = None) -> Figure:
    """Plot preset used to illustrate wavefronts.

    Examples:
        figure = Wavefront_Plot_Preset(wavefront)
    """
    figure = Complex_Imshow_TwoColorbars_Preset(wavefront, abs_min, abs_max, figure)
    figure.get_imshow_axes().set_xlabel("px", size=10)
    figure.get_imshow_axes().set_ylabel("px", size=10)

    return figure


def Image_Plot_Preset(capture: NDArray[np.uint16 | np.float64], cmap_norm: Normalize | None = None, cmap_name: str | None = None, mask: NDArray[float] | None = None, figure: Figure | None = None) -> Figure:
    """Plot preset used to display source images.

    Parameters:
        capture: NDArray[np.float64]
            Capture image

        cmap_norm: bool | None = None
            Colormap normalization

        cmap_name: str | None = None
            Colormap Name

        mask: NDArray[np.bool] | None = None
            Mask

        figure: Figure | None = None
            Figure object

    Returns: figure: Figure | None = None
            Figure object
    """
    figure = Imshow_Colorbar_Preset(capture, figure=figure)

    if cmap_name is not None:
        monkeypatch_AxesImage_cmap_name(figure.get_image())
        figure.get_image().set_cmap_name(cmap_name)

    if cmap_norm is not None:
        monkeypatch_AxesImage_cmap_norm(figure.get_image())
        figure.get_image().set_cmap_norm(cmap_norm)

    if mask is not None:
        monkeypatch_Axes_mask_image(figure.get_imshow_axes(), mask.astype(float))

    figure.get_imshow_axes().set_xlabel("px", size=10)
    figure.get_imshow_axes().set_ylabel("px", size=10)

    figure.cbar_min_line = figure.get_cbar_axes().axhline(np.min(capture), color="cyan", linewidth=1)
    figure.cbar_max_line = figure.get_cbar_axes().axhline(np.max(capture), color="cyan", linewidth=1)

    return figure


# def Wavefront_Plot_Preset(wavefront: NDArray[np.complex128], abs_min: float | None = None, abs_max: float | None = None, figure: Figure | None = None) -> Figure:
#     """Plot preset used to illustrate wavefronts.

#     Examples:
#         figure = Wavefront_Plot_Preset(wavefront)
#     """
#     figure = Complex_Imshow_TwoColorbars_Preset(wavefront, abs_min, abs_max, figure)
#     figure.get_imshow_axes().set_xlabel("px", size=10)
#     figure.get_imshow_axes().set_ylabel("px", size=10)

#     return figure


def Histogram_Plot_Preset(capture: NDArray[np.float64], cmap_norm: Normalize | None = None, cmap_name: str | None = None) -> Figure:
    return Histogram_Colorbar_Preset(capture, nbins=256, cmap_norm=cmap_norm, cmap_name=cmap_name, position="bottom")


def Speckle_Modulation_Plot_Preset(phs_lim: tuple[float, float], amp_lim: tuple[float, float], figure: Figure | None = None) -> Figure:
    """Plot preset used to illustrate speckle nulling process. Consists of two plot axes for the phase and the intensity sweeps.

    Examples:
        figure = Speckle_Modulation_Plot_Preset(phs_lim, amp_lim)

    Parameters:
        phs_lim: tuple[float, float]
            phase plot limits min and max

        amp_lim: tuple[float, float]
            amplitude plot limits min and max

        figure: Figure | None = None
            Figure object

    Returns: figure: Figure | None = None
            Figure object

    Functions:
        set_phs_data_plot(phs_array: np.ndarray, phs_intensity_data_array: np.ndarray)
            set phase data plot

        set_phs_fit_plot(phs_intensity_fit_x_data: np.ndarray, phs_intensity_fit_y_data: np.ndarray)
            set phase fit plot

        set_amp_data_plot(amp_array: np.ndarray, amp_intensity_data_array: np.ndarray)
            set amplitude data plot

        set_amp_fit_plot(amp_intensity_fit_x_data: np.ndarray, amp_intensity_fit_y_data: np.ndarray)
            set amplitude fit plot

        set_phs_solve(solve: float)
            set phase solve

        set_amp_solve(solve: float)
            set amplitude solve

        close()
            Properly close the figure

    """
    figure = GridSpec_Layout(nrows=2, ncols=1, hspace=0.5, figure=figure)

    plot_ax_phs, plot_ax_amp = figure.get_axes()

    plot_ax_phs.set_title("Phase modulation", size=10)
    plot_ax_phs.set_xlabel("Phase", size=10)
    plot_ax_phs.set_ylabel("Intensity", size=10)
    plot_ax_phs.set_xlim((phs_lim[0], phs_lim[1]))
    plot_ax_phs.xaxis.set_major_locator(MaxNLocator(nbins=7))
    (phs_data_plot,) = plot_ax_phs.plot([], [], marker="+", linestyle="None")
    (phs_fit_plot,) = plot_ax_phs.plot([], [], color="red")
    phs_solve_line = plot_ax_phs.axvline(np.nan, color="red")

    plot_ax_amp.set_title("Amplitude modulation", size=10)
    plot_ax_amp.set_xlabel("Amplitude", size=10)
    plot_ax_amp.set_ylabel("Intensity", size=10)
    plot_ax_amp.set_xlim((amp_lim[0], amp_lim[-1]))
    (amp_data_plot,) = plot_ax_amp.plot([], [], marker="+", linestyle="None")
    (amp_fit_plot,) = plot_ax_amp.plot([], [], color="red")
    amp_solve_line = plot_ax_amp.axvline(np.nan, color="red")

    # -----------------------------------------------------------------------------------------------------------------
    def _set_phs_data_plot(phs_array: np.ndarray, phs_intensity_data_array: np.ndarray):
        phs_data_plot.set_xdata(phs_array)
        phs_data_plot.set_ydata(phs_intensity_data_array)

    figure.set_phs_data_plot = _set_phs_data_plot
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_phs_fit_plot(phs_intensity_fit_x_data: np.ndarray, phs_intensity_fit_y_data: np.ndarray):
        phs_fit_plot.set_xdata(phs_intensity_fit_x_data)
        phs_fit_plot.set_ydata(phs_intensity_fit_y_data)

    figure.set_phs_fit_plot = _set_phs_fit_plot
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_amp_data_plot(amp_array: np.ndarray, amp_intensity_data_array: np.ndarray):
        amp_data_plot.set_xdata(amp_array)
        amp_data_plot.set_ydata(amp_intensity_data_array)

    figure.set_amp_data_plot = _set_amp_data_plot
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_amp_fit_plot(amp_intensity_fit_x_data: np.ndarray, amp_intensity_fit_y_data: np.ndarray):
        amp_fit_plot.set_xdata(amp_intensity_fit_x_data)
        amp_fit_plot.set_ydata(amp_intensity_fit_y_data)

    figure.set_amp_fit_plot = _set_amp_fit_plot
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_phs_solve(solve: float):
        phs_solve_line.set_xdata([solve, solve])

    figure.set_phs_solve = _set_phs_solve
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _set_amp_solve(solve: float):
        amp_solve_line.set_xdata([solve, solve])

    figure.set_amp_solve = _set_amp_solve
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _phs_ax() -> Axes:
        return plot_ax_phs

    figure.phs_ax = _phs_ax
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _amp_ax() -> Axes:
        return plot_ax_amp

    figure.amp_ax = _amp_ax
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    return figure


def Contrast_Evolution_Plot_Preset(contrast: np.ndarray, n_iteration: int, dark_hole_mask: NDArray[np.float64], figure: Figure | None = None) -> Figure:
    """Plot preset used to illustrate contrast evolution. Consists of and Imshow axis and corresponding colorbar for contrast and a plot for change in contrast with iteration.

    Examples:
        figure = Contrast_Evolution_Plot_Preset(contrast, n_iteration, dark_hole_mask)

    Parameters:
        contrast: NDArray[np.float64]
            Contrast image

        n_iteration: int
            Loop iterations

        dark_hole_mask: NDArray[np.bool] = None
            dark hole mask

        figure: Figure | None = None
            Figure object.

    Returns: figure: Figure | None = None
            Figure object.

    Functions:

        close()
            Properly close the figure

    """
    figure = GridSpec_Layout(nrows=1, ncols=1, aspect_ratios=(8,), figure=figure)

    (image_ax,) = figure.get_axes()

    image_ax.set_title("Speckles", size=10)
    image_ax.set_xlabel("px", size=10)
    image_ax.set_ylabel("px", size=10)
    image_ax.axhline(contrast.shape[0] / 2 - 0.5, alpha=0.25, linewidth=0.5, color="white")
    image_ax.axvline(contrast.shape[1] / 2 - 0.5, alpha=0.25, linewidth=0.5, color="white")
    # speckle = (image_ax.axvline(np.nan, alpha=0.5, linewidth=0.5, color="red"), image_ax.axhline(np.nan, alpha=0.5, linewidth=0.5, color="red"))
    (speckle,) = image_ax.plot([], [], color="red", marker="o", markersize=10, markerfacecolor="none", linestyle="none")
    image_ax.add_patch(patches.Circle((contrast.shape[0] / 2 - 0.5, contrast.shape[1] / 2 - 0.5), radius=contrast.shape[1] / 2, fill=False, alpha=0.25, linewidth=0.5, color="white", transform=image_ax.transData))
    imshow_image = image_ax.imshow(contrast)
    image_ax.set_facecolor("black")
    image_ax.invert_yaxis()

    divider = make_axes_locatable(image_ax)

    colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(imshow_image, cax=colorbar_ax)
    colorbar_ax.set_title("Contrast", size=10)

    plot_ax = divider.append_axes("right", size="200%", pad=0.5)
    plot_ax.set_title("Evolution", size=10)
    plot_ax.set_xlabel("Iteration", size=10)
    plot_ax.set_xlim((0, n_iteration))
    plot_ax.set_yscale("log")
    plot_ax.set_ylim(1, 2**16 - 1)
    plot_ax.set_ylabel("", size=10)
    plot_ax.set_yticklabels([])
    (data_plot,) = plot_ax.plot([], [], marker="+", linestyle="None")

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image() -> AxesImage:
        return imshow_image

    figure.get_image = _get_image
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image_axes() -> Axes:
        return image_ax

    figure.get_image_axes = _get_image_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_plot() -> Line2D:
        return data_plot

    figure.get_plot = _get_plot
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_plot_axes() -> Axes:
        return plot_ax

    figure.get_plot_axes = _get_plot_axes
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _get_speckle() -> Line2D:
        return speckle

    figure.get_speckle = _get_speckle
    # -----------------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------------
    def _close():
        plt.close(figure)

    figure.close = _close
    # -----------------------------------------------------------------------------------------------------------------

    monkeypatch_AxesImage_cmap_name(imshow_image)
    imshow_image.set_cmap_name("jet")

    monkeypatch_AxesImage_cmap_norm(imshow_image)
    imshow_image.set_cmap_norm(LogNorm(1, 1e-5))

    monkeypatch_Axes_mask_image(image_ax, dark_hole_mask)
    image_ax.get_mask().set_cmap("gray")

    return figure


def DOTF_Measurement_Plot_Preset(measure_dict: dict[int, NDArray[np.complex64]], figure: Figure | None = None) -> Figure:
    """Plot preset used to illustrate DOTF wavefront measurement process. Consists of complex image plot axes to display DOTF maps.

    Examples:
        figure = DOTF_Sensing_Process_Plot_Preset(measure_dict, command)
    """
    figure = Complex_ImageGrid_TwoColorbars_Preset(measure_dict.values(), figure=figure)

    imshow_image_dict = {}
    for index, (imshow_axes, image, key) in enumerate(zip(figure.get_imshow_axes_list(), figure.get_image_list(), measure_dict.keys())):
        imshow_axes.set_title(f"{key:02d}", size=10)
        imshow_axes.set_xlabel("px", size=10)
        if index == 0:
            imshow_axes.set_ylabel("px", size=10)
        imshow_image_dict[key] = image

    # -----------------------------------------------------------------------------------------------------------------
    def _get_image_dict() -> dict[int, AxesImage]:
        return imshow_image_dict

    figure.get_image_dict = _get_image_dict
    # -----------------------------------------------------------------------------------------------------------------

    return figure
