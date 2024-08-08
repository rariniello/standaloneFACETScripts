import os
import math
import json

import numpy as np
from scipy import integrate, interpolate, constants
import matplotlib.pyplot as plt

import plot
import image
import dataset
import analysis as an

c = constants.physical_constants["speed of light in vacuum"][0]
e = constants.physical_constants["elementary charge"][0]
me = constants.physical_constants["electron mass"][0]
eps_0 = constants.physical_constants["vacuum electric permittivity"][0]


def forward(
    y: float | np.ndarray, d_nom: float, E_bend: float, dy: float
) -> float | np.ndarray:
    """Returns the energy corresponding to the given screen position(s).

    Args:
        y: The y-coordinate, measured from the bottom of the screen [mm].
        d_nom: The nominal dispersion for 10GeV at the spectrometer screen [mm].
        E_bend: The energy setting of the dipole (energy difflected by d_nom) [GeV].
        dy: The distance from y=0 to the bottom of the screen [mm].
    """
    return d_nom * E_bend / (dy - y)


def inverse(
    E: float | np.ndarray, d_nom: float, E_bend: float, dy: float
) -> float | np.ndarray:
    """Returns the screen position corresponding to the energy(s).

    Args:
        E: The energy [GeV].
        d_nom: The nominal dispersion for 10GeV at the spectrometer screen [mm].
        E_bend: The energy setting of the dipole (energy difflected by d_nom) [GeV].
        dy: The distance from y=0 to the bottom of the screen [mm].
    """
    return dy - d_nom * E_bend / E


def loadCalibration(path: str | os.PathLike):
    """Returns a dictionary with calibration data loaded from a JSON file.

    Args:
        path: The path to the JSON file.
    """
    with open(path) as f:
        d = json.load(f)
    return d


def calcCharge(E: np.ndarray, spectrum: np.ndarray) -> float:
    """Returns the total charge in the spectrum [nC].

    Args:
        E: The energy array [GeV].
        spectrum: The spectrum array [nC/GeV].
    """
    Q = integrate.trapezoid(np.flip(spectrum), x=np.flip(E))
    return Q


def calcChargeError(
    E: np.ndarray,
    spectrum: np.ndarray,
    chargeCal: float | np.ndarray,
    chargeCalErr: float | np.ndarray,
):
    """Calculates the error in the charge measurement.

    Args:
        E: The energy array [GeV].
        spectrum: The spectrum array [nC/GeV].
        charge_cal: The charge calibration as a float or an array for each energy [nC/count].
        charge_cal_err: The error in the charge calibration factor [nC/count].

    Returns:
        The calculated charge error [nC].
    """
    QErr = integrate.trapezoid(
        np.flip(spectrum * chargeCalErr / chargeCal), x=np.flip(E)
    )
    return QErr


def calcMinMaxEnergy(
    E: np.ndarray, spectrum: np.ndarray, threshold: float, constant: bool = False
) -> tuple[float, float]:
    """Calculates the minimum and maximum energy from the spectrum.

    The minimum energy and maximum energy are the points where the spectrum intercepts
    a threshold. By default, the threshold is relative to the maximum value, i.e., 0.03.
    If constant is true, threshold is treated as a constant value in nC/GeV.

    Args:
        E: The energy array [GeV].
        spectrum: The spectrum array [nC/GeV].
        threshold: The threshold value used to find the minimum and maximum energy.
        constant: If True, the threshold is not relative to the spectrum max and
            instead a constant value in nC/GeV.

    Returns:
        A tuple(minEnergy, maxEnergy) containing the minimum and maximum energy.
    """
    projMax = np.max(spectrum)
    minThres = threshold * projMax
    if constant:
        minThres = threshold
    limit = 2.5e-3
    if minThres < limit:
        minThres = limit
    t = spectrum - minThres
    try:
        roots = an.find_roots(E, t)
    except:
        minEnergy = 0.0
        maxEnergy = 0.0
        if minThres < spectrum[-1]:
            minEnergy = E[-1]
        if minThres < spectrum[0]:
            maxEnergy = E[0]
        return minEnergy, maxEnergy
    minEnergy = roots[-1]
    maxEnergy = roots[0]
    if minThres < spectrum[-1]:
        minEnergy = E[-1]
    if minThres < spectrum[0]:
        maxEnergy = E[0]
    return minEnergy, maxEnergy


def calcVisibleEnergy(E: np.ndarray, spectrum: np.ndarray) -> float:
    """Returns the total amount of energy visible in the spectrum [J].

    Args:
        E: The energy array [GeV].
        spectrum: The spectrum array [nC/GeV].
    """
    E_vis = integrate.trapezoid(np.flip(spectrum * E), x=np.flip(E))
    return E_vis


def calcVisibleEnergyError(
    E: np.ndarray,
    spectrum: np.ndarray,
    chargeCal: float | np.ndarray,
    chargeCalErr: float | np.ndarray,
    y: np.ndarray,
    d_nom: float,
    d_nom_err: float,
    dy: float,
    dy_err: float,
) -> float:
    """Calculates the error in the visible energy calculation.

    Args:
        E: The energy array [GeV].
        spectrum: The spectrum array [nC/GeV].
        charge_cal: The charge calibration as a float or an array for each energy [nC/count].
        charge_cal_err: The error in the charge calibration factor [nC/count].
        y: The y-coordinate on the spectrometer screen of each point in the spectrum.
        d_nom: The nominal dispersion for 10GeV at the spectrometer screen [mm].
        d_nom_err: The error in d_nom [mm].
        dy: The distance from y=0 to the bottom of the screen [mm].
        dy_err: The error in dy [mm].

    Returns:
        The calculated visible energy error [J].
    """
    E_vis = calcVisibleEnergy(E, spectrum)
    # Error due to charge calibration error (not charge error)
    err_Q = integrate.trapezoid(
        np.flip(spectrum * E * chargeCalErr / chargeCal), x=np.flip(E)
    )
    # Error due to d_nom error
    err_dnom = E_vis * d_nom_err / d_nom
    # Error due to d_y error
    err_dy = (
        integrate.trapezoid(np.flip(spectrum * E / (dy - y)), x=np.flip(E)) * dy_err
    )
    E_vis_err = np.sqrt(err_Q**2 + err_dnom**2 + err_dy**2)
    return E_vis_err


class SPEC:
    """A spectrometer screen image, taken by either CHER, LFOV, DTOTR1/2, or EDC_SCREEN.

    This class handles charge calibration, energy calibration, and extraction of the
    spectrum from a raw image taken by the DAQ. The spectrum and total charge are calculated
    when the object is initialized and any of the calibrations are changed. If the image is
    manually modified, the spectrum can be regenerated by calling generateSpectrum.

    Attributes:
        image: The image object for the DAQ image.
        d_nom_err: The error in d_nom [mm].
        dy_err: The error in dy [mm].
        chargeCalErr: The error in the charge calibration factor [nC/count].
        E: The energy array [GeV].
        dE: The spacing between elements in the energy array [GeV]
        spectrum: The spectrum array [nC/GeV].
        Q: Total charge in the spectrum [nC].
    """

    def __init__(
        self,
        image: image.Image,
        E_bend: float,
        d_nom: float,
        d_nom_err: float,
        dy: float,
        dy_err: float,
        chargeCal: float | np.ndarray,
        chargeCalErr: float | np.ndarray,
    ):
        self.image = image
        self._d_nom = d_nom
        self.d_nom_err = d_nom_err
        self._E_bend = E_bend
        self._dy = dy
        self.dy_err = dy_err
        self._chargeCal = chargeCal
        self.chargeCalErr = chargeCalErr
        self.E = None
        self.dE = None
        self.spectrum = None
        self.Q = None
        self.generateSpectrum()

    @property
    def d_nom(self) -> float:
        """The nominal dispersion for 10GeV at the spectrometer screen [mm]."""
        return self._d_nom

    @d_nom.setter
    def d_nom(self, value: float):
        self._d_nom = value
        self.generateSpectrum()

    @property
    def dy(self) -> float:
        """The distance from y=0 to the bottom of the screen [mm]."""
        return self._dy

    @dy.setter
    def dy(self, value: float):
        self._dy = value
        self.generateSpectrum()

    @property
    def E_bend(self) -> float:
        """The energy setting of the dipole (energy difflected by d_nom) [GeV]."""
        return self._E_bend

    @E_bend.setter
    def E_bend(self, value: float):
        self._E_bend = value
        self.generateSpectrum()

    @property
    def chargeCal(self) -> float | np.ndarray:
        """The charge calibration as a float or an array for each energy [nC/count]."""
        return self._chargeCal

    @chargeCal.setter
    def chargeCal(self, value: float | np.ndarray):
        self._chargeCal = value
        self.generateSpectrum()

    def forward(self, y):
        """Returns the energy corresponding to the given screen position(s).

        Args:
            y: The y-coordinate, measured from the bottom of the screen [mm].
        """
        return forward(y, self.d_nom, self.E_bend, self.dy)

    def inverse(self, E):
        """Returns the screen position corresponding to the energy(s).

        Args:
            E: The energy [GeV].
        """
        return inverse(E, self.d_nom, self.E_bend, self.dy)

    def getSpectrum(self):
        """Returns a tuple containing the energy, spectrum, and spacing between energies."""
        if self.spectrum is not None:
            return self.E, self.spectrum, self.dE
        else:
            self.generateSpectrum()
            return self.E, self.spectrum, self.dE

    def generateSpectrum(self):
        """Recalculates the spectrum, useful if the image has been modified."""
        y = np.flip(self.image.y)
        self.E = self.forward(y)
        N = np.sum(self.image.data, axis=1)
        E_bin_s = self.forward(y - 0.5 * self.image.cal)
        E_bin_e = self.forward(y + 0.5 * self.image.cal)
        self.dE = E_bin_e - E_bin_s
        dNdE = N / self.dE
        dQdE = dNdE * self.chargeCal
        self.spectrum = dQdE
        self.Q = calcCharge(self.E, self.spectrum)

    def plot_image(self, cmap=plot.cmap_W_Viridis, metadata: bool = True):
        """Plots the spectrometer image with energy on the y-axis.

        Args:
            cmap: The colormap to use.
            metadata: Whether to include metadata in the plot.

        Returns:
            A tuple containing the following elements:
                - fig: Matplotlib figure object.
                - ax: Matplotlib axes object for plotting the image.
                - im: Matplotlib imshow object for the image.
                - cb: Matplotlib colorbar object.
                - ext: Array of shape (4,) representing the extent of the image.
        """
        fig, ax, im, cb, ext = self.image.plot_image(
            cal=True, cmap=cmap, metadata=metadata
        )
        # TODO Calculate width based on figure width
        fig.set_size_inches(2.85, 6.1)
        E_ticks = np.arange(
            math.ceil(self.forward(ext[3])), math.floor(self.forward(ext[2])) + 1
        )
        tick_locs = [ext[2] - self.inverse(E) for E in E_ticks]
        tick_lbls = ["{:0.1f}".format(E) for E in E_ticks]
        ax.set_ylabel("Energy (GeV)")
        plt.yticks(tick_locs, tick_lbls)
        ax.tick_params(color="k")
        ax.spines["bottom"].set_color("k")
        ax.spines["top"].set_color("k")
        ax.spines["left"].set_color("k")
        ax.spines["right"].set_color("k")
        if metadata:
            ax.stepText.set_color("k")
            ax.datasetText.set_color("k")
            ax.metadataText.set_color("k")
            self.append_metadata_text(ax.metadataText)
        return fig, ax, im, cb, ext

    def plot_spectrum(self, metadata: bool = True):
        """Plots the spectrum calculated from the spectrometer image.

        Args:
            metadata: Whether to include metadata in the plot.

        Returns:
            A tuple(fig, ax) containing the Matplotlib figure (fig) and axis (ax).
        """
        self.getSpectrum()
        fig = plt.figure(figsize=(2.85, 2), dpi=300)
        ax = fig.add_subplot()
        ax.plot(self.E, self.spectrum)
        ax.set_ylabel(r"$dQ/dE$ (nC/GeV)")
        ax.set_xlabel(r"Energy (GeV)")
        ax.set_xlim(self.E[-1], self.E[0])
        ax.set_ylim(0, 1.2 * np.max(self.spectrum))
        if metadata:
            self.image.plot_dataset_text(ax)
            # self.image.plot_metadata_text(ax)
            ax.stepText.set_color("k")
            ax.datasetText.set_color("k")
            # ax.metadataText.set_color("k")
        return fig, ax

    def append_metadata_text(self, metadataText):
        """Appends calibration information to the metadata text printed in the image.

        Args:
            metadataText: The metadata text object.
        """
        text = metadataText.get_text()
        addText = (
            r"$dy$:      "
            + "{:0.2f}\n".format(self.dy)
            + r"$d_{nom}$:   "
            + "{:0.2f}\n".format(self.d_nom)
            + r"$E_{bend}$:  "
            + "{:0.1f}GeV\n".format(self.E_bend)
        )
        text = addText + text
        metadataText.set_text(text)


class SPEC_DS:
    """A dataset of spectrometer images taken by a single spectrometer camera.

    This class provides convenient methods to extract individual spectrum images
    and perform common image processing on as well as for producing waterfalls.

    Attributes:
        cam: The camera used for the dataset.
        dataset: The dataset containing the images.
        d_nom: The nominal dispersion for 10GeV at the spectrometer screen [mm].
        d_nom_err: The error in d_nom [mm].
        dy: The distance from y=0 to the bottom of the screen [mm].
        dy_err: The error in dy [mm].
        chargeCal: The charge calibration as a float or an array for each energy [nC/count].
        chargeCalErr: The error in the charge calibration factor [nC/count].
        E_bend_default: The default dipole setting if not found in the dataset [GeV].
        crop (tuple | np.ndarray): The crop rectangle, as a (left, upper, right, lower) tuple.
        subtract_background: Flag to subtract background from images.
        median_filter (int): Size of the median filter to apply to the images.
        N: The number of images in the dataset.
        img: The initial image from the dataset, useful for getting image size.
        initial_height: The initial height of the image before cropping.
        wf: The waterfall spectrum, None until get_waterfall is called.
        wf_linear: The linearized waterfall spectrum, None until get_linear_waterfall is called.
        E_linear: Energy for the linearized waterfall, None until get_linear_waterfall is called.
    """

    def __init__(
        self,
        dataset: dataset.DATASET,
        cam: str,
        d_nom: float,
        d_nom_err: float,
        dy: float,
        dy_err: float,
        chargeCal: float | np.ndarray,
        chargeCalErr: float | np.ndarray,
        E_bend_default: float = 10.0,
        crop=None,
        subtract_background: bool = True,
        median_filter=None,
    ):
        self.cam = cam
        self.dataset = dataset
        self.d_nom = d_nom
        self.d_nom_err = d_nom_err
        self.dy = dy
        self.dy_err = dy_err
        self.chargeCal = chargeCal
        self.chargeCalErr = chargeCalErr
        self.E_bend_default = E_bend_default
        self.crop = crop
        self.N = len(dataset.common_index)
        self.img = dataset.getImage(cam, 0)
        self.initial_height = self.img.height
        if self.crop is not None:
            self.img.crop(self.crop)
        self.wf = None
        self.wf_linear = None
        self.E_linear = None
        self.subtract_background = subtract_background
        self.median_filter = median_filter

        self.adjust_dy()

        ds = self.dataset
        list = ds.getListForPV("LI20:LGPS:3330:BACT")
        self.E_bend = ds.getScalar(list, "LI20:LGPS:3330:BACT")[0]

        self._warned = False

    def forward(self, y):
        """Returns the energy corresponding to the given screen position(s).

        Args:
            y: The y-coordinate, measured from the bottom of the screen [mm].
        """
        return forward(y, self.d_nom, self.E_bend, self.dy)

    def inverse(self, E):
        """Returns the screen position corresponding to the energy(s).

        Args:
            E: The energy [GeV].
        """
        return inverse(E, self.d_nom, self.E_bend, self.dy)

    def getSpectrumImage(self, ind):
        """Returns the spectrum image for a given index in the dataset.

        Before creating the SPEC, this function performs common image analysis process
        on the image based on flags passed to the class constructor.

        Args:
            ind: The index of the image in the dataset.
        """
        ds = self.dataset
        img = ds.getImage(self.cam, ind)
        if self.subtract_background:
            bkgd = ds.getImageBackground(self.cam)
            img.subtract_background(bkgd)
        list = ds.getListForPV("LI20:LGPS:3330:BACT")
        if list is not None:
            E_bend = ds.getScalar(list, "LI20:LGPS:3330:BACT")[ind]
        else:
            E_bend = self.E_bend_default
            print(
                "Dipole setting not found in scalar lists, defaulting to {:0.1f}GeV.".format(
                    E_bend
                )
            )
        if np.isnan(E_bend):
            E_bend = self.E_bend_default
            if not self._warned:
                print(
                    "Dipole setting is nan in scalar lists, defaulting to {:0.1f}GeV.".format(
                        E_bend
                    )
                )
                self._warned = True
        if self.crop is not None:
            img.crop(self.crop)
        if self.median_filter is not None:
            img.median_filter(self.median_filter)
        specImg = SPEC(
            img,
            E_bend,
            self.d_nom,
            self.d_nom_err,
            self.dy,
            self.dy_err,
            self.chargeCal,
            self.chargeCalErr,
        )
        return specImg

    def get_waterfall(self):
        """Builds a waterfall of all the specta in the dataset.

        Returns:
            A 2D np.ndarray with the waterfall.
        """
        if self.wf is not None:
            return self.wf
        N = self.N
        M = self.img.height
        wf = np.zeros((M, N))

        for i in range(N):
            specImg = self.getSpectrumImage(i)
            specImg.getSpectrum()
            wf[:, i] = specImg.spectrum
        self.wf = wf
        return wf

    def get_linear_waterfall(self):
        """Builds a waterfall of all the linearized specta in the dataset.

        Linearized means the spectra is interpolated to a uniform E grid.

        Returns:
            A 2D np.ndarray with the linearized waterfall.
        """
        if self.wf_linear is not None:
            return self.wf_linear

        self.get_waterfall()
        N = self.N
        M = self.img.height
        wf_linear = np.zeros((M, N))

        specImg = self.getSpectrumImage(0)
        E, spectrum, dE = specImg.getSpectrum()
        E_linear = np.linspace(E[0], E[-1], M)

        for i in range(N):
            f = interpolate.interp1d(E, self.wf[:, i])
            wf_linear[:, i] = f(E_linear)
        self.wf_linear = wf_linear
        self.E_linear = E_linear
        return wf_linear

    def adjust_dy(self):
        """Adjust dy if the crop region doesn't reach the bottom of the screen."""
        if self.initial_height != self.crop[3]:
            self.dy -= (self.initial_height - self.crop[3]) * self.img.cal

    def plot_waterfall(self):
        """Plot the waterfall of spectra for the dataset.

        Returns:
            A tuple(fig, ax, im) containing the Matplotlib figure (fig), axis (ax) and imshow (im).
        """
        self.get_waterfall()
        ext = self.img.get_ext()
        ext[0] = -0.5
        ext[1] = self.N - 0.5

        fig = plt.figure(figsize=(6, 4), dpi=300)
        ax = fig.add_subplot()
        ax.dataset_text = ax.text(
            0.01,
            1.02,
            "{}, Dataset: {}".format(self.dataset.experiment, self.dataset.number),
            transform=ax.transAxes,
        )
        im = ax.imshow(
            self.wf,
            aspect="auto",
            cmap=plot.cmap_W_Viridis,
            interpolation="none",
            extent=ext,
        )
        cb = fig.colorbar(im)
        cb.set_label(r"$dQ/dE$ (nC/GeV)")
        im.set_clim(0, np.max(self.wf))
        ax.set_xlabel("Shot number")

        E_ticks = np.arange(
            math.ceil(self.forward(ext[3])),
            math.floor(self.forward(ext[2])) + 1,
        )
        tick_locs = [ext[2] - self.inverse(E) for E in E_ticks]
        tick_lbls = ["{:0.1f}".format(E) for E in E_ticks]
        ax.set_ylabel("Energy (GeV)")
        plt.yticks(tick_locs, tick_lbls)
        return fig, ax, im

    def plot_linear_waterfall(self):
        """Plot the linearized waterfall of spectra for the dataset.

        Returns:
            A tuple(fig, ax, im) containing the Matplotlib figure (fig), axis (ax) and imshow (im).
        """
        self.get_linear_waterfall()
        dE = self.E_linear[1] - self.E_linear[0]
        ext = np.zeros(4)
        ext[0] = -0.5
        ext[1] = self.N - 0.5
        ext[3] = self.E_linear[0] - 0.5 * dE
        ext[2] = self.E_linear[-1] + 0.5 * dE

        fig = plt.figure(figsize=(6, 4), dpi=300)
        ax = fig.add_subplot()
        ax.dataset_text = ax.text(
            0.01,
            1.02,
            "{}, Dataset: {}".format(self.dataset.experiment, self.dataset.number),
            transform=ax.transAxes,
        )
        im = ax.imshow(
            self.wf_linear,
            aspect="auto",
            cmap=plot.cmap_W_Viridis,
            interpolation="none",
            extent=ext,
        )
        cb = fig.colorbar(im)
        cb.set_label(r"$dQ/dE$ (nC/GeV)")
        im.set_clim(0, np.max(self.wf))
        ax.set_xlabel("Shot number")
        ax.set_ylabel("Energy (GeV)")
        return fig, ax, im
