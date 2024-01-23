import argparse
import numpy as np
import dask
from dask.diagnostics import ProgressBar
from netCDF4 import Dataset
from pathlib import Path
from typing import Union, Tuple
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy.stats import pearsonr
from datetime import datetime

FIXED_HEIGHTS = np.array([10, 15, 25, 30, 35, 40, 45, 55, 60, 65, 75, 85, 270])
FIXED_SZA = np.arange(20, 116, 5)  # 19 SZA

FINE_HEIGHTS = np.linspace(0, 100, 340)  # ~0.295 km resolution from 0 to 100 km


@dask.delayed
def get_info(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read from the input data files

    Inputs:
        - file_path (str): full path to the input data file

    Outputs:
        - mean_sza (np.ndarray): array of solar zenith angles (degrees)
        - tangent_height (np.ndarray): array of tangent heights (km)
        - excited_o2 (np.ndarray): array of O2 in the singlet delta state (molec/cm3)
    """
    with Dataset(file_path, "r") as nc:
        excited_o2 = nc["singlet_delta/excited_O2"][:]
        excited_o2_dofs = nc["singlet_delta/excited_O2_dofs"][:]
        sza = nc["solar_zenith_angle"][:]
        tangent_height = nc["tangent_height"][:]

    # make sure the arrays have homogenous dimensions so we can concatenate them between files, pad with NaNs
    while sza.shape[1] < 9:
        sza = np.insert(sza, -1, np.nan, axis=1)
        excited_o2 = np.insert(excited_o2, -1, np.nan, axis=1)
        excited_o2_dofs = np.insert(excited_o2_dofs, -1, np.nan, axis=1)
        tangent_height = np.insert(tangent_height, -1, np.nan, axis=1)

    while sza.shape[-1] < 16:
        sza = np.insert(sza, -1, np.nan, axis=2)
        excited_o2 = np.insert(excited_o2, -1, np.nan, axis=2)
        excited_o2_dofs = np.insert(excited_o2_dofs, -1, np.nan, axis=2)
        tangent_height = np.insert(tangent_height, -1, np.nan, axis=2)

    excited_o2[excited_o2_dofs < 0.9] = np.nan

    return sza, tangent_height, excited_o2


def gaussian(x: Union[float, np.ndarray], a: float, b: float, c: float) -> Union[float, np.ndarray]:
    return a * np.exp(-((x - b) ** 2) / (2 * c**2))


def polynomial(x: np.ndarray, *args) -> np.ndarray:
    return np.sum([v * x**i for i, v in enumerate(args[::-1])], axis=0)


def airglow_lut(indir: str, outfile: str) -> None:
    """
    Use data from:
    Sun, Kang, 2022, "Level 2 data for SCIAMACHY airglow retrieval in 2010"
    https://doi.org/10.7910/DVN/T1WRWQ, Harvard Dataverse, V1
    https://dataverse.harvard.edu/file.xhtml?fileId=6343101&version=1.0

    To build a lookup table for airglow as a function a (height,sza)

    Inputs:
        - indir (str): full path to the input data directory
        - outfile (str): full path to the output LUT file
    """
    file_list = Path(indir).rglob("*.nc")
    with ProgressBar():
        results = dask.compute(*[get_info(i) for i in file_list])

    sza = np.concatenate([i[0] for i in results], axis=0).reshape(-1, 16)
    tangent_height = np.concatenate([i[1] for i in results], axis=0).reshape(-1, 16)
    excited_o2 = np.concatenate([i[2] for i in results], axis=0).reshape(-1, 16)

    fsza = sza.flatten()
    fth = tangent_height.flatten()
    fo2 = excited_o2.flatten()

    ids = ~np.isnan(fsza) & ~np.isnan(fth) & ~np.isnan(fo2)

    x, y = np.meshgrid(FIXED_SZA, FIXED_HEIGHTS)

    excited_o2_lut = griddata((fsza[ids], fth[ids]), fo2[ids], (x, y))

    nSZA = FIXED_SZA.size
    excited_o2_gauss_fit_lut = np.zeros((FINE_HEIGHTS.size, nSZA)) * np.nan
    peak_height = np.zeros(nSZA) * np.nan
    peak_width = np.zeros(nSZA) * np.nan
    peak_amp = np.zeros(nSZA) * np.nan
    for i, sza in enumerate(FIXED_SZA):
        y = np.nan_to_num(excited_o2_lut[:, i], nan=0)  # set nans to 0
        # initial gueses are peak [amplitude,height,width]
        popt, _ = curve_fit(gaussian, FIXED_HEIGHTS, y, p0=[8e10, 50, 10])
        gauss_fit = gaussian(FIXED_HEIGHTS, *popt)
        r_squared = pearsonr(gauss_fit, y)[0] ** 2
        if r_squared > 0.9 and sza <= 100:
            peak_amp[i] = popt[0]
            peak_height[i] = popt[1]
            peak_width[i] = popt[2]

            excited_o2_gauss_fit_lut[:, i] = gaussian(FINE_HEIGHTS, *popt)

    # Parametrize the gaussian parameters as functions of SZA using quadratic fits
    vid = ~np.isnan(peak_height)
    peak_amp_popt, _ = curve_fit(polynomial, FIXED_SZA[vid], peak_amp[vid], p0=[1, 1, 7e10])
    peak_height_popt, _ = curve_fit(polynomial, FIXED_SZA[vid], peak_height[vid], p0=[0, 0, 0])
    peak_width_popt, _ = curve_fit(polynomial, FIXED_SZA[vid], peak_width[vid], p0=[0, 0, 0])

    description = """Lookup table of number density of O2 molecules at singlet delta state
    as a function of solar zenith angle and altitude. Also includes gaussian fits to these
    profiles and the parametrization of the gaussian parameters as functions of solar zenith angle.
    """

    source = """The data used to generate this file was obtained from:
    Sun, Kang, 2022, 'Level 2 data for SCIAMACHY airglow retrieval in 2010'
    https://doi.org/10.7910/DVN/T1WRWQ, Harvard Dataverse, V1
    """

    usage = """You can use peak_amplitude_coefficients, peak_width_coefficients, and
    peak_height_coefficients with a 2nd order polynomial to get the corresponding
    parameter at a given solar zenith angle:
    peak_ampltiude(sza) = poly(sza,peak_amplitude_coefficients)
    Then the excited_o2 profile can be obtained as:
    peak_ampltiude*exp(-(height-peak_height)**2/(2*peak_width**2))
    """

    # Write the lookup table
    with Dataset(outfile, "w") as nc:
        nc.history = f"Created on {datetime.utcnow()} UTC"
        nc.description = description
        nc.source = source
        nc.usage = usage

        # used to save the parametrized gaussian coefficents
        nc.createDimension("fit", 3)
        nc.createVariable("peak_amplitude_coefficients", np.float32, ("fit",))
        nc["peak_amplitude_coefficients"][:] = peak_amp_popt
        peak_amp_atts = {
            "standard_name": "peak_amplitude_coefficients",
            "long_name": "peak amplitude coefficients",
            "description": """2nd order polynomial coefficients for the gaussian peak amplitude of
            excited_O2 as a function of solar zenith angle.
            Peak amplitude = coef[0]*SZA**2+coef[1]*SZA+coef[2]""",
        }
        nc["peak_amplitude_coefficients"].setncatts(peak_amp_atts)

        nc.createVariable("peak_height_coefficients", np.float32, ("fit",))
        nc["peak_height_coefficients"][:] = peak_height_popt
        peak_height_atts = {
            "standard_name": "peak_height_coefficients",
            "long_name": "height coefficients",
            "description": """2nd order polynomial coefficients for the gaussian peak height of
            excited_O2 as a function of solar zenith angle.
            Peak height = coef[0]*SZA**2+coef[1]*SZA+coef[2]""",
        }
        nc["peak_amplitude_coefficients"].setncatts(peak_height_atts)

        nc.createVariable("peak_width_coefficients", np.float32, ("fit",))
        nc["peak_width_coefficients"][:] = peak_width_popt
        peak_width_atts = {
            "standard_name": "peak_width_coefficients",
            "long_name": "width coefficients",
            "description": """2nd order polynomial coefficients for the gaussian peak width of
            excited_O2 as a function of solar zenith angle.
            Peak width = coef[0]*SZA**2+coef[1]*SZA+coef[2]""",
        }
        nc["peak_amplitude_coefficients"].setncatts(peak_width_atts)

        nc.createDimension("sza", nSZA)
        nc.createVariable("sza", np.float32, dimensions=("sza",))
        nc["sza"][:] = FIXED_SZA
        sza_atts = {
            "standard_name": "solar_zenith_angle",
            "long_name": "solar zenith angle",
            "units": "degrees",
        }
        nc["sza"].setncatts(sza_atts)

        nc.createDimension("height", FIXED_HEIGHTS.size)
        nc.createVariable("height", np.float32, dimensions=("height",))
        nc["height"][:] = FIXED_HEIGHTS
        height_atts = {
            "standard_name": "height",
            "long_name": "height",
            "units": "km",
        }
        nc["height"].setncatts(height_atts)

        nc.createDimension("fine_height", FINE_HEIGHTS.size)
        nc.createVariable("fine height", np.float32, dimensions=("fine_height",))
        nc["fine height"][:] = FINE_HEIGHTS
        height_atts = {
            "standard_name": "fine_height",
            "long_name": "fine height",
            "units": "km",
        }
        nc["height"].setncatts(height_atts)

        nc.createVariable("excited_o2", np.float32, dimensions=("height", "sza"))
        nc["excited_o2"][:] = excited_o2_lut
        excited_o2_atts = {
            "standard_name": "excited_o2",
            "long_name": "excited o2",
            "units": "molec/cm3",
            "description": "number density of O2 molecules at singlet delta state",
        }
        nc["excited_o2"].setncatts(excited_o2_atts)

        nc.createVariable("excited_o2_gauss_fit", np.float32, dimensions=("fine_height", "sza"))
        nc["excited_o2_gauss_fit"][:] = excited_o2_gauss_fit_lut
        excited_o2_gauss_fit_atts = {
            "standard_name": "excited_o2_gauss_fit",
            "long_name": "excited o2 gauss fit",
            "units": "molec/cm3",
            "description": "gauss fit to the number density of O2 molecules at singlet delta state",
        }
        nc["excited_o2_gauss_fit"].setncatts(excited_o2_gauss_fit_atts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--indir",
        help="full path to input directory",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="SCIAMACHY_airglow_lut.nc",
        help="full path to the output LUT file",
    )
    args = parser.parse_args()

    airglow_lut(args.indir, args.outfile)


if __name__ == "__main__":
    main()
