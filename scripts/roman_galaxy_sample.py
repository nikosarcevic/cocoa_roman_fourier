#  Niko Sarcevic
#  nikolina.sarcevic@gmail.com
#  github/nikosarcevic
#  ----------

import numpy as np
from numpy import exp
import os
import pandas
from scipy.integrate import simpson, cumulative_trapezoid
from scipy.special import erf
import yaml

class RomanGalaxySample:
    """
    Handles the generation and processing of redshift distributions and tomographic binning
    for Roman galaxy samples (source and lens) for different forecast years.
    Combines functionality for generating Smail-type distributions and slicing them into bins.
    """

    def __init__(self, forecast_year, redshift_range=None, decimal_places=None, verbose=False):
        # Load the YAML file
        # Define parameter file path internally (not exposed to user)
        param_file = "pz_parameters.yaml"
        yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "parameters", param_file))

        # Load the YAML file
        with open(yaml_path, "r") as f:
            survey_parameters = yaml.load(f, Loader=yaml.FullLoader)

        # Handle redshift grid setup
        if redshift_range is None:
            nz_grid = survey_parameters["nz_grid"]  # Raises KeyError if missing
            zmin = nz_grid["zmin"]
            zmax = nz_grid["zmax"]
            dz = nz_grid["dz"]  # or use .get("dz") if it's optional
            if dz is not None:
                self.redshift_range = np.arange(zmin, zmax, dz)
                print(f"Using nz_grid with dz={dz}, zmin={zmin}, zmax={zmax}")
            else:
                self.redshift_range = redshift_range
                print("Using user-provided redshift range.")

        # Forecast year validation
        supported_forecast_years = {"1"}
        if forecast_year in supported_forecast_years:
            self.forecast_year = forecast_year
        else:
            raise ValueError(f"Forecast_year must be one of {supported_forecast_years}.")

        # Default decimal places (use to round the values)
        self.decimal_places = decimal_places if decimal_places is not None else 4
        self.verbose = verbose

        # Assign galaxy sample parameters
        self.lens_params = survey_parameters.get("lens_sample", {}).get(self.forecast_year, {})
        self.source_params = survey_parameters.get("source_sample", {}).get(self.forecast_year, {})
        self.lens_type = "lens_sample"
        self.source_type = "source_sample"

        if self.verbose:
            print(f"[INIT] Forecast year: {self.forecast_year}")
            print(f"[INIT] Redshift grid: {self.redshift_range[0]:.2f}–{self.redshift_range[-1]:.2f} "
                  f"with {len(self.redshift_range)} points")

    def lens_sample(self, normalized=True, save_file=True, file_format="npy", decimal_places=None):
        """
        Generates the lens sample redshift distribution.

        Arguments:
            normalized (bool): Normalize the redshift distribution (default: True).
            save_file (bool): Save the output file (default: True).
            file_format (str): Output file format ('npy', 'csv').
            decimal_places (int or None): Number of decimal places to round outputs.
                If None, defaults to self.decimal_places.

        Returns:
            np.array: Normalized p(z), suitable for CCL.
        """
        decimal_places = decimal_places if decimal_places is not None else self.decimal_places

        # Compute the Smail-type redshift distribution (not yet normalized)
        p_z = self.smail_type_distribution(self.redshift_range,
                                           self.lens_params["z_0"],
                                           self.lens_params["alpha"],
                                           self.lens_params["beta"])

        # Normalize the distribution if requested (ALWAYS needed for CCL)
        if normalized:
            p_z = self.normalize_distribution(p_z, method="trapz")

        # Save the data if requested
        if save_file:
            self.save_to_file("lens_sample_pz",
                              {"redshift": np.round(self.redshift_range, decimal_places),
                               "distribution": np.round(p_z, decimal_places)},
                              file_format)

        return p_z

    def source_sample(self, normalized=True, save_file=True, file_format="npy", decimal_places=None):
        """
        Generates the source sample redshift distribution.

        Arguments:
            normalized (bool): Normalize the redshift distribution (default: True).
            save_file (bool): Save the output file (default: True).
            file_format (str): Output file format ('npy', 'csv').
            decimal_places (int or None): Number of decimal places to round outputs.
                If None, defaults to self.decimal_places.

        Returns:
            np.array: Normalized p(z), suitable for CCL.
        """
        decimal_places = decimal_places if decimal_places is not None else self.decimal_places

        # Compute the Smail-type redshift distribution (not yet normalized)
        p_z = self.smail_type_distribution(self.redshift_range,
                                           self.source_params["z_0"],
                                           self.source_params["alpha"],
                                           self.source_params["beta"])

        # Normalize the distribution if requested (ALWAYS needed for CCL)
        if normalized:
            p_z = self.normalize_distribution(p_z, method="trapz")

        # Save the data if requested
        if save_file:
            self.save_to_file("source_sample_pz",
                              {"redshift": np.round(self.redshift_range, decimal_places),
                               "distribution": np.round(p_z, decimal_places)},
                              file_format)

        return p_z

    def lens_bins(self,
                  normalized=True,
                  save_file=True,
                  file_format='npy',
                  decimal_places=None):
        """
        Split the initial redshift distribution of lens galaxies (lenses) into tomographic bins.
        ----------
        Arguments:
            normalized: bool
                normalize the redshift distribution (defaults to True).
            save_file: bool
                option to save the output (defaults to True).
            file_format: str (defaults to 'npy')
                format of the output file. Accepted values are 'csv', 'txt', and 'npy'.
            decimal_places: int or None
                Decimal places for rounding. If None, uses self.decimal_places.
        Returns:
            dict: A lens galaxy sample, appropriately binned.
        """
        decimal_places = decimal_places if decimal_places is not None else self.decimal_places

        # Generate the redshift distribution for the lens sample
        lens_dndz = self.lens_sample(normalized=normalized, save_file=False)

        # Define the bin edges
        bin_edges = self.get_tomo_bin_edges("lens")

        # Create a dictionary of the redshift distributions for each bin
        redshift_distribution_dict = self.create_redshift_bins(lens_dndz, bin_edges, self.lens_params)

        # Normalize the distributions
        if normalized:
            redshift_distribution_dict = self.normalize_distribution(redshift_distribution_dict, method="trapz")

        # Round output
        rounded_bins = {k: np.round(v, decimal_places) for k, v in redshift_distribution_dict.items()}
        rounded_z = np.round(self.redshift_range, decimal_places)

        combined_data = {'redshift_range': rounded_z, 'bins': rounded_bins}

        if save_file:
            self.save_to_file("lens_bins", combined_data, file_format)

        return rounded_bins

    def source_bins(self,
                    normalized=True,
                    save_file=True,
                    file_format='npy',
                    decimal_places=None):
        """
        Split the initial redshift distribution of source galaxies into tomographic bins.

        Arguments:
            normalized (bool): Normalize the redshift distribution (default: True).
            save_file (bool): Save the output (default: True).
            file_format (str): Output file format ('npy', 'txt', etc.).
            decimal_places (int or None): Number of decimal places for rounding. Defaults to self.decimal_places.

        Returns:
            dict: A dictionary of the binned source redshift distributions.
        """
        decimal_places = decimal_places if decimal_places is not None else self.decimal_places

        # Generate the redshift distribution for the source sample
        source_dndz = self.source_sample(normalized=normalized, save_file=False)

        # Get the bin edges as redshift values directly
        bin_edges = self.get_tomo_bin_edges("source")

        # Create a dictionary of the redshift distributions for each bin
        redshift_distribution_dict = self.create_redshift_bins(source_dndz, bin_edges, self.source_params)

        # Normalize the distributions
        if normalized:
            redshift_distribution_dict = self.normalize_distribution(redshift_distribution_dict, method="trapz")

        # Round output
        rounded_bins = {k: np.round(v, decimal_places) for k, v in redshift_distribution_dict.items()}
        rounded_z = np.round(self.redshift_range, decimal_places)

        combined_data = {'redshift_range': rounded_z, 'bins': rounded_bins}

        if save_file:
            self.save_to_file("source_bins", combined_data, file_format)

        return rounded_bins

    def lens_bin_centers(self, decimal_places=2, save_file=True, file_format="npy"):
        """
        Compute the lens bin centers for the Roman forecast year.

        Arguments:
            decimal_places (int or None): Number of decimal places to round the bin centers.
                                          Defaults to self.decimal_places if not specified.
            save_file (bool): Option to save the output (default: True).
            file_format (str): Output file format ('npy', 'txt').

        Returns:
            dict: Dictionary of bin centers with bin indices as keys.
        """
        decimal_places = decimal_places if decimal_places is not None else self.decimal_places

        lens_bins = self.lens_bins(normalized=True, save_file=False)
        bin_centers = self.compute_tomo_bin_centers(lens_bins, decimal_places=decimal_places)

        if save_file:
            self.save_to_file("lens_bin_centers", bin_centers, file_format)

        return bin_centers

    def source_bin_centers(self, decimal_places=2, save_file=True, file_format="npy"):
        """
        Compute the source bin centers for the Roman forecast year.

        Arguments:
            decimal_places (int or None): Number of decimal places to round the bin centers.
                                          Defaults to self.decimal_places if not specified.
            save_file (bool): Option to save the output (default: True).
            file_format (str): Format of the output file ('npy', 'txt').

        Returns:
            dict: Dictionary of bin centers with bin indices as keys.
        """
        decimal_places = decimal_places if decimal_places is not None else self.decimal_places

        source_bins = self.source_bins(normalized=True, save_file=False)
        bin_centers = self.compute_tomo_bin_centers(source_bins, decimal_places=decimal_places)

        if save_file:
            self.save_to_file("source_bin_centers", bin_centers, file_format)

        return bin_centers

    def smail_type_distribution(self,
                                redshift_range,
                                pivot_redshift,
                                alpha,
                                beta):

        """
        Generate a Smail-type parametric redshift distribution of the form:

        N(z) = (z / z0)^β * exp[-(z / z0)^α],

        where z is the redshift, z0 is the pivot redshift, and α and β are power-law indices
        controlling the shape of the distribution.

        ----------
        Arguments:
            redshift_range: array
                redshift range
            pivot_redshift: float
                pivot redshift
            alpha: float
                power law index in the prefctor
            beta: float
                power law index in the exponent
        Returns:
            redshift_distribution: array
                A Smail-type redshift distribution over a range of redshifts.
                """

        redshift_distribution = [(z / pivot_redshift) ** alpha * exp(-(z / pivot_redshift) ** beta) for z in redshift_range]

        return np.array(redshift_distribution)

    def true_redshift_distribution(self, redshift_distribution, bin_min, bin_max, sigma_z, z_bias):
        """Compute the true (photometric) redshift distribution of a galaxy sample.

        The true distribution is obtained by applying the probability distribution p(z_phot | z),
        which accounts for photometric errors, to the overall galaxy redshift distribution.

        This follows Ma, Hu & Huterer 2018 (https://arxiv.org/abs/astro-ph/0506614, Eq. 6).

        Arguments:
            redshift_distribution (array): The intrinsic redshift distribution n(z).
            bin_min (float): Lower bound of the redshift bin.
            bin_max (float): Upper bound of the redshift bin.
            sigma_z (float): Photometric redshift scatter (variance).
            z_bias (float): Photometric redshift bias.

        Returns:
            p_z_photometric (array): The convolved redshift distribution after including photometric errors.
        """
        # Compute photometric scatter as a function of redshift
        scatter = np.maximum(sigma_z * (1 + self.redshift_range), 1e-10)  # Avoid division by zero

        # Compute integration limits
        upper_limit = (bin_max - self.redshift_range + z_bias) / (np.sqrt(2) * scatter)
        lower_limit = (bin_min - self.redshift_range + z_bias) / (np.sqrt(2) * scatter)

        # Compute the photometric redshift distribution
        p_z_photometric = 0.5 * np.array(redshift_distribution) * (erf(upper_limit) - erf(lower_limit))

        return p_z_photometric

    def compute_cumulative_distribution(self, redshift_range, redshift_distribution):
        """
        Computes the cumulative distribution and total number of galaxies.

        Arguments:
            redshift_range: ndarray
                Array of redshift values.
            redshift_distribution: ndarray
                Redshift distribution over the given range.

        Returns:
            cumulative_distribution: ndarray
                The cumulative distribution of the redshift values.
            total_galaxies: float
                The total number of galaxies (area under the distribution).
        """
        cumulative_distribution = cumulative_trapezoid(redshift_distribution, redshift_range, initial=0)
        total_galaxies = cumulative_distribution[-1]
        return cumulative_distribution, total_galaxies

    def compute_equal_number_bounds(self, redshift_range, redshift_distribution, n_bins):
        """
        Determines the redshift values that divide the distribution into bins
        with an equal number of galaxies.

        Arguments:
            redshift_range (array): an array of redshift values
            redshift_distribution (array): the corresponding redshift distribution defined over redshift_range
            n_bins (int): the number of tomographic bins

        Returns:
            An array of redshift values that are the boundaries of the bins.
        """
        redshift_range = np.asarray(redshift_range)
        redshift_distribution = np.asarray(redshift_distribution)

        # Calculate the cumulative distribution
        cumulative_distribution, total_galaxies = self.compute_cumulative_distribution(redshift_range,
                                                                                       redshift_distribution)

        # Find the bin edges
        bin_edges = []
        for i in range(1, n_bins):
            fraction = i / n_bins * total_galaxies
            # Find the redshift value where the cumulative distribution crosses this fraction
            bin_edge = np.interp(fraction, cumulative_distribution, redshift_range)
            bin_edges.append(bin_edge)

        return [redshift_range[0]] + bin_edges + [redshift_range[-1]]

    def create_redshift_bins(self, redshift_distribution, bin_edges, pz_parameters):
        """
        Creates the redshift bins for the given distribution and bin edges.

        Arguments:
            redshift_distribution (array): The overall redshift distribution.
            bin_edges (array): The edges of the redshift bins.
            pz_parameters (dict): Parameter dictionary containing 'n_tomo_bins', 'sigma_z', and 'z_bias'.

        Returns:
            dict: A dictionary of redshift distributions per bin.
        """
        n_bins = pz_parameters["n_tomo_bins"]
        z_bias_list = np.repeat(pz_parameters["z_bias"], n_bins)
        z_variance_list = np.repeat(pz_parameters["sigma_z"], n_bins)

        redshift_distribution_dict = {}
        for index, (z_min, z_max) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            redshift_distribution_dict[index] = self.true_redshift_distribution(
                redshift_distribution,
                bin_min=z_min,
                bin_max=z_max,
                sigma_z=z_variance_list[index],
                z_bias=z_bias_list[index]
            )
        return redshift_distribution_dict

    def normalize_distribution(self, redshift_distribution, method: str = "trapz"):
        """
        Normalizes redshift distributions.

        Arguments:
            redshift_distribution: dict or array
                A dictionary containing redshift distributions for each bin,
                or a single array for an unbinned distribution.
            method: str
                The normalization method to use. Options: 'trapz' (default) and 'simpson'.

        Returns:
            normalized_distribution: dict or array
                Normalized distributions, same structure as input.
        """
        valid_methods = {"trapz", "simpson"}
        if method not in valid_methods:
            raise ValueError(f"Unsupported normalization method: {method}. Choose from {valid_methods}")

        def compute_norm_factor(distribution: np.ndarray) -> float:
            """Helper function to compute normalization factor from a 1D array."""
            if distribution.ndim != 1:
                raise ValueError("Expected a 1D array for normalization.")

            result = simpson(distribution, x=self.redshift_range) if method == "simpson" else np.trapz(distribution,
                                                                                                       x=self.redshift_range)
            norm_factor = float(result)

            if np.isclose(norm_factor, 0.0, atol=1e-10):
                raise ValueError("Normalization factor is zero or too small, check the input distribution.")

            return norm_factor

        # Handle single array input
        if isinstance(redshift_distribution, np.ndarray):
            return redshift_distribution / compute_norm_factor(redshift_distribution)

        # Handle dictionary input
        if isinstance(redshift_distribution, dict):
            return {key: dist / compute_norm_factor(dist) for key, dist in redshift_distribution.items()}

        raise TypeError("Input must be a dictionary of distributions or a single numpy array.")

    def get_n_tomo_bins(self, sample_type: str) -> int:
        n_bins = self.lens_params["n_tomo_bins"] if sample_type == "lens" else self.source_params["n_tomo_bins"]
        return int(n_bins)

    def get_tomo_bin_edges(self, sample_type):
        """
        Get the redshift tomographic bin edges for the given galaxy sample type.

        Arguments:
            sample_type: str
                the type of the galaxy sample (lens or source)
        Returns:
            bin_edges: array
                the redshift bin edges
        """
        sample_dict = {
            "source": self.compute_equal_number_bounds(
                self.redshift_range,
                self.source_sample(),
                self.get_n_tomo_bins("source")
            ),
            "lens": np.linspace(
                self.lens_params["bin_start"],
                self.lens_params["bin_stop"],
                self.get_n_tomo_bins("lens") + 1
            )
        }

        return sample_dict[sample_type]

    def compute_tomo_bin_centers(self, bins_dict, decimal_places=2):
        """
        Compute tomographic bin centers from the bins dictionary using numpy.trapz.

        Arguments:
            bins_dict (dict): Dictionary of redshift distributions per bin.
            decimal_places (int): Number of decimal places to round the bin centers.

        Returns:
            bin_centers (dict): Dictionary of bin centers with bin indices as keys,
                                rounded to the specified decimal places.
        """
        bin_centers = {}

        for bin_idx, distribution in bins_dict.items():
            weighted_sum = np.trapz(distribution * self.redshift_range, self.redshift_range)
            total_area = np.trapz(distribution, self.redshift_range)

            # Calculate weighted mean redshift (bin center)
            weighted_mean = weighted_sum / total_area

            # Round the result to the specified number of decimal places
            bin_centers[bin_idx] = round(weighted_mean, decimal_places)

        return bin_centers

    def save_to_file(self, data_name, data, file_format="npy"):
        """
        Save data to a file.

        ...

        file_format: str
            The format of the output file (default: 'npy').
            Accepted values are 'txt' and 'npy'.
        """

        # Make sure 'data_output' directory exists
        output_dir = "data_products"
        os.makedirs(output_dir, exist_ok=True)

        # Define file path
        file_path = os.path.join(output_dir, f"roman_{data_name}_year_{self.forecast_year}.{file_format}")

        if file_format == "npy":
            np.save(file_path, data)

        elif file_format == "txt":
            try:
                # Attempt to save directly if data is not nested
                dndz_df = pandas.DataFrame(data)
                dndz_df.to_csv(file_path, sep="\t", index=False, float_format="%.6f")
            except ValueError:
                # Handle nested dictionary cases
                for key, sub_data in data.items():
                    if isinstance(sub_data, dict):
                        sub_df = pandas.DataFrame(sub_data)
                        sub_df.to_csv(
                            os.path.join(output_dir, f"roman_{data_name}_{key}_year_{self.forecast_year}.txt"),
                            sep="\t", index=False, float_format="%.6f"
                        )
        else:
            raise ValueError(f"Unsupported file format: {file_format}. Use 'npy' or 'txt'.")
