# cluster_generating.py

import numpy as np
from astropy.table import Table, join
from scipy.interpolate import interp1d
import imf
from scopesim_templates.stellar import cluster_utils as cu
from astropy.io import fits, ascii
from copy import deepcopy
import scopesim as sim
from scopesim.source import source_templates as sim_tp
from scopesim_templates.stellar.stars import stars
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy import units as u
from mw_plot import MWSkyMap
import jinja2
import os

def generate_cluster_masses(total_mass, imf_type):
    """Generate stellar masses for a cluster using a specified IMF (IMF package by keflavich: https://github.com/keflavich/imf)"""
    if imf_type.lower() == 'kroupa':
        return imf.make_cluster(total_mass)
    elif imf_type.lower() == 'salpeter':
        return imf.make_cluster(total_mass, massfunc='salpeter')
    elif imf_type.lower() == 'chabrier':
        return imf.make_cluster(total_mass, massfunc='chabrier')
    else:
        raise ValueError("IMF type must be 'kroupa', 'salpeter', or 'chabrier'.")

def interpolate_magnitudes(masses):
    """Interpolate J and Ks magnitudes based on stellar masses."""
    mass_data = np.array([20.2, 18.7, 17.7, 14.8, 11.8, 9.9, 7.3, 6.1, 5.4, 5.1, 
                          4.7, 4.3, 3.92, 3.38, 2.75, 2.68, 2.18, 2.05, 1.98, 1.86, 
                          1.93, 1.88, 1.83, 1.77, 1.81, 1.75, 1.61, 1.5, 1.46, 1.44, 
                          1.38, 1.33, 1.25, 1.21, 1.18, 1.13, 1.08, 1.06, 1.03, 1.0, 
                          0.99, 0.985, 0.98, 0.97, 0.95, 0.94, 0.9, 0.88, 0.86, 0.82, 
                          0.78, 0.73, 0.7, 0.69, 0.64, 0.62, 0.59, 0.57, 0.54, 0.5, 
                          0.47, 0.44, 0.4, 0.37, 0.27, 0.23, 0.184, 0.162, 0.123, 0.102, 
                          0.093, 0.09, 0.088, 0.085, 0.08, 0.079, 0.078, 0.077, 0.076, 0.075])

    mj_data = np.array([-3.44, -3.3, -3.17, -2.8, -2.33, -2.03, -1.34, -1.09, -0.83, -0.66, 
                        -0.54, -0.28, -0.16, 0.19, 0.59, 0.63, 0.95, 1.08, 1.19, 1.49, 
                        1.65, 1.68, 1.74, 1.8, 1.81, 1.89, 1.98, 2.11, 2.24, 2.32, 
                        2.4, 2.52, 2.76, 2.85, 3.05, 3.21, 3.26, 3.37, 3.47, 3.6, 
                        3.66, 3.7, 3.73, 3.81, 3.9, 3.96, 4.14, 4.31, 4.39, 4.46, 
                        4.7, 4.91, 5.1, 5.31, 5.59, 5.75, 5.82, 5.97, 6.19, 6.48, 
                        6.59, 6.81, 7.01, 7.38, 7.93, 8.2, 8.8, 9.09, 9.72, 10.18, 
                        10.47, 10.7, 10.88, 11.05, 11.46, 11.59, 11.75, 11.76, 12.12, 
                        12.47])

    mks_data = np.array([-3.20, -3.073, -2.942, -2.587, -2.126, -1.848, -1.198, -0.956, -0.708, -0.553, 
                         -0.376 -0.192, -0.075, 0.254, 0.621, 0.648, 0.949, 1.07, 1.07, 1.172, 
                         1.447, 1.587, 1.607, 1.655, 1.702, 1.694, 1.756, 1.836, 1.941, 2.045, 
                         2.116, 2.188, 2.291, 2.5, 2.579, 2.76, 2.915, 2.947, 3.043, 3.12, 
                         3.236, 3.282, 3.319, 3.345, 3.41, 3.49, 3.53, 3.69, 3.82, 3.89, 
                         3.93, 4.1, 4.25, 4.4, 4.56, 4.81, 4.95, 5.01, 5.15, 5.36, 
                         5.64, 6.75, 5.98, 6.18, 6.55, 7.10, 7.36, 7.93, 8.2, 8.8, 
                         9.22, 9.5, 9.7, 9.81, 9.92, 10.3, 10.4, 10.5, 10.55, 10.77,11])

    interp_func_j = interp1d(mass_data, mj_data, kind='linear', fill_value=(mj_data[-1], mj_data[0]), bounds_error=False)
    interp_func_ks = interp1d(mass_data, mks_data, kind='linear', fill_value=(mks_data[-1], mks_data[0]), bounds_error=False)

    mj_magnitudes = interp_func_j(masses)
    mks_magnitudes = interp_func_ks(masses)

    return mj_magnitudes, mks_magnitudes

def king_density(r, r_c, r_t, sigma_0):
    """Calculate the King density profile for a given radius."""
    term1 = (1 + (r / r_c)**2)**-0.5
    term2 = (1 + (r_t / r_c)**2)**-0.5
    return sigma_0 * (term1 - term2)**2

def compute_king_cdf(r_c, r_t, sigma_0, steps=1000):
    """Compute the cumulative distribution function (CDF) for the King profile."""
    r = np.linspace(0, r_t, steps)
    densities = king_density(r, r_c, r_t, sigma_0)
    cdf = np.cumsum(densities) * np.diff(r, prepend=0)
    cdf /= cdf[-1]
    return interp1d(cdf, r, bounds_error=False, fill_value=(r[0], r[-1]))

def sample_king_profile(r_c, r_t, sigma_0, n_stars):
    """Sample positions for stars based on the King profile."""
    cdf_func = compute_king_cdf(r_c, r_t, sigma_0)
    random_values = np.random.rand(n_stars)
    radii = cdf_func(random_values)
    angles = np.random.uniform(0, 2 * np.pi, n_stars)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y

def filter_stars_by_age(cluster_table, log_age):
    """Filter stars in the cluster based on the age and spectral type."""
    if log_age >= 9.0:
        spectral_types_to_keep = ['F', 'G', 'K', 'M']
    elif log_age >= 8.0:
        spectral_types_to_keep = ['A', 'F', 'G', 'K', 'M']
    elif log_age >= 7.0:
        spectral_types_to_keep = ['B', 'A', 'F', 'G', 'K', 'M']
    else:
        spectral_types_to_keep = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

    filtered_table = cluster_table[np.isin([s[0] for s in cluster_table['Spectral Type']], spectral_types_to_keep)]
    return filtered_table

def create_cluster_table(total_mass, distance, imf_type, core_radius, tidal_radius, norm_factor, log_age):
    # Generate cluster masses based on the IMF type
    cluster_masses = generate_cluster_masses(total_mass, imf_type)

    # Interpolate magnitudes for J and Ks bands
    j_magnitudes_absolute, ks_magnitudes_absolute = interpolate_magnitudes(cluster_masses)

    # Calculate apparent magnitudes for J and Ks bands
    j_magnitudes_apparent = j_magnitudes_absolute + 5 * np.log10(distance / 10)
    ks_magnitudes_apparent = ks_magnitudes_absolute + 5 * np.log10(distance / 10)

    # Generate star positions using the King profile
    x_positions, y_positions = sample_king_profile(core_radius, tidal_radius, norm_factor, len(cluster_masses))

    # Convert positions from parsecs to arcseconds
    x_positions_arcsec = x_positions / distance * 206265
    y_positions_arcsec = y_positions / distance * 206265

    # Assign spectral types based on cluster masses
    spectral_types = cu.mass2spt(cluster_masses)

    # Create an Astropy Table with all the calculated data
    cluster_table = Table([cluster_masses, j_magnitudes_absolute, j_magnitudes_apparent, ks_magnitudes_absolute, ks_magnitudes_apparent, x_positions, y_positions,
                           x_positions_arcsec, y_positions_arcsec, spectral_types],
                          names=['Mass', 'J Mag(abs)', 'J Mag(app)', 'Ks Mag(abs)', 'Ks Mag(app)', 'X Position (pc)', 'Y Position (pc)',
                                 'X Position (arcsec)', 'Y Position (arcsec)', 'Spectral Type'])

    # Filter stars based on the age of the cluster
    filtered_table = filter_stars_by_age(cluster_table, log_age)
    
    return filtered_table

def get_cluster_params(cluster_name, table):
    cluster_row = table[table['NAME'] == cluster_name]
    if len(cluster_row) == 0:
        raise ValueError(f"Cluster with name {cluster_name} not found.")
    
    params = {
        'total_cluster_mass': cluster_row['TIDAL_MASS'][0],
        'r_c': cluster_row['KING_CORE_RADIUS'][0],
        'r_t': cluster_row['KING_TIDAL_RADIUS'][0],
        'sigma_0': cluster_row['KING_NORM_FACTOR'][0],
        'cluster_distance': cluster_row['DISTANCE'][0],
        'log_age': cluster_row['LOG_AGE'][0],
        'gal_lon': cluster_row['GAL_LON'][0],
        'gal_lat': cluster_row['GAL_LAT'][0],
        'ra': cluster_row['RA'][0],
        'dec': cluster_row['DEC'][0],
        'num_stars_core': cluster_row['NUM_STARS_CORE'][0],
        'num_stars_micado': cluster_row['NUM_STARS_MICADO'][0],
        'num_stars_tidal': cluster_row['NUM_STARS_TIDAL'][0],
        'num_stars_centraldetector': cluster_row['NUM_STARS_CENTRALDETECTOR'][0],
        'star_density_tidal': cluster_row['STAR_DENSITY_TIDAL'][0],
        'star_density_core': cluster_row['STAR_DENSITY_CORE'][0],
        'star_density_centraldetector': cluster_row['STAR_DENSITY_CENTRALDETECTOR'][0],
        'pixels_per_star_micado': cluster_row['PIXELS_PER_STAR_MICADO'][0],
        'pixels_per_star_jwst': cluster_row['PIXELS_PER_STAR_JWST'][0],
        'pixels_per_star_hawki': cluster_row['PIXELS_PER_STAR_HAWKI'][0],
    }
    return params


def simulate_micado(cluster_table, filter_name='J'):
    """Simulate an observation with the MICADO instrument."""
    micado = sim.OpticalTrain("MICADO")
    src = stars(filter_name, 
                cluster_table[f'{filter_name} Mag(app)'], 
                cluster_table['Spectral Type'], 
                cluster_table['X Position (arcsec)'], 
                cluster_table['Y Position (arcsec)'])
    
    micado.cmds['!DET.width'] = 4096
    micado.cmds['!DET.height'] = 4096
    micado.cmds['!OBS.dit'] = 60
    micado.cmds['!OBS.ndit'] = 60
    
    if filter_name == 'J':
        micado.cmds['!OBS.filter_name_fw1'] = "J"
        micado.cmds['!OBS.filter_name_fw2'] = "open"
    elif filter_name == 'Ks':
        micado.cmds['!OBS.filter_name_fw1'] = "open"
        micado.cmds['!OBS.filter_name_fw2'] = "Ks"
    else:
        raise ValueError("Invalid filter name. Choose 'J' or 'Ks'.")

    micado.observe(src)
    hdus = micado.readout()[0]
    return hdus, micado


def perform_photometry(hdus, micado):
    """Perform aperture photometry on the simulated data."""
    data = hdus[1].data
    star_positions = Table({
        'x': micado._last_fovs[0].fields[0]['x'] * 250 + 2048,
        'y': micado._last_fovs[0].fields[0]['y'] * 250 + 2048,
        'spectral_type': micado._last_fovs[0].fields[0]['spec_types']
    })

    results = []
    ap0, ap1, ap2 = 7, 20, 30
    ndit = 25

    for row in star_positions:
        x, y, spt = int(row['x']), int(row['y']), row['spectral_type']
        im_cutout = data[y-ap2:y+ap2, x-ap2:x+ap2]
        sig, snr = aperture_photometry(im_cutout, ap0, ap1, ap2, ndit=ndit)
        results.append((x, y, spt, sig, snr))

    dtype = [('x', int), ('y', int), ('spectral_type', 'U10'), ('flux', float), ('snr', float)]
    results_array = np.array(results, dtype=dtype)
    results_table = Table(results_array)

    return results_table


def aperture_photometry(im, ap0=10, ap1=20, ap2=30, ndit=60, ron=10):
    """Perform aperture photometry on a given image cutout."""
    x0, y0 = im.shape[0] // 2, im.shape[1] // 2
    bg = np.copy(im)
    bg[y0-ap1:y0+ap1, x0-ap1:x0+ap1] = 0
    bg_median = np.median(bg[bg!=0])
    n_pix = (2 * ap0) ** 2
    sig = np.sum(im[y0-ap0:y0+ap0, x0-ap0:x0+ap0] - bg_median)
    noise_sig = sig
    noise_sky = bg_median * n_pix
    noise_ron = ron**2 * ndit * n_pix
    noise = np.sqrt(noise_sig + noise_sky + noise_ron)
    snr = sig / noise
    return sig, snr


def calculate_snr_percentage(photometry_results, snr_threshold, snr_column):
    """Calculate the percentage of stars with SNR above a given threshold."""
    total_stars = len(photometry_results)
    stars_above_threshold = len(photometry_results[photometry_results[snr_column] > snr_threshold])
    percentage = (stars_above_threshold / total_stars) * 100
    return percentage

def print_snr_percentages(photometry_results_j, photometry_results_ks):
    """Calculate and print the percentages for SNR > 5 and SNR > 10 for both filters."""
    for filter_name, photometry_results, snr_column in zip(
        ['J', 'Ks'], 
        [photometry_results_j, photometry_results_ks], 
        ['SNR J', 'SNR Ks']
    ):
        snr5_percentage = calculate_snr_percentage(photometry_results, 5, snr_column)
        snr10_percentage = calculate_snr_percentage(photometry_results, 10, snr_column)
        print(f"Filter: {filter_name}")
        print(f"Percentage of stars with SNR > 5: {snr5_percentage:.2f}%")
        print(f"Percentage of stars with SNR > 10: {snr10_percentage:.2f}%\n")


def plot_cluster_image_j(hdus, cluster_name, plot_dir, contrast_factor=10):
    data = hdus[1].data
    # Calculate the background noise
    background_noise = np.std(data)
    # Ensure contrast_factor is a numeric value
    if not isinstance(contrast_factor, (int, float)):
        raise ValueError("contrast_factor must be a numeric value")
    
    # Set the contrast limits
    vmin = 10000  # or other suitable value for J filter
    vmax = contrast_factor * background_noise
    
    plt.figure(figsize=(8, 8))
    plt.imshow(data, origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar()
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title(f'Image of {cluster_name} in J Filter')
    
    # Save the image in the specified directory
    output_file = os.path.join(plot_dir, f'{cluster_name}_J_image.png')
    plt.savefig(output_file)
    plt.close()

def plot_cluster_image_ks(hdus, cluster_name, plot_dir, contrast_factor=40):
    data = hdus[1].data
    # Calculate the background noise
    background_noise = np.std(data)
    # Ensure contrast_factor is a numeric value
    if not isinstance(contrast_factor, (int, float)):
        raise ValueError("contrast_factor must be a numeric value")
    
    # Set the contrast limits
    vmin = 100000  # or other suitable value for Ks filter
    vmax = contrast_factor * background_noise
    
    plt.figure(figsize=(8, 8))
    plt.imshow(data, origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar()
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title(f'Image of {cluster_name} in Ks Filter')
    
    # Save the image in the specified directory
    output_file = os.path.join(plot_dir, f'{cluster_name}_Ks_image.png')
    plt.savefig(output_file)
    plt.close()
    
    
# Function to convert core and tidal radii from parsecs to arcseconds
def convert_to_arcsec(radius_pc, distance_pc):
    return (radius_pc / distance_pc) * 206265

def calculate_radii_and_fov(params):
    """Calculate core radius, tidal radius in arcseconds and MICADO FOV."""
    core_radius_pc = params['r_c']
    tidal_radius_pc = params['r_t']
    distance_pc = params['cluster_distance']

    # Convert core and tidal radii to arcseconds
    core_radius_arcsec = convert_to_arcsec(core_radius_pc, distance_pc)
    tidal_radius_arcsec = convert_to_arcsec(tidal_radius_pc, distance_pc)

    # MICADO FOV in arcseconds
    micado_fov_arcsec = 53.4  # given value

    return core_radius_arcsec, tidal_radius_arcsec, micado_fov_arcsec

def plot_cluster_distribution(cluster_table, params, cluster_name, plot_dir):
    """Plot the star distribution in the cluster with core radius, tidal radius, and MICADO FOV."""
    
    # Calculate core radius, tidal radius, and MICADO FOV
    core_radius_arcsec, tidal_radius_arcsec, micado_fov_arcsec = calculate_radii_and_fov(params)
    
    # Extract star positions in arcseconds
    x_positions_arcsec = cluster_table['X Position (arcsec)']
    y_positions_arcsec = cluster_table['Y Position (arcsec)']
    
    # Set up the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(x_positions_arcsec, y_positions_arcsec, s=1, color='black', label='Stars')
    
    # Overlay circles for core and tidal radii
    core_circle = patches.Circle((0, 0), core_radius_arcsec, fill=False, color='yellow', linestyle='--', linewidth=2, label='Core Radius')
    tidal_circle = patches.Circle((0, 0), tidal_radius_arcsec, fill=False, color='blue', linestyle='--', linewidth=2, label='Tidal Radius')
    plt.gca().add_patch(core_circle)
    plt.gca().add_patch(tidal_circle)
    
    # Overlay the MICADO FOV (divided into 9 detectors)
    micado_fov_size = micado_fov_arcsec / 3
    offsets = [-micado_fov_size, 0, micado_fov_size]
    for dx in offsets:
        for dy in offsets:
            micado_rect = patches.Rectangle((dx - micado_fov_size/2, dy - micado_fov_size/2), micado_fov_size, micado_fov_size, fill=False, color='green', linestyle='-', linewidth=2, label='MICADO FOV' if dx == -micado_fov_size and dy == -micado_fov_size else "")
            plt.gca().add_patch(micado_rect)
    
    # Add labels and legend
    plt.xlabel('X Position (arcsec)')
    plt.ylabel('Y Position (arcsec)')
    #plt.title(f'Star Distribution in {cluster_name} with Core Radius, Tidal Radius, and MICADO FOV')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    # Save the plot to the specified directory
    output_file = os.path.join(plot_dir, f'{cluster_name}_distribution.png')
    plt.savefig(output_file)
    plt.close()
    plt.show()
    
def plot_milky_way_with_cluster(params, cluster_name, plot_dir):
    ra = params['ra'] * u.degree
    dec = params['dec'] * u.degree
    mw1 = MWSkyMap(projection="aitoff", grayscale=False, grid="galactic")
    #mw1.title = f'Position of {cluster_name} in the Milky Way'
    mw1.scatter(ra, dec, c="y", s=200)
    output_file = os.path.join(plot_dir, f'{cluster_name}_milky_way_position.png')
    plt.savefig(output_file)
    plt.close()
    
#Function to calculate SNR statistics
def calculate_snr_statistics(photometry_results, snr_column, key='spectral_type'):
    """Calculate the average SNR and standard deviation for each unique spectral type."""
    unique_keys = np.unique(photometry_results[key])
    avg_snr = []
    std_snr = []
    for k in unique_keys:
        snr_values = photometry_results[snr_column][photometry_results[key] == k]
        avg_snr.append(np.mean(snr_values))
        std_snr.append(np.std(snr_values))
    return unique_keys, avg_snr, std_snr


def calculate_snr_statistics(photometry_results, snr_column, key='spectral_type'):
    """Calculate the average SNR and standard deviation for each unique spectral type."""
    unique_keys = np.unique(photometry_results[key])
    avg_snr = []
    std_snr = []
    for k in unique_keys:
        snr_values = photometry_results[snr_column][photometry_results[key] == k]
        avg_snr.append(np.mean(snr_values))
        std_snr.append(np.std(snr_values))
    return unique_keys, avg_snr, std_snr

def plot_spectral_type_vs_snr(photometry_results_j, photometry_results_ks, cluster_name, plot_dir):
    spectral_type_order = [
        'O0V', 'O0.5V', 'O1V', 'O1.5V', 'O2V', 'O2.5V', 'O3V', 'O3.5V', 'O4V', 'O4.5V', 'O5V', 'O5.5V', 'O6V', 'O6.5V', 'O7V', 'O7.5V', 'O8V', 'O8.5V', 'O9V', 'O9.5V',
        'B0V', 'B0.5V', 'B1V', 'B1.5V', 'B2V', 'B2.5V', 'B3V', 'B3.5V', 'B4V', 'B4.5V', 'B5V', 'B5.5V', 'B6V', 'B6.5V', 'B7V', 'B7.5V', 'B8V', 'B8.5V', 'B9V', 'B9.5V',
        'A0V', 'A0.5V', 'A1V', 'A1.5V', 'A2V', 'A2.5V', 'A3V', 'A3.5V', 'A4V', 'A4.5V', 'A5V', 'A5.5V', 'A6V', 'A6.5V', 'A7V', 'A7.5V', 'A8V', 'A8.5V', 'A9V', 'A9.5V',
        'F0V', 'F0.5V', 'F1V', 'F1.5V', 'F2V', 'F2.5V', 'F3V', 'F3.5V', 'F4V', 'F4.5V', 'F5V', 'F5.5V', 'F6V', 'F6.5V', 'F7V', 'F7.5V', 'F8V', 'F8.5V', 'F9V', 'F9.5V',
        'G0V', 'G0.5V', 'G1V', 'G1.5V', 'G2V', 'G2.5V', 'G3V', 'G3.5V', 'G4V', 'G4.5V', 'G5V', 'G5.5V', 'G6V', 'G6.5V', 'G7V', 'G7.5V', 'G8V', 'G8.5V', 'G9V', 'G9.5V',
        'K0V', 'K0.5V', 'K1V', 'K1.5V', 'K2V', 'K2.5V', 'K3V', 'K3.5V', 'K4V', 'K4.5V', 'K5V', 'K5.5V', 'K6V', 'K6.5V', 'K7V', 'K7.5V', 'K8V', 'K8.5V', 'K9V', 'K9.5V',
        'M0V', 'M0.5V', 'M1V', 'M1.5V', 'M2V', 'M2.5V', 'M3V', 'M3.5V', 'M4V', 'M4.5V', 'M5V', 'M5.5V', 'M6V', 'M6.5V', 'M7V', 'M7.5V', 'M8V', 'M8.5V', 'M9V', 'M9.5V'
    ]

    # Calculate SNR statistics for both J and Ks filters
    spectral_types_j, avg_snr_j, std_snr_j = calculate_snr_statistics(photometry_results_j, snr_column='SNR J', key='spectral_type')
    spectral_types_ks, avg_snr_ks, std_snr_ks = calculate_snr_statistics(photometry_results_ks, snr_column='SNR Ks', key='spectral_type')

    # Sort spectral types
    sorted_indices_j = [spectral_type_order.index(st) for st in spectral_types_j]
    sorted_indices_ks = [spectral_type_order.index(st) for st in spectral_types_ks]

    # Apply the sorting to spectral types and SNR values
    spectral_types_j_sorted = [spectral_types_j[i] for i in np.argsort(sorted_indices_j)]
    avg_snr_j_sorted = [avg_snr_j[i] for i in np.argsort(sorted_indices_j)]
    std_snr_j_sorted = [std_snr_j[i] for i in np.argsort(sorted_indices_j)]

    spectral_types_ks_sorted = [spectral_types_ks[i] for i in np.argsort(sorted_indices_ks)]
    avg_snr_ks_sorted = [avg_snr_ks[i] for i in np.argsort(sorted_indices_ks)]
    std_snr_ks_sorted = [std_snr_ks[i] for i in np.argsort(sorted_indices_ks)]

    # Plot spectral type vs. average SNR for J and Ks filters with error bars
    plt.figure(figsize=(8, 8))
    plt.errorbar(spectral_types_j_sorted, avg_snr_j_sorted, yerr=std_snr_j_sorted, fmt='o-', color='b', label='J filter')
    plt.errorbar(spectral_types_ks_sorted, avg_snr_ks_sorted, yerr=std_snr_ks_sorted, fmt='o-', color='r', label='Ks filter')
    plt.axhline(y=5, color='g', linestyle='--', linewidth=3, label='SNR = 5')
    plt.axhline(y=10, color='r', linestyle='--', linewidth=3, label='SNR = 10')
    plt.yscale('log')  # Set y-axis to log scale
    plt.xlabel('Spectral Type')
    plt.ylabel('Average SNR')
    #plt.title(f'Spectral Type vs. Average SNR for J and Ks Filters in {cluster_name}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    output_file = os.path.join(plot_dir, f'{cluster_name}_spectral_type_vs_snr.png')
    plt.savefig(output_file)
    plt.close()



# Define a function to append flux and SNR values to the cluster table

TABLE_DIR = "Table/"  # Define this at the top of the file

# Ensure the directory exists
os.makedirs(TABLE_DIR, exist_ok=True)

def append_flux_snr_to_cluster_table(cluster_table, photometry_results_j, photometry_results_ks, cluster_name):

    # Convert cluster table coordinates to match the photometry results
    cluster_table['x'] = ((cluster_table['X Position (arcsec)'] / 0.004) + 2048).astype(int)
    cluster_table['y'] = ((cluster_table['Y Position (arcsec)'] / 0.004) + 2048).astype(int)

    # Filter cluster_table to include only stars within the MICADO FOV
    valid_indices = (cluster_table['x'] >= 0) & (cluster_table['x'] < 4096) & (cluster_table['y'] >= 0) & (cluster_table['y'] < 4096)
    filtered_cluster_table = cluster_table[valid_indices]

    # Rename columns in photometry results to avoid conflicts
    photometry_results_j.rename_column('flux', 'Flux J')
    photometry_results_j.rename_column('snr', 'SNR J')
    photometry_results_ks.rename_column('flux', 'Flux Ks')
    photometry_results_ks.rename_column('snr', 'SNR Ks')

    # Join the filtered cluster table with the photometry results
    filtered_cluster_table = join(filtered_cluster_table, photometry_results_j, keys=['x', 'y'], join_type='left', table_names=['', 'J'])
    filtered_cluster_table = join(filtered_cluster_table, photometry_results_ks, keys=['x', 'y'], join_type='left', table_names=['', 'Ks'])

    # Select only the necessary columns
    columns_to_keep = [
        'Mass', 'J Mag(abs)', 'J Mag(app)', 'Ks Mag(abs)', 'Ks Mag(app)', 
        'X Position (pc)', 'Y Position (pc)', 'X Position (arcsec)', 'Y Position (arcsec)', 
        'Spectral Type', 'Flux J', 'SNR J', 'Flux Ks', 'SNR Ks'
    ]
    filtered_cluster_table = filtered_cluster_table[columns_to_keep]

  # Save the table to a CSV file
    table_filename = os.path.join(TABLE_DIR, f'{cluster_name}_table.csv')
    filtered_cluster_table.write(table_filename, format='csv', overwrite=True)

    return filtered_cluster_table

def calculate_apparent_magnitudes(params):
    """
    Calculate apparent magnitudes for G2 and M9 stars in J and Ks filters given the cluster parameters.
    
    Args:
        params (dict): Dictionary containing cluster parameters including distance.
        
    Returns:
        dict: Apparent magnitudes for G2 and M9 stars in J and Ks filters.
    """
    distance_pc = params['cluster_distance']  # Distance in parsecs
    
    # Absolute magnitudes in J and Ks filters (provided values)
    absolute_magnitudes = {
        'G2_J': 3.6,     # Absolute magnitude for G2 star in J filter
        'G2_Ks': 3.236,  # Absolute magnitude for G2 star in Ks filter
        'M9_J': 11.59,   # Absolute magnitude for M9 star in J filter
        'M9_Ks': 10.4    # Absolute magnitude for M9 star in Ks filter
    }
    
    # Calculate apparent magnitudes using the distance modulus formula
    apparent_magnitudes = {}
    for star_type, abs_mag in absolute_magnitudes.items():
        apparent_magnitudes[star_type] = abs_mag + 5 * np.log10(distance_pc) - 5
    
    return apparent_magnitudes


cluster_template = r"""
% Title and basic information
\begin{center}
    \LARGE \textbf{Cluster: [[ cluster_name ]]} \\
    \vspace{0.5cm}
\end{center}

\begin{multicols}{2}
    % Left column for text
    \raggedright
    \small
    \begin{itemize}
        \item \textbf{Total Mass:} [[ total_mass ]] \(\textup{M}_\odot\)
        \item \textbf{Number of Stars:} [[ number_of_stars ]]
        \item \textbf{Distance:} [[ distance ]] pc
        \item \textbf{Core Radius:} [[ core_radius ]] pc
        \item \textbf{Tidal Radius:} [[ tidal_radius ]] pc
        \item \textbf{Log Age:} [[ age ]] Gyr
        \item \textbf{Galactic Longitude:} [[ galactic_longitude ]]\textdegree
        \item \textbf{Galactic Latitude:} [[ galactic_latitude ]]\textdegree
        \item \textbf{Core Density:} [[ star_density_core ]] stars/arcsec$^2$
        \item \textbf{Crowding Distance (MICADO):} [[ crowding_distance_micado ]] px
        \item \textbf{Crowding Distance (JWST):} [[ crowding_distance_jwst ]] px
        \item \textbf{Crowding Distance (HAWKI):} [[ crowding_distance_hawki ]] px
        \item \textbf{App. Mag G2 \& M9 (J):} [[ apparent_mag_g2_j ]] \& [[ apparent_mag_m9_j ]]
        \item \textbf{App. Mag G2 \& M9 (Ks):} [[ apparent_mag_g2_ks ]] \& [[ apparent_mag_m9_ks ]]
    \end{itemize}

    
    % Right column for images
    \begin{center}
        \includegraphics[width=1\linewidth]{[[ milky_way_position_image ]]} \\
        %\textbf{Cluster position in the Milky Way}
    \end{center}
    
\end{multicols}

\begin{multicols}{2}
    \begin{center}
        \includegraphics[width=1\linewidth]{[[ j_image ]]} \\
        %\textbf{Image in J Filter}
    \end{center}
    
    \begin{center}
        \includegraphics[width=1\linewidth]{[[ ks_image ]]} \\
        %\textbf{Image in Ks Filter}
    \end{center}
\end{multicols}

% Second row of images
\begin{multicols}{2}
    \begin{center}
        \includegraphics[width=0.9\linewidth]{[[ distribution_image ]]} \\
        %\textbf{Tidal radius, core radius and FoV}
    \end{center}

    \begin{center}
        \includegraphics[width=0.9\linewidth]{[[ spectral_type_vs_snr_image ]]} \\
        %\textbf{Spectral type vs SNR}
    \end{center}
    
\end{multicols}

\newpage
"""

main_template = r"""
\documentclass[a4paper, 12pt]{article}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{amsmath}
\usepackage{multicol}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{ragged2e}

\geometry{top=0.2in, bottom=0.2in, left=0.5in, right=0.5in}
\setlength{\columnsep}{10pt}

\begin{document}

[[ content ]]

\end{document}
"""

# Create Jinja2 templates
cluster_jinja_template = jinja2.Template(cluster_template, variable_start_string='[[', variable_end_string=']]')
main_jinja_template = jinja2.Template(main_template, variable_start_string='[[', variable_end_string=']]')

def render_cluster_latex(cluster_data):
    return cluster_jinja_template.render(cluster_data)

def render_main_latex(all_clusters_content):
    return main_jinja_template.render(content=all_clusters_content)



