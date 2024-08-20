import json
import os
import subprocess
import multiprocessing
from cluster_generating import (
    create_cluster_table, 
    get_cluster_params, 
    simulate_micado, 
    perform_photometry, 
    plot_cluster_image_j, 
    plot_cluster_image_ks,
    plot_cluster_distribution, 
    plot_milky_way_with_cluster, 
    print_snr_percentages,
    plot_spectral_type_vs_snr,
    append_flux_snr_to_cluster_table,
    calculate_apparent_magnitudes,
    render_cluster_latex,
    render_main_latex
)
from astropy.table import Table
from astropy.io import fits, ascii
import numpy as np
import logging

# Set the logging level to ERROR to suppress warnings
logging.getLogger('astar').setLevel(logging.ERROR)
logging.getLogger('sim').setLevel(logging.ERROR)


# Define directories relative to the current working directory (IMF_MasterProject)
IMAGE_DIR = "Images/"
PLOT_DIR = "Plots/"
OTHER_DIR = "Other/"
PDF_DIR = "PDF/"
TABLE_DIR = "Table/"

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OTHER_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# Load the OpenCluster table
open_cluster_filename = 'OpenClusters_final.fits'
open_cluster_table = Table.read(open_cluster_filename)

# List of cluster names to process (for test with 10 clusters)
cluster_names = open_cluster_table['NAME'][:1]  # First 10 clusters

def process_cluster(cluster_name):
    global IMAGE_DIR, PLOT_DIR, OTHER_DIR
    
    # Retrieve the parameters for the specified cluster
    params = get_cluster_params(cluster_name, open_cluster_table)

    # Generate the cluster table using the retrieved parameters
    cluster_table = create_cluster_table(params['total_cluster_mass'], params['cluster_distance'], 
                                         'kroupa', params['r_c'], params['r_t'], 
                                         params['sigma_0'], params['log_age'])

    # Simulate observation with MICADO using the J filter
    hdus_j, micado_j = simulate_micado(cluster_table, filter_name='J')

    # Save the simulated image to a FITS file
    output_filename_j = os.path.join(IMAGE_DIR, f'{cluster_name}_J_image.fits')
    hdul_j = fits.HDUList(hdus_j)
    hdul_j.writeto(output_filename_j, overwrite=True)

    # Perform photometry on the simulated data for J filter
    photometry_results_j = perform_photometry(hdus_j, micado_j)

    # Simulate observation with MICADO using the Ks filter
    hdus_ks, micado_ks = simulate_micado(cluster_table, filter_name='Ks')

    # Save the simulated image to a FITS file
    output_filename_ks = os.path.join(IMAGE_DIR, f'{cluster_name}_Ks_image.fits')
    hdul_ks = fits.HDUList(hdus_ks)
    hdul_ks.writeto(output_filename_ks, overwrite=True)

    # Perform photometry on the simulated data for Ks filter
    photometry_results_ks = perform_photometry(hdus_ks, micado_ks)

    # Append flux and SNR values to the cluster table
    cluster_table_with_flux_snr = append_flux_snr_to_cluster_table(cluster_table, photometry_results_j, photometry_results_ks, cluster_name)

    # Convert all values in params to standard Python types
    params = {k: float(v) if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) else v for k, v in params.items()}
    
    # Ensure gal_lon, gal_lat, total_cluster_mass, log_age, star_density_core, and pixels_per_star_micado are correctly assigned
    gal_lon = params.get('gal_lon', 'N/A')
    gal_lat = params.get('gal_lat', 'N/A')
    total_cluster_mass = params.get('total_cluster_mass', 'N/A')
    log_age = params.get('log_age', 'N/A')
    star_density_core = params.get('star_density_core', 'N/A')
    pixels_per_star_micado = params.get('pixels_per_star_micado', 'N/A')
    pixels_per_star_jwst = params.get('pixels_per_star_jwst', 'N/A')
    pixels_per_star_hawki = params.get('pixels_per_star_hawki', 'N/A')

    # Format values to 3 decimal places
    gal_lon = f"{gal_lon:.3f}"
    gal_lat = f"{gal_lat:.3f}"
    total_cluster_mass = f"{total_cluster_mass:.1f}"
    log_age = f"{log_age:.3f}"
    star_density_core = f"{star_density_core:.3f}"
    crowding_distance_micado = f"{np.sqrt(pixels_per_star_micado):.1f}"
    crowding_distance_jwst = f"{np.sqrt(pixels_per_star_jwst):.1f}"
    crowding_distance_hawki = f"{np.sqrt(pixels_per_star_hawki):.1f}"
    
    # Calculate apparent magnitudes for G2 and M9 stars in J and Ks filters
    apparent_magnitudes = calculate_apparent_magnitudes(params)

    # Save cluster parameters to JSON for LaTeX rendering
    json_filename = os.path.join(OTHER_DIR, f'{cluster_name}_data.json')
    cluster_data = {
        'cluster_name': cluster_name,
        'total_mass': total_cluster_mass,
        'number_of_stars': len(cluster_table),
        'distance': params['cluster_distance'],
        'core_radius': params['r_c'],
        'tidal_radius': params['r_t'],
        'age': log_age,
        'galactic_longitude': gal_lon,
        'galactic_latitude': gal_lat,
        'star_density_core': star_density_core,
        'crowding_distance_micado': crowding_distance_micado,
        'crowding_distance_jwst': crowding_distance_jwst,
        'crowding_distance_hawki': crowding_distance_hawki,
        'apparent_mag_g2_j': f"{apparent_magnitudes['G2_J']:.2f}",
        'apparent_mag_g2_ks': f"{apparent_magnitudes['G2_Ks']:.2f}",
        'apparent_mag_m9_j': f"{apparent_magnitudes['M9_J']:.2f}",
        'apparent_mag_m9_ks': f"{apparent_magnitudes['M9_Ks']:.2f}",
        'milky_way_position_image': os.path.join(PLOT_DIR, f'{cluster_name}_milky_way_position.png'),
        'j_image': os.path.join(PLOT_DIR, f'{cluster_name}_J_image.png'),
        'ks_image': os.path.join(PLOT_DIR, f'{cluster_name}_Ks_image.png'),
        'distribution_image': os.path.join(PLOT_DIR, f'{cluster_name}_distribution.png'),
        'spectral_type_vs_snr_image': os.path.join(PLOT_DIR, f'{cluster_name}_spectral_type_vs_snr.png'),
    }

    with open(json_filename, 'w') as f:
        json.dump(cluster_data, f)

    # Generate the plots and save as images
    plot_cluster_image_j(hdus_j, cluster_name, PLOT_DIR)
    plot_cluster_image_ks(hdus_ks, cluster_name, PLOT_DIR)
    plot_cluster_distribution(cluster_table, params, cluster_name, PLOT_DIR)
    plot_milky_way_with_cluster(params, cluster_name, PLOT_DIR)
    plot_spectral_type_vs_snr(photometry_results_j, photometry_results_ks, cluster_name, PLOT_DIR)

    # Render the cluster template with the cluster data
    rendered_cluster_latex = render_cluster_latex(cluster_data)

    return rendered_cluster_latex

def run_simulation():
    global cluster_names
    from multiprocessing import Pool

    # Create a pool of workers
    with Pool() as pool:
        results = [pool.apply_async(process_cluster, args=(cluster_name,)) for cluster_name in cluster_names]

        # Combine the LaTeX content for all clusters
        all_clusters_content = "".join([res.get() for res in results])

    # Render the main LaTeX template with all clusters content
    rendered_main_latex = render_main_latex(all_clusters_content)

    # Save the rendered LaTeX to a file
    output_tex_file = os.path.join(PDF_DIR, 'one_cluster.tex')
    with open(output_tex_file, 'w') as f:
        f.write(rendered_main_latex)

    print(f"Rendered LaTeX saved to {output_tex_file}")

    # Compile the LaTeX file to PDF and capture the output
    output_pdf_file = os.path.join(PDF_DIR, 'test_cluster.pdf')
    process = subprocess.Popen(['pdflatex', '-output-directory', PDF_DIR, output_tex_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decode with 'latin-1' and fallback to 'ignore' in case of issues
    try:
        print(stdout.decode('utf-8'))
    except UnicodeDecodeError:
        print(stdout.decode('latin-1'))

    try:
        print(stderr.decode('utf-8'))
    except UnicodeDecodeError:
        print(stderr.decode('latin-1'))

if __name__ == "__main__":
    run_simulation()

