import json
import os
import subprocess
from cluster_generating import (
    get_cluster_params, 
    plot_cluster_image_j, 
    plot_cluster_image_ks,
    plot_cluster_distribution, 
    plot_milky_way_with_cluster, 
    plot_spectral_type_vs_snr,
    render_cluster_latex,
    render_main_latex,
    calculate_apparent_magnitudes
)
from astropy.table import Table
from astropy.io import fits
import numpy as np

# Define directories relative to the current working directory
BASE_DIR = os.path.abspath(os.getcwd())
IMAGE_DIR = os.path.join(BASE_DIR, "Images")
PLOT_DIR = os.path.join(BASE_DIR, "Plots")
OTHER_DIR = os.path.join(BASE_DIR, "Other")
PDF_DIR = os.path.join(BASE_DIR, "PDF")
TABLE_DIR = os.path.join(BASE_DIR, "Table")

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OTHER_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# Load the OpenCluster table (sorted by crowding distance)
open_cluster_filename = os.path.join(BASE_DIR, 'OpenClusters_final.fits')
open_cluster_table = Table.read(open_cluster_filename)

# List of cluster names to process (all clusters, sorted by MICADO crowding distance)
cluster_names = open_cluster_table['NAME'][:500]  

# Total number of clusters
total_clusters = len(cluster_names)

def process_cluster(cluster_name, index):
    global IMAGE_DIR, PLOT_DIR, OTHER_DIR, total_clusters
    
    # Print progress
    print(f"Processing cluster {index + 1}/{total_clusters}: {cluster_name}")

    # Retrieve the parameters for the specified cluster
    params = get_cluster_params(cluster_name, open_cluster_table)

    # Load the already generated FITS files for J and Ks images
    fits_j_path = os.path.join(IMAGE_DIR, f'{cluster_name}_J_image.fits')
    fits_ks_path = os.path.join(IMAGE_DIR, f'{cluster_name}_Ks_image.fits')
    
    if not os.path.exists(fits_j_path) or not os.path.exists(fits_ks_path):
        print(f"FITS files missing for cluster {cluster_name}, skipping...")
        return ""

    hdus_j = fits.open(fits_j_path)
    hdus_ks = fits.open(fits_ks_path)

    # Calculate crowding distances
    crowding_distance_micado = np.sqrt(params.get('pixels_per_star_micado', np.nan))
    crowding_distance_jwst = np.sqrt(params.get('pixels_per_star_jwst', np.nan))
    crowding_distance_hawki = np.sqrt(params.get('pixels_per_star_hawki', np.nan))
    
    # Calculate core and tidal radii in arcseconds
    core_radius_arcsec = (params['r_c'] / params['cluster_distance']) * 206265
    tidal_radius_arcsec = (params['r_t'] / params['cluster_distance']) * 206265
    
    # Format the age in a more readable way
    age_in_years = 10**params['log_age']
    age_formatted = f"{age_in_years / 1e9:.2f} Gyr" if age_in_years >= 1e9 else f"{age_in_years / 1e6:.2f} Myr"

    # Calculate apparent magnitudes for G2 and M9 stars in J and Ks filters
    apparent_magnitudes = calculate_apparent_magnitudes(params)
    
    # Determine suggested instrument
    if crowding_distance_hawki > 10:
        recommended_instrument = "HAWK-I"
    elif crowding_distance_jwst > 6:
        recommended_instrument = "JWST or MICADO"
    elif crowding_distance_micado > 8:
        recommended_instrument = "MICADO"
    else:
        recommended_instrument = "None"
    
    # Generate the plots again with arcseconds on the axes
    plot_cluster_image_j(hdus_j, os.path.join(IMAGE_DIR, f'{cluster_name}_J_image.png'))
    plot_cluster_image_ks(hdus_ks, os.path.join(IMAGE_DIR, f'{cluster_name}_Ks_image.png'))
    distribution_image = os.path.join(PLOT_DIR, f'{cluster_name}_distribution.png')
    spectral_type_vs_snr_image = os.path.join(PLOT_DIR, f'{cluster_name}_spectral_type_vs_snr.png')
    milky_way_image = os.path.join(PLOT_DIR, f'{cluster_name}_milky_way_position.png')
    
    # Prepare LaTeX content
    cluster_data = {
        'cluster_name': cluster_name,
        'total_mass': f"{params['total_cluster_mass']:.1f}",
        'number_of_stars': params['num_stars_tidal'],  # Adjust as needed
        'distance': params['cluster_distance'],
        'core_radius': params['r_c'],
        'tidal_radius': params['r_t'],
        'core_radius_arcsec': f"{core_radius_arcsec:.2f}",
        'tidal_radius_arcsec': f"{tidal_radius_arcsec:.2f}",
        'age_years': age_formatted,
        'galactic_longitude': f"{params['gal_lon']:.3f}",
        'galactic_latitude': f"{params['gal_lat']:.3f}",
        'star_density_core': f"{params['star_density_core']:.3f}",
        'crowding_distance_micado': f"{crowding_distance_micado:.1f}",
        'crowding_distance_jwst': f"{crowding_distance_jwst:.1f}",
        'crowding_distance_hawki': f"{crowding_distance_hawki:.1f}",
        'apparent_mag_g2_j': f"{apparent_magnitudes['G2_J']:.2f}",
        'apparent_mag_g2_ks': f"{apparent_magnitudes['G2_Ks']:.2f}",
        'apparent_mag_m9_j': f"{apparent_magnitudes['M9_J']:.2f}",
        'apparent_mag_m9_ks': f"{apparent_magnitudes['M9_Ks']:.2f}",
        'recommended_instrument': recommended_instrument,
        'milky_way_position_image': os.path.join(PLOT_DIR, f'{cluster_name}_milky_way_position.png'),
        'j_image': os.path.join(PLOT_DIR, f'{cluster_name}_J_image.png'),
        'ks_image': os.path.join(PLOT_DIR, f'{cluster_name}_Ks_image.png'),
        'distribution_image': os.path.join(PLOT_DIR, f'{cluster_name}_distribution.png'),
        'spectral_type_vs_snr_image': os.path.join(PLOT_DIR, f'{cluster_name}_spectral_type_vs_snr.png'),
    }

    return render_cluster_latex(cluster_data)

def run_simulation():
    global cluster_names, total_clusters
    from multiprocessing import Pool

    print("Starting run_simulation()")  # Debugging print

    with Pool(32) as pool:
        print(f"Total clusters: {total_clusters}")  # Debugging print

        # Apply the process_cluster function to all clusters in parallel
        results = [pool.apply_async(process_cluster, args=(cluster_name, index)) for index, cluster_name in enumerate(cluster_names)]

        print("Waiting for results...")  # Debugging print
        all_clusters_content = "".join([res.get() for res in results])

    # Render the main LaTeX template with all clusters content
    rendered_main_latex = render_main_latex(all_clusters_content)

    # Save the rendered LaTeX to a file
    output_tex_file = os.path.join(OTHER_DIR, '1-500_Clusters.tex')
    with open(output_tex_file, 'w') as f:
        f.write(rendered_main_latex)

    print(f"Rendered LaTeX saved to {output_tex_file}")

    # Compile the LaTeX file to PDF and capture the output
    output_pdf_file = os.path.join(PDF_DIR, '1-500_Clusters.pdf')
    process = subprocess.Popen(['pdflatex', '-output-directory', PDF_DIR, output_tex_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    try:
        print(stdout.decode('utf-8'))
    except UnicodeDecodeError:
        print(stdout.decode('latin-1'))

    try:
        print(stderr.decode('utf-8'))
    except UnicodeDecodeError:
        print(stderr.decode('latin-1'))


if __name__ == "__main__":
    print("Starting the simulation...")
    run_simulation()
