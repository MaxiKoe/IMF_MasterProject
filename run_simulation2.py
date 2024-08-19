import json
import jinja2
import os
import subprocess
import shutil
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
    calculate_apparent_magnitudes
)
from astropy.table import Table
from astropy.io import fits, ascii
import numpy as np

# Load the OpenCluster table
open_cluster_filename = 'OpenClusters_final.fits'
open_cluster_table = Table.read(open_cluster_filename)

# List of cluster names to process
cluster_names = open_cluster_table['NAME']  # Assuming 'NAME' is the column with cluster names

# Set the batch size for processing clusters
batch_size = 50

# LaTeX template for individual clusters
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
    \end{center>
    
    <begin{center}
        \includegraphics[width=1\linewidth]{[[ ks_image ]]} \\
        %\textbf{Image in Ks Filter}
    \end{center}
</begin{multicols}

% Second row of images
<begin{multicols}{2}
    \begin{center>
        \includegraphics[width=0.9\linewidth]{[[ distribution_image ]]} \\
        %\textbf{Tidal radius, core radius and FoV}
    \end{center}

    \begin{center}
        \includegraphics[width=0.9\linewidth]{[[ spectral_type_vs_snr_image ]]} \\
        %\textbf{Spectral type vs SNR}
    \end{center>
    
</begin{multicols>

\newpage
"""

# Main LaTeX document structure
main_template = r"""
\documentclass[a4paper, 12pt]{article}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{amsmath}
\usepackage{multicol}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{ragged2e>

\geometry{top=0.2in, bottom=0.2in, left=0.5in, right=0.5in}
\setlength{\columnsep}{10pt}

\begin{document>

[[ content ]]

\end{document>
"""

# Create Jinja2 templates with custom delimiters
cluster_jinja_template = jinja2.Template(cluster_template, variable_start_string='[[', variable_end_string=']]')
main_jinja_template = jinja2.Template(main_template, variable_start_string='[[', variable_end_string=']]')

def process_cluster_batch(batch_names):
    all_clusters_content = ""  # Reset LaTeX content for each batch
    
    for cluster_name in batch_names:
        # Retrieve the parameters for the specified cluster
        params = get_cluster_params(cluster_name, open_cluster_table)

        # Generate the cluster table using the retrieved parameters
        cluster_table = create_cluster_table(params['total_cluster_mass'], params['cluster_distance'], 
                                             'kroupa', params['r_c'], params['r_t'], 
                                             params['sigma_0'], params['log_age'])

        # Simulate observation with MICADO using the J filter
        hdus_j, micado_j = simulate_micado(cluster_table, filter_name='J')

        # Save the simulated image to a FITS file
        output_filename_j = f'{cluster_name}_J_image.fits'
        hdul_j = fits.HDUList(hdus_j)
        hdul_j.writeto(output_filename_j, overwrite=True)

        # Perform photometry on the simulated data for J filter
        photometry_results_j = perform_photometry(hdus_j, micado_j)

        # Simulate observation with MICADO using the Ks filter
        hdus_ks, micado_ks = simulate_micado(cluster_table, filter_name='Ks')

        # Save the simulated image to a FITS file
        output_filename_ks = f'{cluster_name}_Ks_image.fits'
        hdul_ks = fits.HDUList(hdus_ks)
        hdul_ks.writeto(output_filename_ks, overwrite=True)

        # Perform photometry on the simulated data for Ks filter
        photometry_results_ks = perform_photometry(hdus_ks, micado_ks)

        # Append flux and SNR values to the cluster table
        cluster_table_with_flux_snr = append_flux_snr_to_cluster_table(cluster_table, photometry_results_j, photometry_results_ks)

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
            'milky_way_position_image': f'{cluster_name}_milky_way_position.png',
            'j_image': f'{cluster_name}_J_image.png',
            'ks_image': f'{cluster_name}_Ks_image.png',
            'distribution_image': f'{cluster_name}_distribution.png',
            'spectral_type_vs_snr_image': f'{cluster_name}_spectral_type_vs_snr.png',
        }

        with open(f'{cluster_name}_data.json', 'w') as f:
            json.dump(cluster_data, f)

        # Generate the plots and save as images
        plot_cluster_image_j(hdus_j, cluster_name)
        plot_cluster_image_ks(hdus_ks, cluster_name)
        plot_cluster_distribution(cluster_table, params, cluster_name)
        plot_milky_way_with_cluster(params, cluster_name)
        plot_spectral_type_vs_snr(photometry_results_j, photometry_results_ks, cluster_name)

        # Render the cluster template with the cluster data
        rendered_cluster_latex = cluster_jinja_template.render(cluster_data)

        # Accumulate the rendered LaTeX content
        all_clusters_content += rendered_cluster_latex

        # Create a directory for this cluster's output files
        cluster_dir = f'{cluster_name}_output'
        os.makedirs(cluster_dir, exist_ok=True)

        # Move the generated files to the cluster directory
        shutil.move(f'{cluster_name}_J_image.fits', cluster_dir)
        shutil.move(f'{cluster_name}_Ks_image.fits', cluster_dir)
        shutil.move(f'{cluster_name}_data.json', cluster_dir)
        shutil.move(f'{cluster_name}_milky_way_position.png', cluster_dir)
        shutil.move(f'{cluster_name}_distribution.png', cluster_dir)
        shutil.move(f'{cluster_name}_spectral_type_vs_snr.png', cluster_dir)

    # Save and compile the LaTeX content for the batch
    rendered_main_latex = main_jinja_template.render(content=all_clusters_content)
    batch_index = open_cluster_table['NAME'].tolist().index(batch_names[0])
    output_tex_file = f'all_clusters_batch_{batch_index // batch_size + 1}.tex'
    
    with open(output_tex_file, 'w') as f:
        f.write(rendered_main_latex)

    # Compile the LaTeX file to PDF
    process = subprocess.Popen(['pdflatex', output_tex_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    try:
        print(stdout.decode('utf-8'))
    except UnicodeDecodeError:
        print(stdout.decode('latin-1'))

    try:
        print(stderr.decode('utf-8'))
    except UnicodeDecodeError:
        print(stderr.decode('latin-1'))

    # Move the compiled LaTeX and PDF files to the cluster directory
    shutil.move(output_tex_file, cluster_dir)
    shutil.move(output_tex_file.replace('.tex', '.pdf'), cluster_dir)

# Divide clusters into batches
batches = [cluster_names[i:i + batch_size] for i in range(0, len(cluster_names), batch_size)]

# Create a pool of workers equal to the number of available CPU cores
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(process_cluster_batch, batches)

# Finally, merge all the batch PDFs into a single PDF
subprocess.call(['pdfunite'] + [f'all_clusters_batch_{i}.pdf' for i in range(1, len(batches) + 1)] + ['all_clusters.pdf'])

print("All clusters processed and merged into all_clusters.pdf.")
