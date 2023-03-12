This folder contains the scripts used for the Experiments - Splatter section of the paper.

If you would like to reproduce our data and analysis, you should to the following:

# Procedure
1. Execute "1--simulation.r" This carries out the simulation in the Splatter framework.
2. Run "2--find_variable_and_plot.ipynb". Most importantly, this script filters the highly variable genes and puts them into a csv, which is read in in the next script.
	If you wish, you can also create a t-SNE plot here
3. Run "3--compute_pairwise_distances.py" This actually loops over the simulated groups of cells and computes the pairwise distances. It requires that the file "genes_splatter.csv" (created in step 2) lies in the same directory. 

# Data files
For convenience, our simulated data has been left in the Github repository so that you may work with these data should the Splatter installation not work right away.
These include
#
- "counts_splatter.csv" - the counts of each simulated cell, created by the splatter simulation in the 1--simulation.r script
- "types_splatter.csv" - the label of each simulated cell, created by the splatter simulation in the 1--simulation.r script
- "genes_splatter.csv" - the filtered count data, created by 2--find_variable_and_plot.ipynb To be read in the final 3--compute_pairwise_distances.py script

# Installation requirements

## Splatter
Splatter is a R-package used for creating mock single-cell RNA sequencing count data. See also: https://bioconductor.org/packages/release/bioc/html/splatter.html

To install, you need to:
#
1. Install R.
2. Enter a R console and enter 

	    if (!require("BiocManager", quietly = TRUE))

	        install.packages("BiocManager")

	    BiocManager::install("splatter")

I had the problem that my R version was not the newest one, so I had to install an older version of the BiocManager, which was appropriate to my version of R. See also https://bioconductor.org/about/release-announcements/ for a full list of old versions.

## Python
There are no more library requirements other than the generally stated ones. If you wish to create the t-SNE plot, you need to have the specific packages installed, though this step is optional.
