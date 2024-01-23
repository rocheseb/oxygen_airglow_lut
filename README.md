# README

This code generates a lookup table and parametrization of oxygen airglow density profiles.

It relies on data from:

`Sun, Kang, 2022, "Level 2 data for SCIAMACHY airglow retrieval in 2010", https://doi.org/10.7910/DVN/T1WRWQ, Harvard Dataverse, V1`

### Install

`pip install git+https://github.com/rocheseb/oxygen_airglow_lut`

### Usage

Download and unzip `level2_nomal.tar.gz` at https://doi.org/10.7910/DVN/T1WRWQ into a `L2DIR` directory

Generate the lookup table with:

`airglowlut -i L2DIR`

### Contact

sroche@g.harvard.edu