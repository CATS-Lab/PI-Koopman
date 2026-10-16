# OpenACC

Source: [European Commission Joint Research Centre](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/).
License: CC BY 4.0. Copyright European Union; see the
[AstaZero](AstaZero_copyright.txt) and [Casale](Casale_copyright.txt) notices.

Cite [Makridis et al. (2021)](https://doi.org/10.1016/j.trc.2021.103047)
and the [JRC database](https://doi.org/10.2905/JRC.KMH3D00).

The experiments use 284 non-overlapping 30-s episodes at 10 Hz:
240 from AstaZero and 44 from Casale. Each episode has positive gap,
finite nonnegative speeds, and a constant ACC/Human follower label.
Up to six episodes per source-file/follower pair are selected at evenly spaced
indices. Gap uses the released IVS field; evaluation holds out entire source files.

`episode_manifest.csv` records the selected windows. OpenACC inputs in
`data/timing/` are relative-state excerpts derived from these windows.
