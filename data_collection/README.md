# Step 1: Data Collection

Builds the composite 5-class dataset for the astronomical object classifier.

## Class plan

| Class | Source | Target count |
|---|---|---|
| Spiral Galaxy | Galaxy10 DECaLS (classes 5, 6, 7) | ~1500 |
| Elliptical Galaxy | Galaxy10 DECaLS (classes 2, 3, 4) | ~1500 |
| Nebula | ESA/Hubble image archive | ~700 |
| Star Cluster | ESA/Hubble image archive | ~500 |
| Planetary Object | NASA Image and Video Library (solar system planets) | ~800 |

The galaxy classes are deliberately downsampled — Galaxy10 has thousands per
sub-class and we want to keep the dataset balanced against the scraped classes.

## Setup

Python 3.9+ recommended.

```bash
pip install -r requirements.txt
```

`astroNN` pulls in TensorFlow/Keras as a transitive dependency. If you don't
want that, you can manually download the Galaxy10 H5 file from
https://github.com/henrysky/Galaxy10 and load it with `h5py` instead (edit
`01_download_galaxy10.py` accordingly).

## Run the scripts in order

```bash
python 01_download_galaxy10.py          # ~2.5 GB download on first run, then ~5 min
python 02_scrape_hubble.py              # ~30-45 min, rate-limited
python 03_fetch_nasa_planets.py         # ~10-15 min
python 04_inspect_data.py               # generates sample grid and counts
```

All scripts are idempotent — safe to rerun if interrupted. They skip files that
already exist locally.

## Common adjustments

```bash
# more spiral/elliptical per class
python 01_download_galaxy10.py --per_class 2000

# more aggressive Hubble scrape
python 02_scrape_hubble.py --max 1200 --delay 1.0

# only fetch Mars and Jupiter (edit PLANETS dict in 03_fetch_nasa_planets.py)
```

## Output structure

```
data/
└── processed/
    ├── spiral_galaxy/        spiral_galaxy_00000.jpg ...
    ├── elliptical_galaxy/    elliptical_galaxy_00000.jpg ...
    ├── nebula/               nebula_heic1234a.jpg ...
    ├── star_cluster/         star_cluster_potw2021a.jpg ...
    └── planetary_object/     jupiter_PIA12345.jpg ...
data/sample_grid.png          visual QA grid from 04_inspect_data.py
```

## After running — REQUIRED manual QA

Open `data/sample_grid.png`. For each class, the 5 thumbnails should clearly
match the class label. Two known noise sources:

1. **NASA planetary results** include ~10-25% non-planet content
   (mission patches, scientist photos, infographics, artist concepts).
   Skim the `planetary_object/` folder and delete anything that isn't a
   recognizable telescope/spacecraft image of a planet.

2. **ESA/Hubble nebula vs star cluster** is editorially clean, but a handful of
   images appear in both categories (e.g. open clusters embedded in nebulae).
   This is fine — they're genuinely ambiguous and your model should learn that.

If a class ends up with < 300 images, top it up:
- Nebula / Star Cluster: rerun the Hubble scraper with a higher `--max` or
  supplement from NASA Images API with `q=nebula`, `q=globular cluster`, etc.
- Planetary Object: rerun with a higher `--per_planet` count.

## Licensing / attribution

- **Galaxy10 DECaLS**: redistributed by Henry Leung (henrysky/Galaxy10).
  Original credit: Galaxy Zoo (Lintott et al.) and DESI Legacy Imaging Surveys.
- **ESA/Hubble images**: public domain under
  https://esahubble.org/copyright/. Credit "NASA, ESA, and the Hubble Heritage Team"
  in any publication.
- **NASA Image and Video Library**: public domain unless otherwise marked.
  https://www.nasa.gov/multimedia/guidelines/index.html
