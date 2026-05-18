Dataset Source

Dataset Name: MARIne Debris Archive (MARIDA)

Dataset Link: https://github.com/earthlab/MARIDA (Primary repository)
Dataset Owner/Contact: Earth Observation Laboratory, University of the Aegean; National Technical University of Athens (K. Topouzelis).

Dataset Characteristics
Number of Observations: 837,357 pixels (Total samples across all splits). The data is derived from Sentinel-2 Multi-Spectral Instrument (MSI) Level-2A imagery.

Number of Features: 11 (Corresponding to specific Sentinel-2 spectral bands).

Label Type: Multi-class Classification

Label Description: Categorization of pixels based on the physical material or environmental condition present on the marine surface. The primary task is the identification of marine plastic debris amidst natural occurrences and sensor noise.

Label Values:

Marine Debris: Anthropogenic floating plastic and mixed debris.

Dense Sargassum: High-density macroalgae.

Sparse Sargassum: Low-density macroalgae.

Natural Foam: Sea foam or organic suds.

Waves: Sun glint or breaking waves.

Oil Spill: Intentional or accidental oil discharge.

Turbid Water: Water with high sediment concentration.

Shallow Water: Areas where the seabed is visible through the water column.

Clear Water: Open, deep marine water.

Cloud Shadows: Pixels obscured by shadows from overlying clouds.

Clouds: Atmospheric cloud cover.

Label Distribution: Highly imbalanced. The "Marine Debris" class (Class 1) represents a small minority of the dataset (approximately 0.37%), whereas "Clear Water" and "Clouds" represent the vast majority of the samples.

Feature Description
Spectral Band Group (Sentinel-2 MSI Bands): These features represent the surface reflectance values extracted from the Sentinel-2 Level-2A products. Each feature is a numerical value (float32) representing light reflectance at specific wavelengths.

B1 (Coastal Aerosol): Wavelength ~443 nm. Used for aerosol detection and water quality.

B2 (Blue): Wavelength ~490 nm. Visible blue spectrum.

B3 (Green): Wavelength ~560 nm. Visible green spectrum.

B4 (Red): Wavelength ~665 nm. Visible red spectrum.

B5 (Vegetation Red Edge): Wavelength ~705 nm. Crucial for identifying chlorophyll/organic matter.

B6 (Vegetation Red Edge): Wavelength ~740 nm. Secondary red-edge band.

B7 (Vegetation Red Edge): Wavelength ~783 nm. Tertiary red-edge band.

B8 (NIR - Near Infrared): Wavelength ~842 nm. Essential for the Floating Debris Index (FDI).

B8A (Narrow NIR): Wavelength ~865 nm. High-precision near-infrared data.

B11 (SWIR 1): Wavelength ~1610 nm. Short-wave infrared; vital for distinguishing polymers from water.

B12 (SWIR 2): Wavelength ~2190 nm. Secondary short-wave infrared band; highlights chemical absorption features.
