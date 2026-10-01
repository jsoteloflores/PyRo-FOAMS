# Benchmark Results — Brief 03

## Environment

- Python: 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)]
- Platform: Windows-10-10.0.26200-SP0
- NumPy: 2.4.6
- OpenCV: 5.0.0
- Pillow: 12.3.0
- Matplotlib: 3.11.2

**Caveat:** tracemalloc reports Python-level allocations only; native OpenCV allocations (C-level) may not be captured fully.
Each case used one warm-up followed by 5 measured runs; times and peak observations are medians.

## Results (median of runs)

| Case | Baseline (s / peak) | Current (s / peak) | Observed ratio | Verification |
|---|---:|---:|---:|---|
| colorize_labels_512_100 | 0.033354 / 1.8MB | 0.012801 / 8.3MB | 2.61x | outputs verified before timing |
| measure_labels_512_100 | 0.160515 / 1.3MB | 0.038878 / 1.0MB | 4.13x | outputs verified before timing |
| colorize_labels_512_sparse | 0.005849 / 1.8MB | 0.012158 / 8.3MB | 0.48x | outputs verified before timing |
| measure_labels_512_sparse | 0.021640 / 1.3MB | 0.006405 / 1.0MB | 3.38x | outputs verified before timing |
| colorize_labels_2048_1000 | 5.884235 / 28.0MB | 0.277309 / 132.0MB | 21.22x | outputs verified before timing |
| measure_labels_2048_1000 | 23.201472 / 20.0MB | 0.458943 / 19.0MB | 50.55x | outputs verified before timing |
| sampling_labels_512_100 | 0.023351 / 1.0MB | 0.001758 / 1.0MB | 13.28x | outputs verified before timing |
| sampling_labels_512_sparse | 0.004606 / 1.0MB | 0.001644 / 1.0MB | 2.80x | outputs verified before timing |
| sampling_labels_2048_1000 | 5.667935 / 16.0MB | 0.026730 / 16.0MB | 212.04x | outputs verified before timing |
| distribution_n20_bins_50 | 0.055044 / 171.5KB | 0.042999 / 171.7KB | 1.28x | outputs verified before timing |
| distribution_n20_bins_500 | 0.088108 / 245.2KB | 0.052702 / 245.4KB | 1.67x | outputs verified before timing |
| distribution_n100_bins_50 | 0.316607 / 982.5KB | 0.255771 / 982.7KB | 1.24x | outputs verified before timing |
| distribution_n100_bins_500 | 0.452534 / 1.0MB | 0.285747 / 1.0MB | 1.58x | outputs verified before timing |
| processing_same_core_512 | 0.003208 / 1.3MB | 0.003170 / 1.3MB | 1.01x | outputs verified before timing |
| processing_same_core_2048 | 0.055748 / 20.0MB | 0.054431 / 20.0MB | 1.02x | outputs verified before timing |
| rendering_100_events | 0.003840 / 1.0KB | 0.000644 / 272.0B | 5.96x | outputs verified before timing |