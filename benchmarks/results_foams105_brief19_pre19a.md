# Brief 19 FOAMS Workflow Benchmarks

- Python: `3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)]`
- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD`
- NumPy/OpenCV: `2.2.6` / `4.12.0`
- Iterations: 1 warm-up, 5 measured runs
- Memory: sampled process working set at 5 ms intervals; short peaks may be missed.
- Workloads are synthetic, non-overlapping circular pore masks, not the original MATLAB catalog images.

| Workload | Operation | Median seconds | Range seconds | Median sampled RSS delta | Pixels | Components | Selected rows | Export bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512x512_100components | fresh_prepare | 0.027581 | 0.026435-0.028779 | 5.93 MiB | 262144 | 100 | 100 | 122299 |
| 512x512_100components | complete_analysis_prepare_plus_reanalysis | 0.070650 | 0.068919-0.075491 | 5.68 MiB | 262144 | 100 | 100 | 122299 |
| 512x512_100components | cached_cutoff_normalization_shape_reanalysis | 0.039097 | 0.037085-0.047577 | 0.16 MiB | 262144 | 100 | 100 | 122299 |
| 512x512_100components | export_projection_and_write | 0.021051 | 0.020252-0.022077 | 0.04 MiB | 262144 | 100 | 100 | 122299 |
| 1024x1024_500components | fresh_prepare | 0.192258 | 0.185810-0.209871 | 25.84 MiB | 1048576 | 500 | 500 | 477914 |
| 1024x1024_500components | complete_analysis_prepare_plus_reanalysis | 0.468438 | 0.426114-0.473207 | 25.37 MiB | 1048576 | 500 | 500 | 477914 |
| 1024x1024_500components | cached_cutoff_normalization_shape_reanalysis | 0.282050 | 0.234012-0.317873 | 2.46 MiB | 1048576 | 500 | 500 | 477914 |
| 1024x1024_500components | export_projection_and_write | 0.062195 | 0.061777-0.065262 | 0.12 MiB | 1048576 | 500 | 500 | 477914 |
| 2048x2048_1000components | fresh_prepare | 1.242338 | 1.217438-1.282377 | 193.49 MiB | 4194304 | 1000 | 1000 | 929882 |
| 2048x2048_1000components | complete_analysis_prepare_plus_reanalysis | 1.745680 | 1.720909-1.791870 | 155.65 MiB | 4194304 | 1000 | 1000 | 929882 |
| 2048x2048_1000components | cached_cutoff_normalization_shape_reanalysis | 0.520032 | 0.479525-0.532164 | 5.28 MiB | 4194304 | 1000 | 1000 | 929882 |
| 2048x2048_1000components | export_projection_and_write | 0.118508 | 0.112692-0.192555 | 0.03 MiB | 4194304 | 1000 | 1000 | 929882 |
