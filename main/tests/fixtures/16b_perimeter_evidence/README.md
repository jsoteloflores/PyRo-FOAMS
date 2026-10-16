# Reviewer MATLAB perimeter evidence

Source checkout under review: jsoteloflores/PyRo-FOAMS at 62cf3bf5db2797e324446f654008ba0f51a41e73.
MATLAB 25.2.0.2998904 (R2025b), Image Processing Toolbox 25.2.
Both MATLAB commands completed with exit 0. Original source pin: 179663203f2d0f86b2863d5ea7f8f70dadca02f8.

Primary invocation from the audit checkout:
matlab.exe -wait -nosplash -logfile matlab-review.log -batch "run('capture_review.m')"
Supplementary invocation:
matlab.exe -wait -nosplash -logfile matlab-supplementary.log -batch "run('capture_supplementary.m')"

Primary importer accepted all 29 cases/31 components with matching labels/topology. Within-run current/old comparison reports 24 differences, exit 1 as expected. This compares two MATLAB properties, not MATLAB against Python.

prototype_perimeter.py was separately checked against both captured properties: zero mismatches across 31 objects at relative/absolute 1e-12 pixel.
Supplementary raw masks were generated with NumPy default_rng seed 160105. MATLAB independently applied imclearborder(...,8), imfill holes, bwlabel(...,4), and both regionprops properties. All 64 full label images match Python measurement output; both perimeter variants match for 526 objects. Maximum absolute supplementary difference is 1.4210854715202004e-14 pixel.

review_supplementary_cases.json contains raw inputs and REVIEW PROTOTYPE predictions, clearly distinct from MATLAB values in review_supplementary.tsv. Do not use predictions as the independent MATLAB oracle.
MATLAB's empty supplementary log is expected: that script wrote TSV output without printing status. Its process returned 0.

The narrow_neck_final primary case has a diagonally joined same-ID region, not a valid four-connected component. Retain as explicitly labeled diagnostic data; production tests must respect accepted measurement topology.

This archive contains reviewer-created methods/scripts and measured output, not redistributed MathWorks source code. The installed regionprops implementation was inspected to confirm the formulas. Runtime evidence is R2025b-specific and does not identify the historical runtime of original FOAMS workbooks.

SHA256SUMS.json records raw file hashes for archive integrity (not proof of execution authenticity). The preparation manifest instead uses LF-normalized hashes as specified by the tooling. Script paths in copied review helpers refer to the audit checkout layout; use the original MATLAB outputs directly or adapt paths when reproducing.
