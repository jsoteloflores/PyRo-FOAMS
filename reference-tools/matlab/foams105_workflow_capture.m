function foams105_workflow_capture(sourceRoot, constructorPath, extractionMapPath, outputDirectory, pyroRoot)
% Replay pinned FOAMS numerical source fragments on a generated input constructor.

if nargin ~= 5
    error('Brief19:Arguments', 'Expected sourceRoot, constructorPath, extractionMapPath, outputDirectory, and pyroRoot.');
end
if exist(outputDirectory, 'file') || exist(outputDirectory, 'dir')
    error('Brief19:OutputExists', 'Capture output must be a fresh directory: %s', outputDirectory);
end
if is_within(outputDirectory, sourceRoot)
    error('Brief19:OutputInsideSource', 'Capture output must not be written inside the original source checkout.');
end
if is_within(sourceRoot, pyroRoot)
    error('Brief19:SourceInsideRepository', 'The original source checkout must remain outside PyRo-FOAMS.');
end

extractionMap = jsondecode(fileread(extractionMapPath));
verifiedSourceFiles = verify_source_checkout(sourceRoot, extractionMap);
[toolboxVersion, toolboxAvailable] = toolbox_identity();
if ~toolboxAvailable || license('checkout', 'image_toolbox') ~= 1
    error('Brief19:ToolboxUnavailable', 'Image Processing Toolbox is required to capture this reference.');
end
if exist('roundn', 'file') ~= 2
    error('Brief19:RoundnUnavailable', 'MATLAB roundn is required by the pinned source.');
end

addpath(sourceRoot);
[constructorFolder, constructorName] = fileparts(constructorPath);
if ~isempty(constructorFolder)
    addpath(constructorFolder);
end
constructorHash = sha256_file(constructorPath);
bundle = feval(constructorName);
if ~strcmp(bundle.source_commit, extractionMap.source_commit)
    error('Brief19:ConstructorSourceMismatch', 'Generated constructor and extraction map source commits differ.');
end

mkdir(outputDirectory);
capture = struct();
capture.schema_version = 1;
capture.source_repository = extractionMap.source_repository;
capture.source_commit = extractionMap.source_commit;
capture.workflow_classification = extractionMap.workflow_classification;
capture.matlab_version = version;
capture.matlab_release = ['R' version('-release')];
capture.image_processing_toolbox_version = toolboxVersion;
capture.runtime_identity = computer;
capture.invocation = sprintf('foams105_workflow_capture(%s)', sourceRoot);
capture.extraction_map_raw_sha256 = sha256_file(extractionMapPath);
capture.constructor_raw_sha256 = constructorHash;
capture.runner_relative_path = 'reference-tools/matlab/foams105_workflow_capture.m';
capture.runner_raw_sha256 = sha256_file(which('foams105_workflow_capture'));
capture.catalog_raw_sha256 = bundle.catalog_raw_sha256;
capture.source_files = verifiedSourceFiles;
capture.cases = repmat(empty_case(), numel(bundle.runs), 1);
capture.image_evidence = {};

dataVariables = {'data1','data2','data3','data16','data4','data5','data6','data7','data8','data9', ...
    'data17','data18','data10','data11','data12','data13','data14','data15','data19','data20'};
scaleVariables = {'a1res','a2res','a3res','a4res','a5res','b1res','b2res','b3res','b4res','b5res', ...
    'c1res','c2res','c3res','c4res','c5res','d1res','d2res','d3res','d4res','d5res'};
dataGroup = [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4];

for caseIndex = 1:numel(bundle.runs)
    currentStage = 'input_setup';
    runSpec = bundle.runs(caseIndex);
    try
        fprintf('CASE %d/%d %s START\n', caseIndex, numel(bundle.runs), runSpec.case_id);
        for index = 1:numel(dataVariables)
            eval(['clear ' dataVariables{index}]); %#ok<EVLDIR>
        end
        for index = 1:numel(scaleVariables)
            eval([scaleVariables{index} ' = 0;']); %#ok<EVLDIR>
        end
        if isempty(runSpec.groups) || numel(runSpec.groups) < 1 || numel(runSpec.groups) > 4
            error('Brief19:InvalidGroups', 'Catalog run must contain one to four groups.');
        end

        firstImage = runSpec.groups(1).images(1);
        J1 = double(firstImage.pore_phase_value); %#ok<NASGU>
        minpix = double(runSpec.preparation.minimum_diameter_px); %#ok<NASGU>
        vesic = double(runSpec.source_metadata.density_derived_vesicularity_percent); %#ok<NASGU>
        if ~isfinite(vesic)
            vesic = 0;
        end
        exclusions = double(firstImage.excluded_phase_values(:)');
        excludedThresholds = double(firstImage.excluded_min_area_mm2(:)');
        excl1 = exclusions(1); %#ok<NASGU>
        excl2 = exclusions(2); %#ok<NASGU>
        excl3 = exclusions(3); %#ok<NASGU>
        excl4 = exclusions(4); %#ok<NASGU>
        excl1_thres = excludedThresholds(1); %#ok<NASGU>
        excl2_thres = excludedThresholds(2); %#ok<NASGU>
        excl3_thres = excludedThresholds(3); %#ok<NASGU>
        excl4_thres = excludedThresholds(4); %#ok<NASGU>

        for groupIndex = 1:numel(runSpec.groups)
            group = runSpec.groups(groupIndex);
            for imageIndex = 1:numel(group.images)
                image = group.images(imageIndex);
                if double(image.pore_phase_value) ~= J1 || ...
                        any(double(image.excluded_phase_values(:)') ~= exclusions) || ...
                        any(double(image.excluded_min_area_mm2(:)') ~= excludedThresholds)
                    error('Brief19:NonGlobalPhaseSettings', 'Pinned project.m has global phase/exclusion settings; catalog varies them.');
                end
                dataSlot = double(image.source_data_slot);
                if dataSlot < 1 || dataSlot > numel(dataVariables) || dataGroup(dataSlot) ~= groupIndex
                    error('Brief19:SourceSlotMapping', 'Catalog source data slot is inconsistent with the original 20-variable group mapping.');
                end
                eval([dataVariables{dataSlot} ' = uint8(image.segmented_image);']); %#ok<EVLDIR>
                eval([scaleVariables{dataSlot} ' = double(group.scale_px_per_mm);']); %#ok<EVLDIR>
            end
        end

        vesnorm_value1 = double(runSpec.analysis.volume_normalization_mode == "integrated_2d"); %#ok<NASGU>
        vesnorm_value2 = double(runSpec.analysis.volume_normalization_mode == "unnormalized"); %#ok<NASGU>
        vesnorm_value3 = double(runSpec.analysis.volume_normalization_mode == "known_vesicularity"); %#ok<NASGU>
        currentStage = 'direct_data_meas_and_mask_instrumentation';
        directImages = instrument_images(runSpec);
        runImageIds = cell(1, numel(directImages));
        for imageIndex = 1:numel(directImages)
            record = directImages{imageIndex};
            runImageIds{imageIndex} = record.image_id;
            knownIndex = find(cellfun(@(item) strcmp(item.image_id, record.image_id), capture.image_evidence), 1);
            if isempty(knownIndex)
                capture.image_evidence{end + 1} = record; %#ok<AGROW>
            elseif ~isequal(capture.image_evidence{knownIndex}, record)
                error('Brief19:ImageEvidenceConflict', 'Shared image ID %s has inconsistent direct MATLAB evidence.', record.image_id);
            end
        end
        currentStage = 'project_group_setup';
        setupFragment = get_fragment(sourceRoot, extractionMap, 'project_group_setup');
        fprintf('CASE_STAGE %s\n', currentStage);
        eval(setupFragment); %#ok<EVLDIR>
        currentStage = 'project_per_image_measurements';
        measurementFragment = get_fragment(sourceRoot, extractionMap, 'project_per_image_measurements');
        fprintf('CASE_STAGE %s\n', currentStage);
        eval(measurementFragment); %#ok<EVLDIR>
        currentStage = 'project_filter_and_bin';
        filterFragment = get_fragment(sourceRoot, extractionMap, 'project_filter_and_bin');
        fprintf('CASE_STAGE %s\n', currentStage);
        eval(filterFragment); %#ok<EVLDIR>
        currentStage = 'project_image_areas_histograms_and_na_tables';
        projectAreaFragment = get_fragment(sourceRoot, extractionMap, 'project_image_areas_histograms_and_na_tables');
        fprintf('CASE_STAGE %s\n', currentStage);
        eval(projectAreaFragment); %#ok<EVLDIR>
        fprintf('PROJECT_GROUP_VECTOR_LENGTHS %s eq=[%d %d %d %d] area=[%d %d %d %d] peri=[%d %d %d %d]\n', ...
            runSpec.case_id, numel(mag1s), numel(magtwos), numel(magthrees), numel(magfours), ...
            numel(magones_area), numel(magtwos_area), numel(magthrees_area), numel(magfours_area), ...
            numel(magones_peri), numel(magtwos_peri), numel(magthrees_peri), numel(magfours_peri));
        fprintf('RAW_TABLE_INPUT_SHAPES %s EqDi=%s Area=%s Peri=%s LA=%s SA=%s Ori=%s\n', ...
            runSpec.case_id, mat2str(size(tot_EqDi)), mat2str(size(tot_area)), mat2str(size(tot_peri)), ...
            mat2str(size(tot_LA)), mat2str(size(tot_SA)), mat2str(size(tot_ori)));
        rawTableFragment = get_fragment(sourceRoot, extractionMap, 'project_pre_nesting_raw_table');
        fprintf('CASE_STAGE project_pre_nesting_raw_table\n');
        eval(rawTableFragment); %#ok<EVLDIR>

        currentStage = 'autosmart_or_manual_cutoffs';
        if runSpec.analysis.selection_mode == "autosmart"
            cutoffFragment = get_fragment(sourceRoot, extractionMap, 'autosmart_cutoff_replay');
            eval(cutoffFragment); %#ok<EVLDIR>
        else
            cutoffNames = {'mag1','magtwos','magthrees','magfours'};
            lowerIndices = double(runSpec.analysis.manual_range_lower_indices(:)');
            upperIndices = double(runSpec.analysis.manual_range_upper_indices(:)');
            for groupIndex = 1:numel(cutoffNames)
                name = cutoffNames{groupIndex};
                if groupIndex > numel(runSpec.groups)
                    eval([name '_lo_bd = 0;']); %#ok<EVLDIR>
                    eval([name '_up_bd = 0;']); %#ok<EVLDIR>
                else
                    lowerIndex = lowerIndices(groupIndex);
                    upperIndex = upperIndices(groupIndex);
                    if lowerIndex < 0 || upperIndex < 0
                        eval([name '_lo_bd = 0;']); %#ok<EVLDIR>
                        eval([name '_up_bd = 0;']); %#ok<EVLDIR>
                    else
                        eval([name '_lo_bd = bins(' num2str(lowerIndex + 1) ');']); %#ok<EVLDIR>
                        eval([name '_up_bd = bins(' num2str(upperIndex + 1) ');']); %#ok<EVLDIR>
                    end
                end
            end
        end
        up_cut = double(runSpec.analysis.size_upper_cutoff_mm); %#ok<NASGU>
        lo_cut = double(runSpec.analysis.size_lower_cutoff_mm); %#ok<NASGU>

        currentStage = 'analysis_selection_vesicularity_shape';
        selectionFragment = get_fragment(sourceRoot, extractionMap, 'analysis_binned_raw_selection_vesicularity_and_shape');
        eval(selectionFragment); %#ok<EVLDIR>
        selectedBinIndices = {encode_numeric(row6), [], [], []};
        if exist('row12', 'var'), selectedBinIndices{2} = encode_numeric(row12); end
        if exist('row18', 'var'), selectedBinIndices{3} = encode_numeric(row18); end
        if exist('row24', 'var'), selectedBinIndices{4} = encode_numeric(row24); end
        current = empty_case();
        current.case_id = runSpec.case_id;
        current.expected_outcome = runSpec.expected_outcome;
        current.outcome = 'success';
        current.stage_completion = {'direct_data_meas_and_mask_instrumentation','project_source_fragment','project_pre_nesting_raw_table','autosmart_or_manual_cutoffs','analysis_selection_vesicularity_shape'};
        current.direct_image_ids = runImageIds;
        current.project_measurements = struct( ...
            'mags', encode_numeric(mags), 'bins', encode_numeric(bins), ...
            'filtered_eq_diameter_groups', {{encode_numeric(mag1s), encode_numeric(magtwos), encode_numeric(magthrees), encode_numeric(magfours)}}, ...
            'retained_indices_original_one_based', {{encode_numeric(find_mag1s_t), encode_numeric(find_mag2s_t), encode_numeric(find_mag3s_t), encode_numeric(find_mag4s_t)}}, ...
            'area1_by_original_data_slot', encode_numeric([mag1_im_area first_im_area second_im_area third_im_area fifteenth_im_area fourth_im_area fifth_im_area sixth_im_area seventh_im_area eigth_im_area ninth_im_area tenth_im_area eleventh_im_area sixteenth_im_area seventeenth_im_area twelth_im_area thirteenth_im_area fourteenth_im_area eighteenth_im_area nineteenth_im_area]), ...
            'area2_by_original_data_slot', encode_numeric([mag1_im_area2 first_im_area2 second_im_area2 third_im_area2 fifteenth_im_area2 fourth_im_area2 fifth_im_area2 sixth_im_area2 seventh_im_area2 eigth_im_area2 ninth_im_area2 tenth_im_area2 eleventh_im_area2 sixteenth_im_area2 seventeenth_im_area2 twelth_im_area2 thirteenth_im_area2 fourteenth_im_area2 eighteenth_im_area2 nineteenth_im_area2]), ...
            'nmag_groups', {{encode_numeric(nmag1), encode_numeric(nmagtwos), encode_numeric(nmagthrees), encode_numeric(nmagfours)}}, ...
            'na_groups_mm2', {{encode_numeric(NAmag1), encode_numeric(NAmagtwos), encode_numeric(NAmagthrees), encode_numeric(NAmagfours)}}, ...
            'selected_binned_labels_mm', encode_numeric(bins_tot), 'selected_na_mm2', encode_numeric(NA_tot), ...
            'binned_source_indices_one_based', {selectedBinIndices}, ...
            'raw_selection_indices_one_based', {{encode_numeric(n1s), encode_numeric(ntwos), encode_numeric(nthrees), encode_numeric(nfours)}});
        retainedByGroup = {find_mag1s_t, find_mag2s_t, find_mag3s_t, find_mag4s_t};
        cutoffByGroup = {n1s, ntwos, nthrees, nfours};
        identities = build_identity_projections(runSpec, directImages, retainedByGroup, cutoffByGroup);
        current.raw_data_identities = identities.raw_data;
        current.rawcutoffs_identities = identities.rawcutoffs;
        current.initial_shape = struct( ...
            'equivalent_diameter_mm', encode_numeric(tot_EqDi_raw), 'area_mm2', encode_numeric(tot_area_raw), ...
            'perimeter_mm', encode_numeric(tot_peri_raw), 'major_axis_mm', encode_numeric(tot_LA_raw), ...
            'minor_axis_mm', encode_numeric(tot_SA_raw), 'orientation_project_deg', encode_numeric(tot_ori_raw), ...
            'eccentricity', encode_numeric(tot_ecc_raw), 'regularity', encode_numeric(regularity), ...
            'elongation', encode_numeric(elong), 'aspect_ratio', encode_numeric(asp_rat), ...
            'fitted_ellipse_perimeter_mm', encode_numeric(Peri_ell), ...
            'threshold_indices_one_based', encode_numeric(location_thres), ...
            'threshold_identities', select_identity_rows(current.rawcutoffs_identities, location_thres));
        currentStage = 'analysis_stats_and_categories';
        fprintf('INITIAL_STATS_INPUTS %s n=[%d %d %d %d] finite=[%d %d %d]\n', ...
            runSpec.case_id, numel(asp_rat), numel(elong), numel(regularity), numel(tot_ecc_raw), ...
            nnz(isfinite(asp_rat)), nnz(isfinite(elong)), nnz(isfinite(regularity)));
        initialStatsFragment = get_fragment(sourceRoot, extractionMap, 'analysis_basicstats_calls');
        eval(initialStatsFragment); %#ok<EVLDIR>
        current.initial_statistics = encode_stats(stats_asp_rat, stats_elong, stats_reg, stats_ecc);
        currentStage = 'analysis_image_count';
        imageCountFragment = get_fragment(sourceRoot, extractionMap, 'analysis_image_count');
        eval(imageCountFragment); %#ok<EVLDIR>
        currentStage = 'analysis_size_category_percentages';
        categoriesFragment = get_fragment(sourceRoot, extractionMap, 'analysis_size_category_percentages');
        categoryLines = regexp(categoriesFragment, '\n', 'split');
        for categoryLineIndex = 1:numel(categoryLines)
            categoryLine = strtrim(categoryLines{categoryLineIndex});
            if isempty(categoryLine) || startsWith(categoryLine, '%')
                continue
            end
            fprintf('CATEGORY_STATEMENT %s\n', categoryLine);
            if strcmp(categoryLine, 'count_up=count_mags(bins_up);')
                fprintf('CATEGORY_INDEX_DIAGNOSTIC %s counts=%d bins=%d max_up=%g up_count=%d\n', ...
                    runSpec.case_id, numel(count_mags), numel(bins), max([0; bins_up(:)]), numel(bins_up));
            end
            eval(categoryLine); %#ok<EVLDIR>
        end

        currentStage = 'bulk_na_plot_source';
        bulkFragment = get_fragment(sourceRoot, extractionMap, 'analysis_bulk_na');
        eval(bulkFragment); %#ok<EVLDIR>
        current.bulk_na = struct('x_areas_mm2', encode_numeric(im_areas), 'y_values_mm2', encode_numeric(bulkNA), ...
            'slots_descending', encode_numeric([bulkNA_magfours bulkNA_magthrees bulkNA_magtwos bulkNA_mag1]));

        currentStage = 'volume_and_distribution_source';
        nvFragment = get_fragment(sourceRoot, extractionMap, 'analysis_nv_conversion');
        eval(nvFragment); %#ok<EVLDIR>
        nvOrientationFragment = get_fragment(sourceRoot, extractionMap, 'analysis_nv_output_orientation');
        eval(nvOrientationFragment); %#ok<EVLDIR>
        fprintf('CASE_STAGE analysis_volume_normalization_and_image_vesicularity\n');
        volumeFragment = get_fragment(sourceRoot, extractionMap, 'analysis_volume_normalization_and_image_vesicularity');
        eval(volumeFragment); %#ok<EVLDIR>
        fprintf('CASE_STAGE analysis_distribution_summary\n');
        summaryFragment = get_fragment(sourceRoot, extractionMap, 'analysis_distribution_summary');
        eval(summaryFragment); %#ok<EVLDIR>
        fprintf('CASE_STAGE analysis_misc_table_array\n');
        miscFragment = get_fragment(sourceRoot, extractionMap, 'analysis_misc_table_array');
        eval(miscFragment); %#ok<EVLDIR>
        alphaValues = eval('alpha'); %#ok<EVLDIR>
        current.volume = struct('Hbar_mm', encode_numeric(Hbar), 'alpha', encode_numeric(alphaValues), ...
            'nv_mm3', encode_numeric(NV_tot), 'sphere_volume_mm3', encode_numeric(eq_sphere), ...
            'raw_volume_fraction', encode_numeric(vol_frac), 'normalization_factor', encode_numeric(mult_fact), ...
            'adjusted_volume_fraction', encode_numeric(adj_vol), 'cumulative_adjusted_volume_fraction', encode_numeric(cum_vol), ...
            'ves2d_percent', encode_numeric(ves2d), 'group_vesicularity_percent', encode_numeric([mag1_ves magtwos_ves magthrees_ves magfours_ves]), ...
            'mean_group_vesicularity_percent', encode_numeric(im_ves), 'summary_bins_low_mm', encode_numeric(bins_low), ...
            'summary_bin_widths_mm', encode_numeric(bins_dif), 'log_cumulative_nv', encode_numeric(Log_Nvl), ...
            'log_label', encode_numeric(Log_l), 'ln_number_density_per_width', encode_numeric(ln_n));

        currentStage = 'shape_histograms_initial';
        current.initial_shape_histograms = capture_shape_histograms(sourceRoot, extractionMap, tot_ecc_thres, tot_elong_thres, tot_AR_thres, tot_reg_thres, tot_Or_thres);
        if runSpec.analysis.shape_statistics_mode == "updated_minimum"
            currentStage = 'updated_shape_statistics_source';
            min_diameter = double(runSpec.analysis.shape_minimum_diameter_px); %#ok<NASGU>
            updatedFragment = get_fragment(sourceRoot, extractionMap, 'updated_shape_minimum_statistics');
            eval(updatedFragment); %#ok<EVLDIR>
            current.updated_shape = struct('threshold_indices_one_based', encode_numeric(location_thres), ...
                'equivalent_diameter_mm', encode_numeric(tot_EqDi_thres), 'area_mm2', encode_numeric(tot_area_thres), ...
                'eccentricity', encode_numeric(tot_ecc_thres), 'orientation_project_deg', encode_numeric(tot_Or_thres), ...
                'regularity', encode_numeric(tot_reg_thres), 'elongation', encode_numeric(tot_elong_thres), ...
                'aspect_ratio', encode_numeric(tot_AR_thres), 'statistics', encode_stats(stats_asp_rat, stats_elong, stats_reg, stats_ecc), ...
                'threshold_identities', select_identity_rows(current.rawcutoffs_identities, location_thres));
            current.updated_shape_histograms = capture_shape_histograms(sourceRoot, extractionMap, tot_ecc_thres, tot_elong_thres, tot_AR_thres, tot_reg_thres, tot_Or_thres);
        else
            current.updated_shape = struct();
            current.updated_shape_histograms = struct();
        end

        currentStage = 'table_arrays';
        calcFragment = get_fragment(sourceRoot, extractionMap, 'analysis_calc_table_array');
        eval(calcFragment); %#ok<EVLDIR>
        vesFragment = get_fragment(sourceRoot, extractionMap, 'analysis_ves_table_array');
        eval(vesFragment); %#ok<EVLDIR>
        miscTableFragment = get_fragment(sourceRoot, extractionMap, 'analysis_misc_table_array');
        eval(miscTableFragment); %#ok<EVLDIR>
        areaFragment = get_fragment(sourceRoot, extractionMap, 'analysis_area_table_array');
        eval(areaFragment); %#ok<EVLDIR>
        shapeFragment = get_fragment(sourceRoot, extractionMap, 'analysis_shape_table_array');
        eval(shapeFragment); %#ok<EVLDIR>
        rawcutoffsFragment = get_fragment(sourceRoot, extractionMap, 'analysis_rawcutoffs_table_array');
        eval(rawcutoffsFragment); %#ok<EVLDIR>
        statsFragment = get_fragment(sourceRoot, extractionMap, 'stats_table_array');
        eval(statsFragment); %#ok<EVLDIR>
        current.tables = struct();
        current.tables.NA_mag = encode_table(titlesrslt, {}, sprdrslt);
        current.tables.Raw_data = encode_table(titlesraw, {}, sprdraw);
        current.tables.Rawcutoffs_data = encode_table(titlesraw_raw, {}, sprdraw_raw);
        current.tables.Calc_out = encode_table(titlescalc, {}, sprdcalc);
        current.tables.Area_out = encode_table(titlesarea, {}, sprdarea);
        current.tables.Ves_out = encode_table(titlesves, {}, sprdves);
        current.tables.Misc_out = encode_table(titlesmisc, {}, sprdmisc);
        current.tables.Shape_out = encode_table(titlesshape, {}, sprdshape);
        current.tables.Stats_out = encode_table(titlesstats, titlesstats2, sprdstats);
        current.stage_completion = {'direct_data_meas_and_mask_instrumentation','project_source_fragment','project_pre_nesting_raw_table','autosmart_or_manual_cutoffs','analysis_selection_vesicularity_shape','bulk_na_plot_source','volume_and_distribution_source','shape_histograms_initial','table_arrays'};
        current.outcome = 'success';
        capture.cases(caseIndex) = current;
    catch sourceException
        failure = struct('case_id', runSpec.case_id, 'stage', currentStage, ...
            'identifier', sourceException.identifier, 'message', sourceException.message, 'stack', sourceException.stack);
        if strcmp(runSpec.expected_outcome, 'expected_domain_failure') && strcmp(currentStage, runSpec.expected_source_stage)
            current = empty_case();
            current.case_id = runSpec.case_id;
            current.expected_outcome = runSpec.expected_outcome;
            current.expected_python_stage = runSpec.expected_stage;
            current.expected_source_stage = runSpec.expected_source_stage;
            current.direct_image_ids = runImageIds;
            current.outcome = 'expected_source_domain_failure';
            current.failure = failure;
            current.stage_completion = {'source_validation','input_setup',currentStage};
            capture.cases(caseIndex) = current;
        else
            capture.failure = failure;
            write_json(fullfile(outputDirectory, 'capture.partial.json'), capture);
            write_json(fullfile(outputDirectory, 'failure.json'), failure);
            rethrow(sourceException);
        end
    end
end

capturePath = fullfile(outputDirectory, 'capture.json');
write_json(capturePath, capture);
manifest = struct('schema_version', 1, 'source_commit', extractionMap.source_commit, ...
    'capture_filename', 'capture.json', 'capture_sha256', sha256_file(capturePath), ...
    'extraction_map_raw_sha256', capture.extraction_map_raw_sha256, ...
    'constructor_raw_sha256', constructorHash, 'runner_raw_sha256', capture.runner_raw_sha256, ...
    'catalog_raw_sha256', bundle.catalog_raw_sha256, ...
    'matlab_release', capture.matlab_release, 'matlab_version', capture.matlab_version, ...
    'image_processing_toolbox_version', toolboxVersion, 'case_count', numel(capture.cases));
manifestPath = fullfile(outputDirectory, 'manifest.json');
write_json(manifestPath, manifest);
fprintf('Original-source numerical replay stages finished; outer runner must persist logs/checksums and write completion marker: %s\n', outputDirectory);
end

function verifiedSourceFiles = verify_source_checkout(sourceRoot, extractionMap)
[status, commit] = system(sprintf('git -C "%s" rev-parse HEAD', sourceRoot));
if status ~= 0
    error('Brief19:InvalidSourceCheckout', 'Cannot inspect original source git checkout.');
end
commit = strtrim(commit);
if ~strcmp(commit, extractionMap.source_commit)
    error('Brief19:WrongSourceCommit', 'Expected source commit %s but found %s.', extractionMap.source_commit, commit);
end
fileNames = fieldnames(extractionMap.source_files);
verifiedSourceFiles = repmat(struct('path', '', 'raw_sha256', '', 'lf_sha256', ''), numel(fileNames), 1);
for index = 1:numel(fileNames)
    mapField = fileNames{index};
    if ~endsWith(mapField, '_m')
        error('Brief19:InvalidSourceMapField', 'Unexpected decoded source-file key %s.', mapField);
    end
    fileName = [mapField(1:end-2) '.m'];
    filePath = fullfile(sourceRoot, fileName);
    rawHash = sha256_file(filePath);
    normalizedHash = sha256_bytes(normalize_lf(read_bytes(filePath)));
    expected = extractionMap.source_files.(mapField);
    if ~strcmp(normalizedHash, expected.lf_sha256)
        error('Brief19:SourceHashMismatch', 'Source file %s has LF hash %s, expected %s (raw hash %s).', ...
            fileName, normalizedHash, expected.lf_sha256, rawHash);
    end
    verifiedSourceFiles(index) = struct('path', fileName, 'raw_sha256', rawHash, 'lf_sha256', normalizedHash);
end
end

function current = empty_case()
current = struct('case_id','','expected_outcome','','expected_python_stage','','expected_source_stage','','outcome','','stage_completion',{{}}, ...
    'direct_image_ids',{{}},'project_measurements',[],'initial_shape',[],'initial_statistics',[], ...
    'raw_data_identities',[],'rawcutoffs_identities',[], ...
    'initial_shape_histograms',[],'updated_shape',[],'updated_shape_histograms',[], ...
    'bulk_na',[],'volume',[],'tables',[],'failure',[]);
end

function images = instrument_images(runSpec)
images = {};
firstImage = runSpec.groups(1).images(1);
phase = double(firstImage.pore_phase_value);
exclusions = double(firstImage.excluded_phase_values(:)');
for groupIndex = 1:numel(runSpec.groups)
    group = runSpec.groups(groupIndex);
    for imageIndex = 1:numel(group.images)
        image = group.images(imageIndex);
        raw = uint8(image.segmented_image);
        K0 = (raw == phase);
        K1 = imclearborder(K0, 8);
        borderMask = K0 - K1;
        K2 = imfill(K1, 'holes');
        K3 = bwlabel(K2, 4);
        [SelA, SelLA, SelSA, SelEcc, SelOr, SelPeri, borderk, excl1_area, excl2_area, excl3_area, excl4_area] = ...
            data_meas(raw, phase, exclusions(1), exclusions(2), exclusions(3), exclusions(4));
        if ~isequal(logical(borderk), logical(borderMask)) || numel(SelA) ~= max(K3(:))
            error('Brief19:DataMeasInstrumentationMismatch', 'Direct masks/labels differ from data_meas outputs for %s.', image.image_id);
        end
        for componentId = 1:numel(SelA)
            if nnz(K3 == componentId) ~= SelA(componentId)
                error('Brief19:DataMeasComponentMismatch', 'Label membership area differs from data_meas for %s:%d.', image.image_id, componentId);
            end
        end
        componentRecords = cell(numel(SelA), 1);
        for componentId = 1:numel(SelA)
            linearIndices = find(K3 == componentId);
            [rows, columns] = ind2sub(size(K3), linearIndices);
            firstRow = min(rows);
            firstColumn = min(columns(rows == firstRow));
            componentRecords{componentId} = struct('component_id', componentId, 'first_row_zero_based', firstRow - 1, ...
                'first_column_zero_based', firstColumn - 1, 'pixel_linear_indices_column_major_one_based', encode_numeric(linearIndices));
        end
        record = struct();
        record.image_id = image.image_id;
        record.source_data_slot = double(image.source_data_slot);
        record.source_image_key = image.source_image_key;
        record.segmented_image = encode_numeric(raw);
        record.pore_mask = encode_numeric(K0);
        record.border_removed_mask = encode_numeric(K1);
        record.border_mask = encode_numeric(borderMask);
        record.filled_mask = encode_numeric(K2);
        record.labels = encode_numeric(K3);
        record.measurements = struct('area_px', encode_numeric(SelA), 'major_axis_px', encode_numeric(SelLA), ...
            'minor_axis_px', encode_numeric(SelSA), 'eccentricity', encode_numeric(SelEcc), ...
            'orientation_regionprops_deg', encode_numeric(SelOr), 'perimeter_regionprops_px', encode_numeric(SelPeri), ...
            'excluded_area_px', {{encode_numeric(excl1_area), encode_numeric(excl2_area), encode_numeric(excl3_area), encode_numeric(excl4_area)}});
        record.components = componentRecords;
        images{end + 1} = record; %#ok<AGROW>
    end
end
end

function result = build_identity_projections(runSpec, directImages, retainedByGroup, cutoffByGroup)
template = struct('output_index', 0, 'source_slot', 0, 'source_data_slot', 0, ...
    'source_object_index_original_one_based', 0, 'image_id', '', 'component_id', 0, ...
    'first_row_zero_based', 0, 'first_column_zero_based', 0);
rawData = repmat(template, 0, 1);
rawCutoffs = repmat(template, 0, 1);
for groupIndex = 1:numel(runSpec.groups)
    group = runSpec.groups(groupIndex);
    objects = repmat(template, 0, 1);
    previousDataSlot = 0;
    for imageIndex = 1:numel(group.images)
        image = group.images(imageIndex);
        dataSlot = double(image.source_data_slot);
        if dataSlot <= previousDataSlot
            error('Brief19:ImageSlotOrder', 'Catalog images are not in original project.m concatenation order.');
        end
        previousDataSlot = dataSlot;
        evidenceIndex = find(cellfun(@(item) strcmp(item.image_id, image.image_id), directImages), 1);
        if isempty(evidenceIndex)
            error('Brief19:MissingDirectImageEvidence', 'No direct image record exists for %s.', image.image_id);
        end
        evidence = directImages{evidenceIndex};
        if evidence.source_data_slot ~= dataSlot
            error('Brief19:ImageDataSlotMismatch', 'Direct image data slot differs for %s.', image.image_id);
        end
        for componentIndex = 1:numel(evidence.components)
            component = evidence.components{componentIndex};
            object = template;
            object.source_slot = double(group.source_slot);
            object.source_data_slot = dataSlot;
            object.source_object_index_original_one_based = numel(objects) + 1;
            object.image_id = image.image_id;
            object.component_id = component.component_id;
            object.first_row_zero_based = component.first_row_zero_based;
            object.first_column_zero_based = component.first_column_zero_based;
            objects(end + 1) = object; %#ok<AGROW>
        end
    end
    retained = double(retainedByGroup{groupIndex}(:));
    if any(retained < 1) || any(retained > numel(objects))
        error('Brief19:RawIdentityIndex', 'Retained index lies outside source data-slot objects.');
    end
    filtered = reshape(objects(retained), [], 1);
    resultStart = numel(rawData);
    for localIndex = 1:numel(filtered)
        filtered(localIndex).output_index = resultStart + localIndex - 1;
    end
    rawData = [rawData; filtered]; %#ok<AGROW>

    selected = double(cutoffByGroup{groupIndex}(:));
    if any(selected < 1) || any(selected > numel(filtered))
        error('Brief19:RawCutoffIdentityIndex', 'Cutoff index lies outside diameter-filtered objects.');
    end
    selectedObjects = reshape(filtered(selected), [], 1);
    resultStart = numel(rawCutoffs);
    for localIndex = 1:numel(selectedObjects)
        selectedObjects(localIndex).output_index = resultStart + localIndex - 1;
    end
    rawCutoffs = [rawCutoffs; selectedObjects]; %#ok<AGROW>
end
result = struct('raw_data', rawData, 'rawcutoffs', rawCutoffs);
end

function identities = select_identity_rows(rows, indices)
indices = double(indices(:));
if any(indices < 1) || any(indices > numel(rows))
    error('Brief19:ShapeIdentityIndex', 'Shape threshold index lies outside the raw-cutoff identity vector.');
end
identities = rows(indices);
end

function fragmentText = get_fragment(sourceRoot, extractionMap, fragmentId)
entries = as_cell(extractionMap.fragments);
entry = [];
for entryIndex = 1:numel(entries)
    candidate = entries{entryIndex};
    if strcmp(candidate.id, fragmentId)
        entry = candidate;
        break
    end
end
if isempty(entry)
    error('Brief19:UnknownFragment', 'Fragment %s is not in the extraction map.', fragmentId);
end
normalized = normalize_lf(read_bytes(fullfile(sourceRoot, entry.file)));
textValue = native2unicode(normalized, 'UTF-8');
lines = regexp(textValue, '\n', 'split');
if ~isempty(lines) && isempty(lines{end})
    lines(end) = [];
end
if entry.end_line > numel(lines)
    error('Brief19:FragmentOutOfRange', 'Fragment %s exceeds %s.', fragmentId, entry.file);
end
selected = lines(entry.start_line:entry.end_line);
fragmentText = sprintf('%s\n', selected{:});
if ~strcmp(sha256_bytes(uint8(fragmentText)), entry.lf_sha256)
    error('Brief19:FragmentHashMismatch', 'Fragment %s no longer matches its extraction map.', fragmentId);
end
executable = {};
for lineIndex = 1:numel(selected)
    if ~should_omit(selected{lineIndex}, entry.omitted_calls)
        executable{end + 1} = selected{lineIndex}; %#ok<AGROW>
    end
end
fragmentText = sprintf('%s\n', executable{:});
forbidden = '(\b(waitbar|findobj|helpdlg|uiwait|axes|figure|plot|loglog|semilogy|bar|tblwrite|xlswrite|get)\s*\()';
for lineIndex = 1:numel(executable)
    if ~isempty(regexp(executable{lineIndex}, forbidden, 'once'))
        error('Brief19:UnmappedGuiCall', 'Fragment %s line %d retains a GUI/render/write call: %s', ...
            fragmentId, entry.start_line + lineIndex - 1, strtrim(executable{lineIndex}));
    end
end
end

function omitted = should_omit(line, omittedMap)
stripped = strtrim(line);
omitted = false;
if isempty(stripped) || startsWith(stripped, '%')
    return
end
entries = as_cell(omittedMap);
for index = 1:numel(entries)
    item = entries{index};
    call = item.call;
    if strcmp(call, 'waitbar')
        omitted = ~isempty(regexp(stripped, '^(\w+\s*=\s*)?waitbar\s*\(', 'once'));
    elseif strcmp(call, 'findobj/get')
        omitted = contains(stripped, 'findobj(') || contains(stripped, 'get(') || contains(stripped, 'get (');
    elseif strcmp(call, 'set/findobj')
        omitted = contains(stripped, 'set(') && contains(stripped, 'findobj(');
    elseif strcmp(call, 'helpdlg')
        omitted = contains(stripped, 'helpdlg(');
    elseif strcmp(call, 'axes')
        omitted = startsWith(stripped, 'axes(');
    end
    if omitted
        return
    end
end
end

function entries = as_cell(value)
if iscell(value)
    entries = value;
elseif isempty(value)
    entries = {};
else
    entries = num2cell(value);
end
end

function result = capture_shape_histograms(sourceRoot, extractionMap, tot_ecc_thres, tot_elong_thres, tot_AR_thres, tot_reg_thres, tot_Or_thres) %#ok<INUSD>
result = struct();
for histogramIndex = 1:numel(extractionMap.histogram_calls)
    entry = extractionMap.histogram_calls(histogramIndex);
    sourceText = native2unicode(normalize_lf(read_bytes(fullfile(sourceRoot, entry.file))), 'UTF-8');
    sourceLines = regexp(sourceText, '\n', 'split');
    line = sourceLines{entry.line};
    if ~strcmp(sha256_bytes(uint8([line newline])), entry.lf_sha256) || ~contains(line, entry.expression)
        error('Brief19:HistogramHashMismatch', 'Histogram source call %s drifted.', entry.id);
    end
    if strcmp(entry.function, 'histc')
        histCenters = eval(entry.grid_expression); %#ok<EVLDIR>
        histCounts = eval(entry.expression); %#ok<EVLDIR>
    else
        [histCounts, histCenters] = eval(entry.expression); %#ok<EVLDIR>
    end
    result.(entry.id) = struct('source_expression', entry.expression, ...
        'grid_kind', entry.grid_kind, ...
        'centers_or_edges', encode_numeric(histCenters), 'counts', encode_numeric(histCounts));
end
end

function encoded = encode_stats(statsAspect, statsElongation, statsRegularity, statsEccentricity)
encoded = struct('aspect_ratio', encode_numeric(statsAspect), 'elongation', encode_numeric(statsElongation), ...
    'regularity', encode_numeric(statsRegularity), 'eccentricity', encode_numeric(statsEccentricity));
end

function encoded = encode_table(headers, rowLabels, values)
encoded = struct('headers', {cellstr(headers)}, 'row_labels', {cellstr(rowLabels)}, 'values', encode_numeric(values));
end

function encoded = encode_numeric(values)
if ~isnumeric(values) && ~islogical(values)
    error('Brief19:UnsupportedNumericValue', 'Capture value must be numeric/logical.');
end
encoded = struct('kind', 'numeric_array', 'shape', size(values), 'cells', {cell(numel(values), 1)});
for index = 1:numel(values)
    value = values(index);
    if ~isreal(value)
        classification = 'complex'; realText = sprintf('%.17g', real(value)); imaginaryText = sprintf('%.17g', imag(value));
    elseif isnan(value)
        classification = 'nan'; realText = ''; imaginaryText = '';
    elseif isinf(value) && value > 0
        classification = 'positive_infinity'; realText = ''; imaginaryText = '';
    elseif isinf(value)
        classification = 'negative_infinity'; realText = ''; imaginaryText = '';
    else
        classification = 'finite'; realText = sprintf('%.17g', double(value)); imaginaryText = '';
    end
    encoded.cells{index} = struct('classification', classification, 'real', realText, 'imaginary', imaginaryText);
end
end

function write_json(path, value)
textValue = jsonencode(value, 'PrettyPrint', true);
write_text(path, [textValue newline]);
end

function write_text(path, textValue)
fid = fopen(path, 'w', 'n', 'UTF-8');
if fid < 0
    error('Brief19:OutputOpenFailure', 'Unable to open output file %s.', path);
end
cleanup = onCleanup(@() close_if_open(fid));
bytes = unicode2native(textValue, 'UTF-8');
count = fwrite(fid, bytes, 'uint8');
if count ~= numel(bytes)
    error('Brief19:OutputWriteFailure', 'Short write for %s.', path);
end
clear cleanup;
end

function close_if_open(fid)
fclose(fid);
end

function bytes = read_bytes(path)
fid = fopen(path, 'rb');
if fid < 0
    error('Brief19:SourceFileUnavailable', 'Cannot open %s.', path);
end
cleanup = onCleanup(@() fclose(fid));
bytes = fread(fid, Inf, '*uint8')';
clear cleanup;
end

function normalized = normalize_lf(bytes)
textValue = char(bytes);
textValue = strrep(textValue, sprintf('\r\n'), sprintf('\n'));
textValue = strrep(textValue, sprintf('\r'), sprintf('\n'));
normalized = uint8(textValue);
end

function hash = sha256_file(path)
hash = sha256_bytes(read_bytes(path));
end

function hash = sha256_bytes(bytes)
digest = java.security.MessageDigest.getInstance('SHA-256');
digest.update(typecast(uint8(bytes(:)), 'int8'));
hashBytes = typecast(digest.digest(), 'uint8');
hash = lower(reshape(dec2hex(hashBytes, 2).', 1, []));
end

function [toolboxVersion, available] = toolbox_identity()
information = ver('images');
available = ~isempty(information);
if available
    toolboxVersion = information(1).Version;
else
    toolboxVersion = '';
end
end

function contained = is_within(pathValue, parentValue)
pathValue = char(java.io.File(pathValue).getCanonicalPath());
parentValue = char(java.io.File(parentValue).getCanonicalPath());
contained = startsWith([pathValue filesep], [parentValue filesep], 'IgnoreCase', ispc) || strcmpi(pathValue, parentValue);
end