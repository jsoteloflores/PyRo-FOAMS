function foams105_perimeter_capture_tsv(constructorFunction, outputDir, invocation)
% Capture perimeter reference evidence in TSV files for legacy-friendly MATLAB.

if nargin < 1 || isempty(constructorFunction)
    constructorFunction = 'foams105_perimeter_cases_generated';
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile('reference-tools', 'matlab', 'perimeter-run');
end
if nargin < 3 || isempty(invocation)
    invocation = 'foams105_perimeter_capture_tsv default invocation';
end

bundle = feval(constructorFunction);
cases = bundle.cases;

tempDir = [outputDir '.tmp'];
if exist(tempDir, 'dir')
    rmdir(tempDir, 's');
end
mkdir(tempDir);

metadataPath = fullfile(tempDir, 'metadata.tsv');
casesPath = fullfile(tempDir, 'cases.tsv');
labelsPath = fullfile(tempDir, 'labels.tsv');
componentsPath = fullfile(tempDir, 'components.tsv');

metadataFid = fopen(metadataPath, 'w');
if metadataFid < 0
    error('Unable to create metadata.tsv');
end
casesFid = fopen(casesPath, 'w');
if casesFid < 0
    fclose(metadataFid);
    error('Unable to create cases.tsv');
end
labelsFid = fopen(labelsPath, 'w');
if labelsFid < 0
    fclose(metadataFid); fclose(casesFid);
    error('Unable to create labels.tsv');
end
componentsFid = fopen(componentsPath, 'w');
if componentsFid < 0
    fclose(metadataFid); fclose(casesFid); fclose(labelsFid);
    error('Unable to create components.tsv');
end
cleanupFiles = onCleanup(@() close_files(metadataFid, casesFid, labelsFid, componentsFid));

imagesInfo = ver('images');
if isempty(imagesInfo)
    toolboxVersion = '';
else
    toolboxVersion = imagesInfo(1).Version;
end

fprintf(metadataFid, 'schema_version\t%d\n', bundle.schema_version);
fprintf(metadataFid, 'source_commit\t%s\n', bundle.source_commit);
fprintf(metadataFid, 'case_catalog_hash\t%s\n', bundle.case_catalog_hash);
fprintf(metadataFid, 'preparation_hash\t%s\n', bundle.preparation_hash);
fprintf(metadataFid, 'runner_hash\t%s\n', bundle.runner_hash);
fprintf(metadataFid, 'runtime_identity\t%s\n', computer);
fprintf(metadataFid, 'matlab_version\t%s\n', version);
fprintf(metadataFid, 'matlab_release\t%s\n', version('-release'));
fprintf(metadataFid, 'image_processing_toolbox_version\t%s\n', toolboxVersion);
fprintf(metadataFid, 'invocation\t%s\n', invocation);
fprintf(metadataFid, 'timestamp_description\t%s\n', datestr(now, 31));

fprintf(casesFid, 'case_id\tstage\theight\twidth\tscale_px_per_mm\tforeground_pixel_count\tcomponent_count\toutcome\n');
fprintf(labelsFid, 'case_id\trow\tcolumn\tlabel\n');
fprintf(componentsFid, ['case_id\tnative_component_id\tcanonical_component_id\tfirst_row\tfirst_column\tarea_px' ...
    '\tperimeter_px\tperimeter_status\tperimeter_error\tperimeter_old_px\tperimeter_old_status\tperimeter_old_error\n']);

perimeterOldProbeState = 'unprobed';
perimeterOldProbeError = '';

for caseIndex = 1:numel(cases)
    caseRecord = cases(caseIndex);
    if strcmp(caseRecord.stage, 'raw_phase_mask')
        rawImage = uint8(caseRecord.raw_phase_image);
        phaseValue = uint8(caseRecord.raw_phase_value);
        selected = rawImage == phaseValue;
        retained = imclearborder(selected, 8);
        filled = imfill(retained, 'holes');
        labels = int32(bwlabel(filled, 4));
    else
        labels = int32(caseRecord.labels);
    end
    [height, width] = size(labels);

    if height ~= caseRecord.height || width ~= caseRecord.width
        error('Case %s labels do not match declared dimensions', caseRecord.id);
    end

    [topology, canonicalAscending] = compute_topology(labels, caseRecord.id);
    if ~canonicalAscending
        error('Case %s canonical first indices are not ascending', caseRecord.id);
    end

    if numel(topology.native_id) ~= caseRecord.component_count
        error('Case %s component count mismatch during capture', caseRecord.id);
    end

    foregroundCount = nnz(labels > 0);
    outcome = 'success';
    fprintf(casesFid, '%s\t%s\t%d\t%d\t%.17g\t%d\t%d\t%s\n', ...
        caseRecord.id, caseRecord.stage, height, width, caseRecord.scale_px_per_mm, ...
        foregroundCount, numel(topology.native_id), outcome);

    for row = 1:height
        for col = 1:width
            fprintf(labelsFid, '%s\t%d\t%d\t%d\n', caseRecord.id, row - 1, col - 1, labels(row, col));
        end
    end

    props = regionprops(labels, 'Area', 'Perimeter');

    if strcmp(perimeterOldProbeState, 'unprobed') && nnz(labels > 0) > 0
        try
            regionprops(labels, 'PerimeterOld');
            perimeterOldProbeState = 'supported';
            perimeterOldProbeError = '';
        catch probeException
            if is_perimeter_old_unsupported(probeException)
                perimeterOldProbeState = 'unsupported';
            else
                perimeterOldProbeState = 'error';
            end
            perimeterOldProbeError = probeException.identifier;
        end
    end

    oldProps = [];
    oldRunState = perimeterOldProbeState;
    oldRunError = perimeterOldProbeError;
    if strcmp(perimeterOldProbeState, 'supported')
        try
            oldProps = regionprops(labels, 'PerimeterOld');
        catch oldException
            oldRunState = 'error';
            oldRunError = oldException.identifier;
        end
    end

    for componentIndex = 1:numel(topology.native_id)
        nativeId = topology.native_id(componentIndex);
        canonicalId = topology.canonical_id(componentIndex);
        firstRow = topology.first_row(componentIndex) - 1;
        firstCol = topology.first_col(componentIndex) - 1;
        areaPx = topology.area_px(componentIndex);

        perimeterStatus = 'missing';
        perimeterValue = '';
        perimeterError = '';
        if nativeId <= numel(props)
            perimeterCandidate = props(nativeId).Perimeter;
            if isempty(perimeterCandidate)
                perimeterStatus = 'missing';
            elseif ~isfinite(perimeterCandidate)
                perimeterStatus = 'error_nonfinite';
                perimeterError = 'Perimeter_nonfinite';
            elseif perimeterCandidate < 0
                perimeterStatus = 'error_negative';
                perimeterError = 'Perimeter_negative';
            else
                perimeterStatus = 'success';
                perimeterValue = sprintf('%.17g', perimeterCandidate);
            end
        end

        perimeterOldStatus = 'unsupported';
        perimeterOldValue = '';
        perimeterOldError = '';
        if strcmp(oldRunState, 'supported')
            if nativeId <= numel(oldProps)
                oldCandidate = oldProps(nativeId).PerimeterOld;
                if isempty(oldCandidate)
                    perimeterOldStatus = 'missing';
                elseif ~isfinite(oldCandidate)
                    perimeterOldStatus = 'error_nonfinite';
                    perimeterOldError = 'PerimeterOld_nonfinite';
                elseif oldCandidate < 0
                    perimeterOldStatus = 'error_negative';
                    perimeterOldError = 'PerimeterOld_negative';
                else
                    perimeterOldStatus = 'success';
                    perimeterOldValue = sprintf('%.17g', oldCandidate);
                end
            else
                perimeterOldStatus = 'missing';
            end
        elseif strcmp(oldRunState, 'unsupported')
            perimeterOldStatus = 'unsupported';
            perimeterOldError = oldRunError;
        else
            perimeterOldStatus = 'error_runtime';
            perimeterOldError = oldRunError;
        end

        fprintf(componentsFid, '%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n', ...
            caseRecord.id, nativeId, canonicalId, firstRow, firstCol, areaPx, ...
            perimeterValue, perimeterStatus, perimeterError, perimeterOldValue, perimeterOldStatus, perimeterOldError);
    end
end

clear cleanupFiles;

if exist(outputDir, 'dir')
    rmdir(outputDir, 's');
end
mkdir(outputDir);
movefile(metadataPath, fullfile(outputDir, 'metadata.tsv'));
movefile(casesPath, fullfile(outputDir, 'cases.tsv'));
movefile(labelsPath, fullfile(outputDir, 'labels.tsv'));
movefile(componentsPath, fullfile(outputDir, 'components.tsv'));
markerFid = fopen(fullfile(outputDir, 'completed.marker'), 'w');
if markerFid < 0
    error('Unable to create completion marker');
end
fprintf(markerFid, 'COMPLETE\n');
fclose(markerFid);
rmdir(tempDir, 's');

fprintf('Wrote perimeter reference TSV capture: %s\n', outputDir);
end

function supported = is_perimeter_old_unsupported(exception)
identifier = lower(exception.identifier);
messageText = lower(exception.message);
supported = false;
if has_token(identifier, 'regionprops') && has_token(identifier, 'unknown') && has_token(messageText, 'perimeterold')
    supported = true;
    return
end
if has_token(messageText, 'perimeterold') && has_token(messageText, 'unknown') && has_token(messageText, 'property')
    supported = true;
    return
end
if has_token(messageText, 'perimeterold') && has_token(messageText, 'not') && has_token(messageText, 'supported')
    supported = true;
end
end

function present = has_token(textValue, token)
present = ~isempty(strfind(textValue, token)); %#ok<STREMP>
end

function [topology, ascending] = compute_topology(labels, caseId)
maxLabel = max(labels(:));
if maxLabel == 0
    topology.native_id = [];
    topology.canonical_id = [];
    topology.first_row = [];
    topology.first_col = [];
    topology.area_px = [];
    ascending = true;
    return
end

present = unique(labels(labels > 0));
present = present(:)';
height = size(labels, 1);
minima = inf(size(present));
areas = zeros(size(present));
for idx = 1:numel(present)
    nativeId = present(idx);
    [rows, cols] = find(labels == nativeId);
    if isempty(rows)
        error('Case %s native label %d missing from labels', caseId, nativeId);
    end
    flat = rows + (cols - 1) * height;
    minima(idx) = min(flat);
    areas(idx) = numel(rows);
end

[sortedMinima, order] = sort(minima, 'ascend'); %#ok<ASGLU>
canonical = zeros(size(present));
for idx = 1:numel(order)
    canonical(order(idx)) = idx;
end

firstRow = zeros(size(present));
firstCol = zeros(size(present));
for idx = 1:numel(present)
    flat = minima(idx);
    firstCol(idx) = floor((flat - 1) / height) + 1;
    firstRow(idx) = flat - (firstCol(idx) - 1) * height;
end

ascending = true;
for idx = 2:numel(order)
    if minima(order(idx)) <= minima(order(idx - 1))
        ascending = false;
        break
    end
end

topology.native_id = present;
topology.canonical_id = canonical;
topology.first_row = firstRow;
topology.first_col = firstCol;
topology.area_px = areas;
end

function close_files(metadataFid, casesFid, labelsFid, componentsFid)
if metadataFid >= 0
    fclose(metadataFid);
end
if casesFid >= 0
    fclose(casesFid);
end
if labelsFid >= 0
    fclose(labelsFid);
end
if componentsFid >= 0
    fclose(componentsFid);
end
end
