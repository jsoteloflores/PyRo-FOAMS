function foams105_moment_regionprops_reference(fixturePath, outputPath)
% Export MATLAB regionprops geometry/perimeter for Brief 16 analytical masks.
%
% This harness is optional and is not executed by repository tests. It records
% MATLAB release/toolbox metadata, four-connected labels, and regionprops
% values for later independent comparison.

if nargin < 1 || isempty(fixturePath)
    fixturePath = fullfile('main', 'tests', 'fixtures', '16_foams105_moment_cases.json');
end
if nargin < 2 || isempty(outputPath)
    outputPath = fullfile('reference-tools', 'matlab', 'foams105_moment_regionprops_reference.json');
end

raw = fileread(fixturePath);
fixture = jsondecode(raw);

imagesInfo = ver('images');
if isempty(imagesInfo)
    toolboxVersion = '';
else
    toolboxVersion = imagesInfo(1).Version;
end

result.schema_version = 1;
result.fixture_path = fixturePath;
result.generated_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
result.matlab_release = version('-release');
result.matlab_version = version;
result.image_processing_toolbox_version = toolboxVersion;
result.perimeter_old_supported = false;
result.cases = [];

for caseIndex = 1:numel(fixture.cases)
    caseData = fixture.cases(caseIndex);
    mask = false(caseData.shape(1), caseData.shape(2));
    if isfield(caseData, 'pixels') && ~isempty(caseData.pixels)
        rows = caseData.pixels(:, 1) + 1;
        cols = caseData.pixels(:, 2) + 1;
        linear = sub2ind(caseData.shape, rows, cols);
        mask(linear) = true;
    end

    labels = bwlabel(mask, 4);
    labels = reorder_column_major(labels);

    props = regionprops(labels, ...
        'Area', 'MajorAxisLength', 'MinorAxisLength', ...
        'Eccentricity', 'Orientation', 'Perimeter');

    caseResult.id = caseData.id;
    caseResult.shape = caseData.shape;
    caseResult.foreground_pixel_count = nnz(mask);
    caseResult.component_count = numel(props);
    caseResult.components = [];

    for componentIndex = 1:numel(props)
        component.id = componentIndex;
        component.area = props(componentIndex).Area;
        component.major_axis_length = props(componentIndex).MajorAxisLength;
        component.minor_axis_length = props(componentIndex).MinorAxisLength;
        component.eccentricity = props(componentIndex).Eccentricity;
        component.orientation = props(componentIndex).Orientation;
        component.perimeter = props(componentIndex).Perimeter;

        perimOldValue = [];
        try
            oldProps = regionprops(labels == componentIndex, 'PerimeterOld');
            if ~isempty(oldProps)
                perimOldValue = oldProps(1).PerimeterOld;
                result.perimeter_old_supported = true;
            end
        catch
            perimOldValue = [];
        end
        component.perimeter_old = perimOldValue;
        caseResult.components = [caseResult.components; component]; %#ok<AGROW>
    end

    result.cases = [result.cases; caseResult]; %#ok<AGROW>
end

json = jsonencode(result, 'PrettyPrint', true);
outdir = fileparts(outputPath);
if ~isempty(outdir) && ~exist(outdir, 'dir')
    mkdir(outdir);
end
fid = fopen(outputPath, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', json);
fprintf('Wrote regionprops reference: %s\n', outputPath);
end

function reordered = reorder_column_major(labels)
maxLabel = max(labels(:));
if maxLabel == 0
    reordered = zeros(size(labels), 'like', labels);
    return
end

flat = labels(:);
positions = find(flat > 0);
values = flat(positions);
minimum = inf(maxLabel, 1);
for index = 1:numel(positions)
    label = values(index);
    if positions(index) < minimum(label)
        minimum(label) = positions(index);
    end
end
[~, order] = sort(minimum, 'ascend');
lookup = zeros(maxLabel + 1, 1);
for index = 1:numel(order)
    lookup(order(index) + 1) = index;
end
reordered = lookup(labels + 1);
reordered = reshape(reordered, size(labels));
reordered = cast(reordered, 'like', labels);
end
