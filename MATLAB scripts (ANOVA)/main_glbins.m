T = readtable('All_features_uncorrected_dropped246_with_metadata.csv');
allGL = unique(T.GLbins);
featureCols = setdiff(T.Properties.VariableNames, {'PatientName','Model','GLbins','Wavelength','Reconstruction'});
numFeatures = numel(featureCols);
numFactors = 3;
factorNames = {'Model','Wavelength','Reconstruction'};

for g = 1:numel(allGL)
    gl = allGL(g);
    T_gl = T(T.GLbins == gl, :);
    SS_table = zeros(numFeatures, numFactors + 1);  % last col is residual

    for i = 1:numFeatures
        y = T_gl.(featureCols{i});
        factors = {T_gl.Model, T_gl.Wavelength, T_gl.Reconstruction};

        [intFactorSSE, intFactorLabels] = manualAnova2(y, factors, factorNames);

        % Flatten factor names
        flatNames = strings(size(intFactorLabels));
        for k = 1:numel(intFactorLabels)
            entry = intFactorLabels{k};
            if isempty(entry)
                flatNames(k) = "";
            elseif iscell(entry) && numel(entry) == 1
                flatNames(k) = string(entry{1});
            elseif ischar(entry)
                flatNames(k) = string(entry);
            elseif iscell(entry) && all(cellfun(@ischar, entry))
                flatNames(k) = strjoin(entry, 'x');
            end
        end

        % Normalize and extract main effects
        totalSS = sum(intFactorSSE);
        eta2_all = intFactorSSE / totalSS;

        eta2_main = zeros(1, numFactors);
        for j = 1:numFactors
            idx = find(flatNames == factorNames{j}, 1);
            if ~isempty(idx)
                eta2_main(j) = eta2_all(idx);
            end
        end
        residual = 1 - sum(eta2_main);
        SS_table(i, :) = [eta2_main, residual];
    end

    % Export per GLbins
    df = array2table(SS_table, ...
        'VariableNames', [factorNames, {'Residual'}], ...
        'RowNames', featureCols);
    writetable(df, sprintf('anova_eta2_glbins_%d.csv', gl), 'WriteRowNames', true);
end
