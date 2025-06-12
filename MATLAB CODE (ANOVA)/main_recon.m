T = readtable('All_features_uncorrected_with_metadata.csv');

allRecon = unique(T.Reconstruction);  % {'FBP', 'MB'}
featureCols = setdiff(T.Properties.VariableNames, {'PatientName','Model','GLbins','Wavelength','Reconstruction'});
numFeatures = numel(featureCols);
numFactors = 3;
factorNames = {'Model','GLbins','Wavelength'};

for r = 1:numel(allRecon)
    reconType = allRecon{r};
    T_r = T(strcmp(T.Reconstruction, reconType), :);
    SS_table = zeros(numFeatures, numFactors + 1);  % +1 for Residual

    for i = 1:numFeatures
        y = T_r.(featureCols{i});
        factors = {T_r.Model, T_r.GLbins, T_r.Wavelength};

        [intFactorSSE, intFactorNames] = manualAnova2(y, factors, factorNames);

        % Flatten factor names
        flatNames = strings(size(intFactorNames));
        for k = 1:numel(intFactorNames)
            entry = intFactorNames{k};
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

        % Normalize to η²
        totalSS = sum(intFactorSSE);
        eta2_all = intFactorSSE / totalSS;

        % Extract only main effects
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

    % Export table
    df = array2table(SS_table, ...
        'VariableNames', [factorNames, {'Residual'}], ...
        'RowNames', featureCols);
    writetable(df, sprintf('anova_eta2_reconstruction_%s.csv', reconType), 'WriteRowNames', true);
end
