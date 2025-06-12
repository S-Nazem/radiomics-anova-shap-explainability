% main.m — Full Factorial ANOVA η² Plot

% Step 1: Load data
T = readtable('All_features_uncorrected_with_metadata.csv');  % <- Replace with your actual file name

% Step 2: Identify relevant columns
factorNames = {'Model', 'GLbins', 'Wavelength', 'Reconstruction'};
allVars = T.Properties.VariableNames;
featureCols = setdiff(allVars, [{'PatientName'}, factorNames]);  % 93 features
numFeatures = numel(featureCols);
numFactors = numel(factorNames);

% Step 3: Setup
SS_table = zeros(numFeatures, numFactors + 1);  % last col = residual
rowNames = featureCols;

% Step 4: Loop over each radiomic feature
for i = 1:numFeatures
    y = T.(featureCols{i});
    factors = {T.Model, T.GLbins, T.Wavelength, T.Reconstruction};

    [intFactorSSE, intFactorLabels] = manualAnova2(y, factors, factorNames);

    % Flatten labels
    flatNames = strings(size(intFactorLabels));
    for k = 1:numel(intFactorLabels)
        entry = intFactorLabels{k};
        if isempty(entry)
            flatNames(k) = "";
        elseif iscell(entry) && numel(entry) == 1
            flatNames(k) = string(entry{1});
        elseif ischar(entry)
            flatNames(k) = string(entry);
        elseif isstring(entry)
            flatNames(k) = entry;
        elseif iscell(entry) && all(cellfun(@ischar, entry))
            flatNames(k) = strjoin(entry, 'x');
        else
            flatNames(k) = "";  % fallback
        end
    end
    
    % Normalize to get η²
    totalSS = sum(intFactorSSE);
    eta2_all = intFactorSSE / totalSS;
    
    % Extract main effects only
    eta2_main = zeros(1, numFactors);
    for j = 1:numFactors
        idx = find(flatNames == factorNames{j}, 1);
        if ~isempty(idx)
            eta2_main(j) = eta2_all(idx);
        end
    end
    
    % Compute residual
    residual = 1 - sum(eta2_main);
    
    % Save to result matrix
    SS_table(i, :) = [eta2_main, residual];

end

% Step 5: Save to CSV
eta2_table = array2table(SS_table, ...
    'VariableNames', [factorNames, {'Residual'}], ...
    'RowNames', rowNames);

writetable(eta2_table, 'anova_eta2_output.csv', 'WriteRowNames', true);
disp('✅ Saved: anova_eta2_output.csv');
