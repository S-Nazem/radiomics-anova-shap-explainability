% main5fold.m — Stratified 5-fold η² for Skewness with balanced PDX exclusion

% Load data
T = readtable('All_features_uncorrected_dropped246_with_metadata.csv');
feature = 'original_firstorder_Kurtosis';
factors = {'Model','GLbins','Wavelength','Reconstruction'};
numFolds = 5;

% Get all PDX IDs per class
luminalPDXs = unique(T.PatientName(strcmp(T.Model, 'Luminal')));
basalPDXs   = unique(T.PatientName(strcmp(T.Model, 'Basal')));

assert(numel(luminalPDXs) >= 11 && numel(basalPDXs) >= 10, 'Check dataset balance');

rng(42);  % reproducibility
eta2_all_folds = zeros(numFolds, numel(factors)+1);  % +1 for residual

for fold = 1:numFolds
    % Randomly exclude 3 Luminal and 2 Basal PDXs
    excl_lum = randsample(luminalPDXs, 3);
    excl_bas = randsample(basalPDXs, 2);
    excl_all = [excl_lum; excl_bas];

    % Keep only rows with remaining 8+8 models
    keep_idx = ~ismember(T.PatientName, excl_all);
    T_fold = T(keep_idx, :);

    % Response + factor setup
    y = T_fold.(feature);
    f = {T_fold.Model, T_fold.GLbins, T_fold.Wavelength, T_fold.Reconstruction};

    [ss, labels] = manualAnova2(y, f, factors);

    % Flatten labels
    flatNames = strings(size(labels));
    for i = 1:numel(labels)
        entry = labels{i};
        if isempty(entry)
            flatNames(i) = "";
        elseif iscell(entry) && numel(entry) == 1
            flatNames(i) = string(entry{1});
        elseif ischar(entry)
            flatNames(i) = string(entry);
        elseif isstring(entry)
            flatNames(i) = entry;
        elseif iscell(entry) && all(cellfun(@ischar, entry))
            flatNames(i) = strjoin(entry, 'x');
        end
    end

    % Compute η²
    totalSS = sum(ss);
    eta2_all = ss / totalSS;

    % Extract main effect η²s
    eta2_main = zeros(1, numel(factors));
    for j = 1:numel(factors)
        idxj = find(flatNames == factors{j}, 1);
        if ~isempty(idxj)
            eta2_main(j) = eta2_all(idxj);
        end
    end

    % Residual
    residual = 1 - sum(eta2_main);
    eta2_all_folds(fold, :) = [eta2_main, residual];
end

% Save fold-by-fold η² and summary stats to CSV
foldTable = array2table(eta2_all_folds, ...
    'VariableNames', [factors, {'Residual'}], ...
    'RowNames', strcat("Fold", string(1:numFolds)));

summaryTable = table( ...
    mean_eta2', std_eta2', cov_eta2', ...
    'VariableNames', {'Mean', 'Std', 'CoV'}, ...
    'RowNames', [factors, {'Residual'}]);

% Write to CSV
writetable(foldTable, 'eta2_folds_kurtosis.csv', 'WriteRowNames', true);
writetable(summaryTable, 'eta2_summary_kurtosis.csv', 'WriteRowNames', true);
disp('✅ Saved: eta2_folds_skewness.csv and eta2_summary_skewness.csv');



