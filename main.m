clear;

% Variables generales.
sujeto = {'S1', 'S2', 'S3', 'S4', 'S5', 'S5', 'S6'};
num_files = 5;
files = 1:num_files;
train_files = 2;
tam_ventana = 10;
desp = 0.05;
label_open = 1;
label_closed = 0;
folds = nchoosek(1:num_files, train_files); % Train/Test folds.
rng(28);


%%%%%-- PARAMETERS TO CONFIGURE --%%%%%%

%Select EEG device.
eeg_device = 'PP';

% Select classifiers.
lda_one_sensor = 1;
svm_one_sensor = 1;
logreg_one_sensor = 1;
tree_one_sensor = 1;
lda_classification = 1;
svm_classification = 1;
logreg_classification = 1;
tree_classification = 1;

% Feature extraction methods parameteres.
met = 1;
filt = 1;

% Fs Device selection
fs = 200;


%%%%%-- END PARAMETERS TO CONFIGURE --%%%%%%

for s = 1:length(sujeto)
    % Read files.
    [open, closed] = readFiles(sujeto{s}, eeg_device, filt);
    
    for k = 1:size(folds,1)
        
        % Concatenate data from both classes.
        tr_open = zeros(0,size(open,3));
        tr_closed = zeros(0, size(open,3));
        test_open = zeros(0,size(open,3));
        test_closed = zeros(0, size(open,3));
        
        for hh = 1:num_files
            if hh == folds(k,1) || hh == folds(k,2)
                tr_open = [tr_open;squeeze(open(hh,:,:))];
                tr_closed = [tr_closed;squeeze(closed(hh,:,:))];
            else
                test_open = [test_open;squeeze(open(hh,:,:))];
                test_closed = [test_closed;squeeze(closed(hh,:,:))];
            end
        end
 
        [train_idx_cl, train_idx_op] = featuresDeslizantes(tr_closed, ...
            tr_open, desp, fs, tam_ventana, met);
        [test_idx_cl, test_idx_op] = featuresDeslizantes(test_closed, ...
            test_open, desp, fs, tam_ventana, met);
  
        % Train and test sets.
        trainData = [train_idx_cl; train_idx_op];
        trainLabel = [zeros(size(train_idx_cl,1),1); ones(size(train_idx_op,1),1)];
        testData = [test_idx_cl; test_idx_op];
        testLabel = [zeros(size(test_idx_cl,1),1); ones(size(test_idx_op,1),1)];
                
        %% One Sensor.
        if lda_one_sensor == 1
            % Find optimal parameters.
            fit_ldamodel_ch1 = optimize_model(trainData(:,1), trainLabel, 'lda');
            fit_ldamodel_ch2 = optimize_model(trainData(:,2), trainLabel, 'lda');
            
            % Train the model with the optimal hyperparameters (C and Scale).
            ldamodel_ch1 = fitcdiscr(trainData(:,1), trainLabel,'DiscrimType','linear',...
                'Gamma',fit_ldamodel_ch1.HyperparameterOptimizationResults.XAtMinObjective.Gamma,...
                'Delta',fit_ldamodel_ch1.HyperparameterOptimizationResults.XAtMinObjective.Delta);
            
            ldamodel_ch2 = fitcdiscr(trainData(:,2), trainLabel,'DiscrimType','linear',...
                'Gamma',fit_ldamodel_ch2.HyperparameterOptimizationResults.XAtMinObjective.Gamma,...
                'Delta',fit_ldamodel_ch2.HyperparameterOptimizationResults.XAtMinObjective.Delta);
            
            % Acc LDA
            [p_lda_ch1, ~] = predict(ldamodel_ch1, testData(:,1));
            [p_lda_ch2, ~] = predict(ldamodel_ch2, testData(:,2));
            
            acc_op_lda_ch1(s,k) = evaluate_accuracy(p_lda_ch1, testLabel, label_open);
            acc_cl_lda_ch1(s,k) = evaluate_accuracy(p_lda_ch1, testLabel, label_closed);
            
            acc_op_lda_ch2(s,k) = evaluate_accuracy(p_lda_ch2, testLabel, label_open);
            acc_cl_lda_ch2(s,k) = evaluate_accuracy(p_lda_ch2, testLabel, label_closed);
        end
        
        % New 2021: SVM One Channel
        if svm_one_sensor == 1
            % Find optimal parameters.
            fit_svmmodel_ch1 = optimize_model(trainData(:,1), trainLabel, 'svm');
            fit_svmmodel_ch2 = optimize_model(trainData(:,2), trainLabel, 'svm');
            
            % Train the model with the optimal hyperparameters (C and Scale).
            svmmodel_ch1 = fitcsvm(trainData(:,1), trainLabel,'KernelFunction','linear',...
                'BoxConstraint',fit_svmmodel_ch1.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint,...
                'KernelScale',fit_svmmodel_ch1.HyperparameterOptimizationResults.XAtMinObjective.KernelScale, ...
                'Standardize' , true);
            
            svmmodel_ch2 = fitcsvm(trainData(:,2), trainLabel,'KernelFunction','linear',...
                'BoxConstraint',fit_svmmodel_ch2.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint,...
                'KernelScale',fit_svmmodel_ch2.HyperparameterOptimizationResults.XAtMinObjective.KernelScale, ...
                'Standardize' , true);
            
            % Acc SVM.
            [p_svm_ch1, ~] = predict(svmmodel_ch1, testData(:,1));
            [p_svm_ch2, ~] = predict(svmmodel_ch2, testData(:,2));
            
            acc_op_svm_ch1(s,k) = evaluate_accuracy(p_svm_ch1, testLabel, label_open);
            acc_cl_svm_ch1(s,k) = evaluate_accuracy(p_svm_ch1, testLabel, label_closed);
            
            acc_op_svm_ch2(s,k) = evaluate_accuracy(p_svm_ch2, testLabel, label_open);
            acc_cl_svm_ch2(s,k) = evaluate_accuracy(p_svm_ch2, testLabel, label_closed);
        end
        
        if logreg_one_sensor == 1
             % Find optimal parameters.
            logregmodel_ch1 = fitglm(trainData(:,1), trainLabel, 'Distribution', 'binomial', 'link', 'logit');
            logregmodel_ch2 = fitglm(trainData(:,2), trainLabel, 'Distribution', 'binomial', 'link', 'logit');
           
            % Acc LogReg
            [p_logreg_ch1, ~] = predict(logregmodel_ch1, testData(:,1));
            p_logreg_ch1 = round(p_logreg_ch1);
            [p_logreg_ch2, ~] = predict(logregmodel_ch2, testData(:,2));
            p_logreg_ch2 = round(p_logreg_ch2);
            
            acc_op_logreg_ch1(s,k) = evaluate_accuracy(p_logreg_ch1, testLabel, label_open);
            acc_cl_logreg_ch1(s,k) = evaluate_accuracy(p_logreg_ch1, testLabel, label_closed);
            
            acc_op_logreg_ch2(s,k) = evaluate_accuracy(p_logreg_ch2, testLabel, label_open);
            acc_cl_logreg_ch2(s,k) = evaluate_accuracy(p_logreg_ch2, testLabel, label_closed);
        end
        
        if tree_one_sensor == 1
            % Optimize Hyperparameters.
            fit_treemodel_ch1 = optimize_model(trainData(:,1), trainLabel, 'tree');
            fit_treemodel_ch2 = optimize_model(trainData(:,2), trainLabel, 'tree');
            
            % Train the models with the best hyperparameters.
            treemodel_ch1 = fitctree(trainData(:,1), trainLabel, ...
                'MinLeafSize', fit_treemodel_ch1.HyperparameterOptimizationResults.XAtMinObjective.MinLeafSize);
            
            treemodel_ch2 = fitctree(trainData(:,2), trainLabel, ...
                'MinLeafSize', fit_treemodel_ch2.HyperparameterOptimizationResults.XAtMinObjective.MinLeafSize);
            
            % Acc Classification trees.
            [p_tree_ch1, ~] = predict(treemodel_ch1, testData(:,1));
            [p_tree_ch2, ~] = predict(treemodel_ch2, testData(:,2));
            
            acc_op_tree_ch1(s,k) = evaluate_accuracy(p_tree_ch1, testLabel, label_open);
            acc_cl_tree_ch1(s,k) = evaluate_accuracy(p_tree_ch1, testLabel, label_closed);
            
            acc_op_tree_ch2(s,k) = evaluate_accuracy(p_tree_ch2, testLabel, label_open);
            acc_cl_tree_ch2(s,k) = evaluate_accuracy(p_tree_ch2, testLabel, label_closed);
        end
        
        %% Two Sensors
        
        % LDA
        if lda_classification == 1
            fit_ldamodel = optimize_model(trainData, trainLabel, 'lda');
            
            % Train the model with the optimal hyperparameters (C and Scale).
            ldamodel = fitcdiscr(trainData, trainLabel,'DiscrimType','linear',...
                'Gamma',fit_ldamodel.HyperparameterOptimizationResults.XAtMinObjective.Gamma,...
                'Delta',fit_ldamodel.HyperparameterOptimizationResults.XAtMinObjective.Delta);
            
            % Acc LDA
            [p_lda, ~] = predict(ldamodel, testData);
            acc_op_lda(s,k) = evaluate_accuracy(p_lda, testLabel, label_open);
            acc_cl_lda(s,k) = evaluate_accuracy(p_lda, testLabel, label_closed);
            
        end
        
        % SVM
        if svm_classification == 1
            % Search for optimal hyperparameters according to train/validation.
            fit_svmmodel = optimize_model(trainData, trainLabel, 'svm');
            
            % Train the model with the optimal hyperparameters (C and Scale).
            svmmodel = fitcsvm(trainData, trainLabel,'KernelFunction','linear',...
                'BoxConstraint',fit_svmmodel.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint,...
                'KernelScale',fit_svmmodel.HyperparameterOptimizationResults.XAtMinObjective.KernelScale, ...
                'Standardize' , true);
            
            % Acc SVM.
            [p_svm, ~] = predict(svmmodel, testData);
            acc_op_svm(s,k) = evaluate_accuracy(p_svm, testLabel, label_open);
            acc_cl_svm(s,k) = evaluate_accuracy(p_svm, testLabel, label_closed);
        end
        
        if logreg_classification == 1
            % Train the model.
            logregmodel = fitglm(trainData, trainLabel, 'Distribution', 'binomial', 'link', 'logit');
            
            % Acc Logistic Regression
            [p_logreg, ~] = predict(logregmodel, testData);
            p_logreg = round(p_logreg);
            acc_op_logreg(s,k) = evaluate_accuracy(p_logreg, testLabel, label_open);
            acc_cl_logreg(s,k) = evaluate_accuracy(p_logreg, testLabel, label_closed);
        end
        
        if tree_classification == 1
            % Train the model.
            fit_model = optimize_model(trainData, trainLabel, 'tree');
            
            treemodel = fitctree(trainData, trainLabel, ...
                'MinLeafSize', fit_model.HyperparameterOptimizationResults.XAtMinObjective.MinLeafSize);
            
            % Acc Logistic Regression
            [p_tree_ch1, ~] = predict(treemodel, testData);
            acc_op_tree(s,k) = evaluate_accuracy(p_tree_ch1, testLabel, label_open);
            acc_cl_tree(s,k) = evaluate_accuracy(p_tree_ch1, testLabel, label_closed);
        end
    end
    
end
