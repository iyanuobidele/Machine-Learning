function [Model, first_test_sample, second_test_sample] = ...
    fg_return_model(First_sample, Second_sample, size1, size2, percent, l1,l2)
% percent should be in fraction. e.g 0.75 = 75%

First_sample = cell2mat(First_sample);
Second_sample = cell2mat(Second_sample);

first_training_sample = First_sample((1:floor(size1 * percent)),:); % 75%
first_test_sample = First_sample((floor(size1 * percent)+1:size1),:); % 25%

second_training_sample = Second_sample((1:floor(size2 * percent)),:); % 75%
second_test_sample = Second_sample((floor(size2 * percent)+1:size2),:); % 25%

% Classifier
% Combine Training samples
F_B = [first_training_sample;second_training_sample];
% Generate Labels
F_B_labels = [repmat({l1},1,floor(size1 * percent)),repmat({l2},1,...
    floor(size2 * percent))];

% Generate SVM Model
Model = fitcsvm(F_B,F_B_labels);