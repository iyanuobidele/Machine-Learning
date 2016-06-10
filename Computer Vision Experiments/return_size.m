function counter = return_size(foldername,label,who)
% Get size of label samples in PortlandDogWalking folder
labelFiles = dir(fullfile(foldername, '*.labl'));
counter = 0;
for k = 1 : length(labelFiles)
    baseLabelName = labelFiles(k).name;
    fullLabelName = fullfile(foldername, baseLabelName);
    if strcmp(return_label(fullLabelName,who),label)
       counter = counter + 1;
    end
end