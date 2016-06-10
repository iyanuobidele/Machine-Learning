function label = return_label(filename,who)

a = fileread(filename);         % read file
c = strsplit(a,'|');            % Split file by '|'
d_index = strmatch(who,c);      % return index of e.g 'dog walker'
[~,label] = strtok(c(d_index));  % get the label
label = strtrim(label);         % remove any leading whitespace
