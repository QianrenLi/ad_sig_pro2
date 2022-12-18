function data_padded = data_pad(data_raw,obj_len)
data_len = size(data_raw,2);
pad_len = obj_len - data_len;
b = zeros(size(data_raw,1), pad_len);
data_padded = [data_raw b];
end