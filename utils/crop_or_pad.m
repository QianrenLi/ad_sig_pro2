function data_out = crop_or_pad(data_raw,obj_len)
data_len = size(data_raw,2);
if data_len < obj_len
    data_out = data_pad(data_raw,obj_len);
else
    data_out = data_crop(data_raw,obj_len);
end
end