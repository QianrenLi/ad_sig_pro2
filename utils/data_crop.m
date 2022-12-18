function data_cropped = data_crop(data_raw,obj_len)
data_len = size(data_raw,2);
a = randi(data_len - obj_len);
b = a + obj_len-1;
data_cropped = data_raw(a:b);
end