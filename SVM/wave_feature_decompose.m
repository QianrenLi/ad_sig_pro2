function   vector= wave_feature_decompose(samples,filtered_signal)

peak_location = length(samples)/3 + 1:length(samples)/3*2;
peak_interphase = diff(samples(peak_location));
xp = peak_interphase;
xp(end) = [];
xm = peak_interphase;
xm(1) = [];
SD1 = std(xp-xm)/sqrt(2);
SD2 = std(xp+xm)/sqrt(2);

% Require additional toolbox
% if length(peak_interphase) <= 2
%     appen = - inf;
% else

    appen = approximateEntropy(peak_interphase);
% end

wave_duration_mean = mean(peak_interphase);
wave_duration_var = var(peak_interphase);
wave_amplitude_mean = mean(abs(filtered_signal(peak_location)));
wave_amplitude_var = var((filtered_signal(peak_location)));

peak_location_pre = 1:length(samples)/3;
wave_width = samples(peak_location_pre + length(samples)/3*2) - samples(peak_location_pre);
wave_width_mean = mean(wave_width);
wave_width_var = var(wave_width);

wave_half_width_1 = samples(peak_location_pre + length(samples)/3*2) - samples(peak_location);
wave_half_width_mean_1 = mean(wave_half_width_1);
wave_half_width_var_1 = var(wave_half_width_1);

wave_half_width_2 =  samples(peak_location) - samples(peak_location_pre);
wave_half_width_mean_2 = mean(wave_half_width_2);
wave_half_width_var_2 = var(wave_half_width_2);

vector = [SD1,SD2,appen,wave_duration_mean,wave_duration_var,wave_amplitude_mean,wave_amplitude_var,wave_width_mean,wave_width_var, ...
wave_half_width_mean_1,wave_half_width_var_1,wave_half_width_mean_2,wave_half_width_var_2];
