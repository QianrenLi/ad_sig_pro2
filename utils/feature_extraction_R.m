%%
load("data.mat")

% Find R wave 
index = 100;
train_data = train_datas(index,:);
points = 2400;
fs = 300; 
tm = 1:length(train_data);

%% Method A: Not Good 
% figure
% plot(train_data)
% figure
% ecgsig = train_data/max(train_data);
% [c,l] = wavedec(ecgsig,6,'sym4');
% approx=appcoef(c,l,'db2');
% [cd1,cd2,cd3]=detcoef(c,l,[4,5,6]);
% subplot(2,2,1)
% plot(approx)
% title('Approximation Coefficients')
% subplot(2,2,2)
% plot(cd3)
% title('Level 3 Detail Coefficients')
% subplot(2,2,3)
% plot(cd2)
% title('Level 2 Detail Coefficients')
% subplot(2,2,4)
% plot(cd1)
% title('Level 1 Detail Coefficients')
% x=waverec(c,l,'db2');
% figure
% plot(x)

%% Method B: Works in "clean data" -- E.g index = 1
base_equation = 'db5';
ecgsig = normalize(train_data);
wt = modwt(ecgsig,8,base_equation);



wtrec = zeros(size(wt));
w_index = 4;

wtrec(w_index:w_index+1,:) = wt(w_index:w_index+1,:);
y = imodwt(wtrec,base_equation);
y = abs(y).^2;
y = normalize(y);
diff_y = diff(y);
figure
plot(diff_y,'r')
hold on
plot(y,'b')

[qrspeaks,locs] = findpeaks(y,tm,'MinPeakHeight',std(y),'MinPeakDistance',0.4 * fs);
% [qrspeaks,locs] = findpeaks(y,tm);
figure
plot(tm,y)
hold on
plot(locs,qrspeaks,'ro')
plot(tm,ecgsig,'k--')
plot(locs,ecgsig(locs),'bo')
legend("Reconstruct Signal","R peak","Real Signal","R Peak in Real Signal")
xlabel('Seconds')



