initCuda();
installDebug();

a = cuda(rand(10,10));
b = cuda(rand(10,10));

%% basic computing

q=-a;
q=-b;
q=a+b;
q=a-b;
q=a.*b;
q=a./b;
q=a*b;
q=a.*b;

%% reduce operations
q=sum(a);
q=max(a);
q=min(a);

q=sum(a,1);
q=sum(a,2);

q=max(a,0.5);
q=min(a,0.5);

%% referencing

q=a(:);
q=a(1:end);
q=a(:,:);
q=a(1:end,:);
q=a(:,1:end);
q=a(2:8,:);
q=a(:,2:8);

%% FFTs
q=fft(a);

