% clear all;
% close all;
% clear memory;
% clc;

data=1;
if(data==1)
    training_file='Twod.tra';
    N=8;
    M=7;
end
if(data==2)
    training_file='oh7.tra';
    N=20;
    M=3;
end
if(data==3)
    training_file='SINGLE2.tra';
    N=16;
    M=3;
end
if(data==4)
    training_file='Power12trn.dat';
    N=12;
    M=1;
end
if(data==5)
    training_file='mattrn.dat';
    N=4;
    M=4;
end
if(data==6)
    training_file='concrete.tra';
    N=8;
    M=1;
end
if(data==7)
    training_file='forest.tra';
    N=12;
    M=1;
end
if(data==8)
    training_file='housing.tra';
    N=13;
    M=1;
end
if(data==9)
    training_file='redwine.tra';
    N=11;
    M=1;
end
if(data==10)
    training_file='F17C.DAT';
    N=17;
    M=39;
end

Nh = 10;
Nit =100;
% The following code reads a text file and stores all the paterns in
% an Nv by (N+M) matrix
fid = fopen(training_file, 'r');
training_file_values = fscanf(fid, '%f');
fclose(fid);
Nv = numel(training_file_values)/(N+M);
fprintf('Nv = %d\n', Nv);
training_file_values = reshape(training_file_values, [(N+M) Nv])';

% Store the inputs in variable x and the outputs in variable t
x = training_file_values(:, 1:N);
t = training_file_values(:, N+1:N+M);
clear training_file_values;

mx=mean(x);
%making input vector of zero mean
for n=1:N
    x(:,n)=x(:,n)-mx(n);
end
%cheking whether the input mean is zero
mx=mean(x);
fprintf('the mean of the inputs over all the paterns are:');
disp(mx);
%randomizing input weights
Wih=mlp_randn(Nh,N+1);
%computing net and output for hidden units
x = [ones(Nv,1) x];
net=x*Wih';
%calculating mean of net function
hm=mean(net);
%calculation of std of net function
hv=std(net,1);
%net control
hmean=0.5;
hvar=1.0;
Wih = Wih * hvar./ repmat(hv', [1 N+1]);
Wih(:,1) = Wih(:,1) + hmean - hm' .* hvar./ hv';
%computing net and output for hidden units again
net=x*Wih';
O=act(net);
%cheking the mean and std of net function
%calculating mean of net function
hm=mean(net);
%calculation of std of net function
hv=std(net,1);
fprintf('the standard deviations of net function after net control are:');
disp(hv);
fprintf('the mean of net function after net control are:');
disp(hm);
Nu=N+Nh+1;
%% OWO
net=x*Wih';
O=act(net);
x1(:,1:N+1)= x;
x1(:,N+2:N+1+Nh)= O;
[R,C]=autocross(x1,t,Nv);

%performing OWO
[wo] = OLS(R,C,M);
Wo=wo';
Woi=Wo(:,1:N+1);
Woh=Wo(:,N+2:Nu);

%calculation of output
y=x1*Wo';
E=t-y;
MSE(1)=sum(sum(E.*E))/Nv;

%Multiplies
Nu=N+Nh+1;
Nw=M*Nu+(N+1)*Nh;
Mlm=Nv*(M*Nu+2*Nh*(N+1)+M*(N+6*Nh+4)+M*Nu*(Nu+3*Nh*(N+1))+4*(Nh^2)*((N+1)^2))+Nw^3+Nw^2;
Multi=0;

x=x(:,2:N+1);
%computing net and output for hidden units
trainFcn = 'trainlm';
net=cascadeforwardnet(Nh,trainFcn);
net=configure(net,x',t');
net.IW{1}=Wih(:,2:N+1);
net.b{1}=Wih(:,1);
net.IW{2}=Woi(:,2:N+1);
net.LW{2}=Woh;
net.b{2}=Woi(:,1);
%net.input.processFcns = { }; % Remove normalization
%net.output.processFcns= { };
net.divideParam.trainRatio = 1;
net.divideParam.testRatio = 0;
net.divideParam.valRatio = 0;
net.trainParam.min_grad=0;
net.trainParam.goal=0;
net.trainParam.max_fail=200;
net.trainParam.epochs=1;
net.performFcn = 'mse';
net.layers{1}.transferFcn = 'logsig';
for it=1:Nit
    [net, tr]=train(net,x',t');
    if it==1
        net.IW{1}=Wih(:,2:N+1);
        net.b{1}=Wih(:,1);
        net.IW{2}=Woi(:,2:N+1);
        net.LW{2}=Woh;
        net.b{2}=Woi(:,1);
    end
    MSE(it)=(1/Nv)*sumsqr(net(x')-t');
    Multi=Multi+Mlm;
    Mult(it)=Multi;
end
it=1:1:Nit;
plot(it,MSE(it));
hold on



