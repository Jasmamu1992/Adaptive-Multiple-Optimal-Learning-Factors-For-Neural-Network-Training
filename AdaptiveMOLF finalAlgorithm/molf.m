clear all;
clear memory;
clc;
% training_file ='SINGLE2.tra';% input('Enter the training file name: ', 's');
% N =16;% input('Enter the number of inputs (N): ');
% M =3;% input('Enter the number of outputs (M): ');

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

Nh = 10;%input('Enter the number of hidden units (Nh): ');
Nit =100;% input('Enter the number of iterations (Ni): ');

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

%miltiplies
Nu=N+1+Nh;
Mowo_bp=Nv*(Nh*(M+2*N+3)+M*(2*Nu+1))+Nu*(Nu+1)*(M+(2*Nu+1)/6+Nv/2+3/2);
Mhwo= (N+1)*(N+2)*(Nh+(2*N+3)/6+Nv/2+3/2);
Mmolf= Nv*Nh*(2*M+N+2+(Nh+1)*M/2)+Nh*(Nh+1)*((2*Nh+1)/6+5/2);
Mmolf_wupdate=Nh*(N+1);
M_OWO_HWO_MOLF=Mowo_bp+Mhwo+Mmolf+Mmolf_wupdate;

%% net control
%findng mean of input vector
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
Nu  =N+1+Nh;
fprintf('it\t\t\t\tMSE\n');
Mult=0;
Multi=zeros(Nit,1);
for it=1:Nit
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
    MSE(it)=sum(sum(E.*E))/Nv;
    fprintf('%d\t\t\t%f\n',it,MSE(it));
    %calculation of input weight gradient
    diffnet=O.*(1-O);
    do=2*(t-y);
    d=diffnet.*(do*Woh(:,1:Nh));
    G=d'*x/Nv;
    doy=zeros(Nv,M,Nh);
    
    %HWO
    Ro=(x'*x)/Nv;
    G1=G';
    e=OLS(Ro,G1,Nh);
    e=e';
    
    %% calculation of MOLF
    for k=1:Nh
        doy(:,:,k)=((x*e(k,:)').*diffnet(:,k))*Woh(:,k)';
    end
    H=zeros(Nh,Nh);
    for a=1:Nh
        for b=1:Nh
            H(a,b)=sum(sum(doy(:,:,a).*doy(:,:,b)))*2/Nv;
        end
    end
    g=zeros(Nh,1);
    for k=1:Nh
        g(k)=sum(sum((t-y).*(doy(:,:,k))))*2/Nv;
    end
    [z] = OLS(H,g,1);
    %updating weights
    Wih=Wih+diag(z)*e;
    Mult=Mult+M_OWO_HWO_MOLF;
    Multi(it)=Mult;
end
it=[1:1:Nit];
plot(log(Multi(it)),MSE(it),'--k');
% plot(it,MSE(it),'--k');
hold on




