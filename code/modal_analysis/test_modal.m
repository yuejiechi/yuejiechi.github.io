% We test the ability to track frequency components


% Number of rows and columns
numr = 256;
numc = 1000;
% Rank of the underlying matrix.
truerank = 5;

maxrank = 10;%i-1;
M = 30;

f = [0.1769    0.1992    0.2116    0.6776    0.7599];
amp = [ 0.3  0.8   0.5  1  0.1];
%amp = [1 1 1 1 1];
%f = sort(rand(1,truerank));
omega = diag(exp(j*2*pi*f));
Omega = ones(numr,truerank);
for i =2:numr
    Omega(i,:) = Omega(i-1,:)*omega;
end
Omega = Omega*diag(amp);

% coefficients
coeff = randn(numc,truerank);
    
t = ones(truerank,1)*[1:numc];
figure(1);
scatter(reshape(t,numc*truerank,1),reshape(f'*ones(1,numc),numc*truerank,1),3,reshape(amp'*ones(1,numc),numc*truerank,1));
title('Ground Truth');
hold on;


I1 = zeros(M*numc,1);
% Select a random set of M entries of Y.
for it = 1:numc
    p = randperm(numr);
    I1((it-1)*M+1:it*M) = p(1:M);
end
J1 = reshape(repmat([1:numc],M,1),numc*M,1);

% Values of Y at the locations indexed by I and J.
S1 = sum(Omega(I1,:).*coeff(J1,:),2);

%

f = [0.1769    0.1992    0.4116    0.6776    0.8599];
amp = [ 0.3  0.8   0.5  1  0.1];

%f = sort(rand(1,truerank));
omega = diag(exp(j*2*pi*f));
Omega = ones(numr,truerank);
for i =2:numr
    Omega(i,:) = Omega(i-1,:)*omega;
end
Omega = Omega*diag(amp);

% coefficients
coeff = randn(numc,truerank);
    
figure(1);
t = ones(truerank,1)*[1+numc:2*numc];
scatter(reshape(t,numc*truerank,1),reshape(f'*ones(1,numc),numc*truerank,1),3,reshape(amp'*ones(1,numc),numc*truerank,1));
title('Ground Truth');
hold on;

I2 = zeros(M*numc,1);
% Select a random set of M entries of Y.
for it = 1:numc
    p = randperm(numr);
    I2((it-1)*M+1:it*M) = p(1:M);
end
J2 = reshape(repmat([1:numc],M,1),numc*M,1);

% Values of Y at the locations indexed by I and J.
S2 = sum(Omega(I2,:).*coeff(J2,:),2);


truerank = 6;
f = [0.1769    0.1992    0.4116    0.6776    0.8599  0.9513];
amp = [ 0.3  0.8   0.5  1  0.1  0.6];

%f = sort(rand(1,truerank));
omega = diag(exp(j*2*pi*f));
Omega = ones(numr,truerank);
for i =2:numr
    Omega(i,:) = Omega(i-1,:)*omega;
end
Omega = Omega*diag(amp);

% coefficients
coeff = randn(numc,truerank);
    
    
figure(1);
t = ones(truerank,1)*[1+2*numc:3*numc];
scatter(reshape(t,numc*truerank,1),reshape(f'*ones(1,numc),numc*truerank,1),3,reshape(amp'*ones(1,numc),numc*truerank,1));
title('Ground Truth');
hold on;

I3 = zeros(M*numc,1);
% Select a random set of M entries of Y.
for it = 1:numc
    p = randperm(numr);
    I3((it-1)*M+1:it*M) = p(1:M);
end
J3 = reshape(repmat([1:numc],M,1),numc*M,1);

% Values of Y at the locations indexed by I and J.
S3 = sum(Omega(I3,:).*coeff(J3,:),2);


truerank = 5;
f = [0.1769    0.1992    0.4116    0.6776    0.9513];
amp = [ 0.3  0.8   0.5  1    0.6];

%f = sort(rand(1,truerank));
omega = diag(exp(j*2*pi*f));
Omega = ones(numr,truerank);
for i =2:numr
    Omega(i,:) = Omega(i-1,:)*omega;
end
Omega = Omega*diag(amp);

% coefficients
coeff = randn(numc,truerank);
    
    
figure(1);
t = ones(truerank,1)*[1+3*numc:4*numc];
scatter(reshape(t,numc*truerank,1),reshape(f'*ones(1,numc),numc*truerank,1),3,reshape(amp'*ones(1,numc),numc*truerank,1));
title('Ground Truth');
hold on;

I4 = zeros(M*numc,1);
% Select a random set of M entries of Y.
for it = 1:numc
    p = randperm(numr);
    I4((it-1)*M+1:it*M) = p(1:M);
end
J4 = reshape(repmat([1:numc],M,1),numc*M,1);

% Values of Y at the locations indexed by I and J.
S4 = sum(Omega(I4,:).*coeff(J4,:),2);

I = [I1; I2; I3; I4];
J = reshape(repmat([1:4*numc],M,1),4*numc*M,1);


S_noiseFree = [S1; S2; S3; S4];



noiseFac = .1e-2;

    % Add noise.
noise = noiseFac*randn(size(S_noiseFree));
S = S_noiseFree + noise;



maxCycles = 1;
step_size = 0.1;


lambda = 0.98;

numc = 4*numc;

[U,sub_err,err_reg,omega_est,amp_est] = subspace_modal_analysis(Omega,I,J,S,numr,numc,maxrank,maxCycles,lambda);

[U1,sub_err1,err_reg1,omega_est1,amp_est1] = grouse_model(Omega,I,J,S,numr,numc,maxrank,step_size,maxCycles);


% [Usg1, Vsg1, err_reg1, sub_err1] = grouse(Omega,I,J,S,numr,numc,maxrank,step_size,maxCycles);
   
% [Usg, err_reg, sub_err, trR] =
% rls_mc_parallel_mod(Omega,I,J,S,numr,numc,maxrank,maxCycles,lambda);

%% plot
%
t = ones(maxrank,1)*[1:numc];
figure(2);
scatter(reshape(t,numc*maxrank,1),reshape(omega_est,numc*maxrank,1),3,reshape(amp_est,numc*maxrank,1));
title('PETRELS');
xlabel('data stream index');
ylabel('mode locations');
figure(3);
scatter(reshape(t,numc*maxrank,1),reshape(omega_est1,numc*maxrank,1),3,reshape(amp_est1,numc*maxrank,1));
title('GROUSE');
xlabel('data stream index');
ylabel('mode locations');
%figure; % plot the ground truth
