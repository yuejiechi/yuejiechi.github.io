function [U,sub_err,err_reg,omega_est,amp_est] = subspace_modal_analysis(YL,I,J,S,numr,numc,maxrank,maxCycles,lambda,Uinit)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Form some sparse matrices for easier matlab indexing
values = sparse(I,J,S,numr,numc);
Indicator = sparse(I,J,1,numr,numc);
%Main Algorithm


if (nargin<10)
    % initialize U to a random r-dimensional subspace 
    U = orth(randn(numr,maxrank)); 
else
    U = Uinit;
end


err_reg = zeros(maxCycles*numc,1);
sub_err = zeros(maxCycles*numc,1);
omega_est = zeros(maxrank,maxCycles*numc);
       
% initialize the covariance matrix and forgetting parameter
    


R1 = repmat(1*eye(maxrank),1,numr);
R2 = zeros(maxrank,numr);

for outiter = 1:maxCycles,
    
    outiter,
    fprintf('Pass %d...\n',outiter);
    
    % create a random ordering of the columns for the current pass over the
    % data.
    % col_order = randperm(numc);
    col_order = 1:1:numc;
    % initialize the covariance matrix and forgetting parameter
 
        
for k=1:numc,

    % Pull out the relevant indices and revealed entries for this column
    idx = find(Indicator(:,col_order(k)));
    idxc = find(~Indicator(:,col_order(k)));

    v_Omega = values(idx,col_order(k));
    U_Omega = U(idx,:);    
    
    % Predict the best approximation of v_Omega by u_Omega.  
    % That is, find weights to minimize ||U_Omega*weights-v_Omega||^2
   
    weights = pinv(U_Omega'*U_Omega)*U_Omega'*v_Omega; %U_Omega'*v_Omega; %
    norm_weights = norm(weights);
    
    % Compute the residual not predicted by the current estmate of U.

    residual = v_Omega - U_Omega*weights;       
    norm_residual = norm(residual);
    
    err_reg((outiter-1)*numc + k) = norm_residual/norm(v_Omega);
    
    % This step update Rinv matrix with forgetting parameter lambda
    % for each observed row in U
    
    R1 = lambda*R1;
     
    R2 = lambda*R2;

    
    for i=1:length(idx)
        
        T1 = R1(:,(idx(i)-1)*maxrank+1:idx(i)*maxrank);
        
        T2 = R2(:,idx(i));
        
        T1 = T1+weights*weights';
        
        T2 = T2+weights*v_Omega(i);
        % update U_omega
    
        U(idx(i),:) = U_Omega(i,:) + residual(i)*weights'*pinv(T1);

        
        R1(:,(idx(i)-1)*maxrank+1:idx(i)*maxrank) = T1;
        R2(:,idx(i)) = T2; %
        
    end
    
    
    [Uq,Us,Ur] = svd(U,0); 
     

    sub_err((outiter-1)*numc + k) = norm((eye(numr)-Uq*Uq')*YL,'fro')/norm(YL,'fro'); %subspace(U, YL);
    
    % ESPRIT
    T = pinv(U(1:end-1,:))*U(2:end,:);
    [eigvec, eigval] = eig(T);
    f_est = angle(diag(eigval))/(2*pi);
    for i=1:maxrank
        if f_est(i)<0 
            f_est(i) = f_est(i)+1;
        end
    end
    
    omega_est(:,(outiter-1)*numc + k) = f_est;
    amp_est(:,(outiter-1)*numc + k) = abs(diag(eigval));
     
end

end


