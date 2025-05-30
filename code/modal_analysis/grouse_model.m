function [U,err_reg,sub_err,omega_est,amp_est] = grouse_model(YL,I,J,S,numr,numc,maxrank,step_size,maxCycles,Uinit)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  GROUSE (Grassman Rank-One Update Subspace Estimation) matrix completion code 
%  by Ben Recht and Laura Balzano, February 2010.
%
%  Given a sampling of entries of a matrix X, try to construct matrices U
%  and R such that U is unitary and UR' approximates X.  This code 
%  implements a stochastic gradient descent on the set of subspaces.
%
%  Inputs:
%       (I,J,S) index the known entries across the entire data set X. So we
%       know that for all k, the true value of X(I(k),J(k)) = S(k)
%
%       numr = number of rows
%       numc = number of columns
%           NOTE: you should make sure that numr<numc.  Otherwise, use the
%           transpose of X
%       
%       max_rank = your guess for the rank
%
%       step_size = the constant for stochastic gradient descent step size
%
%       maxCycles = number of passes over the data
%
%       Uinit = an initial guess for the column space U (optional)
%
%   Outputs:
%       U and R such that UR' approximates X.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Matlab specific data pre-processing
%

% Form some sparse matrices for easier matlab indexing
values = sparse(I,J,S,numr,numc);
Indicator = sparse(I,J,1,numr,numc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%Main Algorithm
%

if (nargin<10)
    % initialize U to a random r-dimensional subspace 
    U = orth(randn(numr,maxrank)); 
else
    U = Uinit;
end

err_reg = zeros(maxCycles*numc,1);

sub_err = zeros(maxCycles*numc,1);
omega_est = zeros(maxrank,maxCycles*numc);
amp_est = zeros(maxrank,maxCycles*numc);

for outiter = 1:maxCycles,
    
    fprintf('Pass %d...\n',outiter);
    
    % create a random ordering of the columns for the current pass over the
    % data.
    col_order = randperm(numc);
    
for k=1:numc,
     % k,
    % Pull out the relevant indices and revealed entries for this column
    idx = find(Indicator(:,col_order(k)));
    v_Omega = values(idx,col_order(k));
    U_Omega = U(idx,:);    

    
    % Predict the best approximation of v_Omega by u_Omega.  
    % That is, find weights to minimize ||U_Omega*weights-v_Omega||^2
    
    weights = pinv(U_Omega)*v_Omega;
    norm_weights = norm(weights);
    
    % Compute the residual not predicted by the current estmate of U.

    residual = v_Omega - U_Omega*weights;       
    norm_residual = norm(residual);
    
    % This step-size rule is given by combining Edelman's geodesic
    % projection algorithm with a diminishing step-size rule from SGD.  A
    % different step size rule could suffice here...        
    
    sG = norm_residual*norm_weights;
    err_reg((outiter-1)*numc + k) = norm_residual/norm(v_Omega);
    t = step_size*sG/( (outiter-1)*numc + k );
   
    % Take the gradient step.    
   % if t<pi/2, % drop big steps        
        alpha = (cos(t)-1)/norm_weights^2;
        beta = sin(t)/sG;

        step = U*(alpha*weights);
        step(idx) = step(idx) + beta*residual;

        U = U + step*weights';
    %end 
    
  
    
    sub_err((outiter-1)*numc + k) = norm((eye(numr)-U*U')*YL,'fro')/norm(YL,'fro');
    
   
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

