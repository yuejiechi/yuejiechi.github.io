
clear all
%rand('state',271) % random seed fixed so that the example is the one in the paper

% Problem data 

% cutoff frequency
Nc = 16;  
nspikes = 5;
k = 0:Nc-1; 
T = 50;
% nominal spacing
tnominal =  0:nspikes-1;
% spike locations
t1spikes =  rand(1,nspikes);

t2spikes =  rand(1,nspikes);
% amplitudes 
dynamic_range= 10;
x = exp(-1i*2*pi*rand(nspikes,1)) .* (1 + 10.^(rand(nspikes,1).*(dynamic_range/20))); % amplitudes can also be complex
%% data 

k1 = randi(Nc,1,T)-1;
k2 = randi(Nc,1,T)-1;

yfull = exp(1i*2*pi*(kron(k',ones(Nc,1))*t1spikes+kron(ones(Nc,1),k')*t2spikes))*x;


F = exp(1i*2*pi*(k1'*t1spikes+k2'*t2spikes)); % Fourier matrix 

y = F*x; n = Nc;

%% Solve SDP

cvx_solver sdpt3
cvx_begin sdp 
    variable u(2*n-1,2*n-1) 
    variable r(n^2,1) complex
    variable t
    variable X(n^2,n^2) symmetric
    minimize (0.5*trace(X)+0.5*t)
    for m1 = 1:n
        for m2 = m1:n
            X((m1-1)*n+1:m1*n,(m2-1)*n+1:m2*n) == toeplitz(u(n:end,m2-m1+1),u(n:-1:1,m2-m1+1));
        end
    end
    subject to
        [X, r; 
          r', t] >=0
        r(k1*n+k2+1) == y;
  %  end
    
cvx_end


res_err = norm(r-yfull)/norm(yfull);