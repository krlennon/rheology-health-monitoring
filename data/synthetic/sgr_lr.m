clear;

format long e;

% Define the parameter space over which to do the calculation
%k = logspace( -1, 5, 100 ); % Pa 
%f = logspace( -4, 1, 10 ); % Hz
%x = linspace( 1.01, 8, 100 ); % dimensionless
%omega_0 = logspace( -5, -1, 10 ); % Hz

freqs = [0.5, 0.631, 1, 1.585, 2.511, 3.981, 6.31, 10];


% Make a vector out of all parameter combinations
%[ km, fm, xm, omega_0m ] = ndgrid( k, f, x, omega_0 );

%km = km( : );
%fm = fm( : );
%xm = xm( : );
%omega_0m = omega_0m( : );

%length(omega_0m)

% For random generation of n data points
n_train = 9000;
n_test = 1000;

% Loop over all the parameter sets to generate the data
data_strain = zeros(n_test+n_train,11);
data_stress = zeros(n_test+n_train,11);

for i = 1:(n_test+n_train)%1:length( omega_0m )

    % For random generation
    km = 10^(6*rand - 1);
    fm = 10^(4*rand + 1);
    xm = 4.99*rand + 1.01;
    omega_0m = 1;

    disp( i );
    
    LR = hypergeom( [ 1, xm - 1 ], xm, sqrt( -1 ) .* (freqs ./ fm) .^ -1 );
    
    data_strain(i,:) = [ km, fm, xm, km .* LR ];
    
    %j3 = -g3 ./ ( LRc1 .* LRc2 .* LRc3 .* LRsum );
    %j3_tss = -g3_tss ./ ( LRc1 .* LRc2 .* LRc3 .* LRsum );
    %data_stress(i,:) = [ km, fm, xm, omega_0m, 1 / (km * LRc1(1)), 1 / (km * LRc2(1)), 1 / (km * LRc3(1)), j3 / ( km ^ 3 ), j3_tss / ( km ^ 3 ) ];

end

save('SGR_LR.mat','data_strain')

