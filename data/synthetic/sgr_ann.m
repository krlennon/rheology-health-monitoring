clear;

format long e;

% Define the parameter space over which to do the calculation
%k = logspace( -1, 5, 100 ); % Pa 
%f = logspace( -4, 1, 10 ); % Hz
%x = linspace( 1.01, 8, 100 ); % dimensionless
%omega_0 = logspace( -5, -1, 10 ); % Hz

% Define the MAPS experiment in terms of three integers
n1 = 5;
n2 = 6;
n3 = 9;

channels = [n1, n2, n3; n1, n2, -n3; n1, -n2, n3; n1, -n2, -n3; n1, n1, ... 
   n2; n1, n1, -n2; n1, n1, n3; n1, n1, -n3; n2, n2, n1; n2, ...
    n2, -n1; n2, n2, n3; n2, n2, -n3; n3, n3, n1; n3, ...
   n3, -n1; n3, n3, n2; n3, n3, -n2; n1, n1, n1; n2, n2, ...
   n2; n3, n3, n3];


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
data_strain = zeros(n_test+n_train,45);
data_stress = zeros(n_test+n_train,45);

parfor i = 1:(n_test+n_train)%1:length( omega_0m )

    % For random generation
    km = 10^(6*rand - 1);
    fm = 10^(4*rand + 1);
    xm = 4.99*rand + 1.01;
    omega_0m = 1;

    %disp( i );
    
    [ g3, LRc1, LRc2, LRc3, LRsum, g3_tss ] = g3_sgr( xm, channels( :, 1 ) * omega_0m / fm, channels( :, 2 ) * omega_0m / fm, channels( :, 3 ) * omega_0m / fm );
    
    data_strain(i,:) = [ km, fm, xm, omega_0m, km * LRc1( 1 ), km * LRc2( 1 ), km * LRc3( 1 ), km * g3, km * g3_tss ];
    
    j3 = -g3 ./ ( LRc1 .* LRc2 .* LRc3 .* LRsum );
    j3_tss = -g3_tss ./ ( LRc1 .* LRc2 .* LRc3 .* LRsum );
    data_stress(i,:) = [ km, fm, xm, omega_0m, 1 / (km * LRc1(1)), 1 / (km * LRc2(1)), 1 / (km * LRc3(1)), j3 / ( km ^ 3 ), j3_tss / ( km ^ 3 ) ];

end

save('SGR_data_tensorial_569.mat','data_strain','data_stress')

