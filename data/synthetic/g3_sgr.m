function [ g3, LRw1, LRw2, LRw3, LRsum, g3_tss ] = SGR_function( x, w1, w2, w3 )

w1 = w1( : );
w2 = w2( : );
w3 = w3( : );

data = LR_function( x, [ w1 w2 w3 ] );

% Careful about the "." it makes sure that the complex conjugate isn't
% taken.  This is critical.
LRw1 = ( x - 1 ) / x * data( :, 1 ).';
LRw2 = ( x - 1 ) / x * data( :, 2 ).';
LRw3 = ( x - 1 ) / x * data( :, 3 ).';

LRsum = ( x - 1 ) / x * LR_function( x, w1 + w2 + w3 ).';

LRw1w2 = ( x - 1 ) / x * LR_function( x, w1 + w2 ).';
LRw1w3 = ( x - 1 ) / x * LR_function( x, w1 + w3 ).';
LRw2w3 = ( x - 1 ) / x * LR_function( x, w2 + w3 ).';

g3_tss = LRsum - LRw1w2 - LRw1w3 - LRw2w3 + LRw1 + LRw2 + LRw3;

Tw1w2 = ( x - 1 ) / x * T_function( x, w1 + w2 );
Tw1w3 = ( x - 1 ) / x * T_function( x, w1 + w3 );
Tw2w3 = ( x - 1 ) / x * T_function( x, w2 + w3 );

Y_intw2w3 = ( x - 1 ) / x * Y_int_function( x, w2, w3 );
Y_intw1w3 = ( x - 1 ) / x * Y_int_function( x, w1, w3 );
Y_intw1w2 = ( x - 1 ) / x * Y_int_function( x, w1, w2 );

X_int = ( x - 1 ) / x * X_int_function( x, w1, w2, w3 );

g3 = ( ( LRw1 ./ Tw2w3 .* Y_intw2w3 + LRw2 ./ Tw1w3 .* Y_intw1w3 + LRw3 ./ Tw1w2 .* Y_intw1w2 ) / 3 - X_int ) / ( 2 * x );


function LR = LR_function( x, w )

LR = x / ( x - 1 ) * hypergeom( [ 1, x - 1 ], x, sqrt( -1 ) .* w .^ -1 );


function T = T_function( x, w )

ws = w;

ind = find( ws ~= 0 );

if ( ~isempty( ind ) )
       
    w = ws( ind );

    T( ind ) = -sqrt( -1 ) ./ w .* hypergeom( [ 1 x ], x + 1, sqrt( -1 ) .* w .^ -1 );

end;

ind = find( ws == 0 );

if ( ~isempty( ind ) )

    T( ind ) = gamma((-1)+x).*gamma(x).^(-2).*gamma(1+x);

end;



function Y_int = Y_int_function( x, w2, w3 );

w2s = w2;
w3s = w3;

ind = find( w2s ~= -w3s );

if ( ~isempty( ind ) )

    w2 = w2s( ind );
    w3 = w3s( ind );

    Y_int( ind ) = (sqrt(-1)*(-1)).*w3.^(-1).*hypergeom([1,x],1+x,sqrt(-1).*w2.^(-1)) ...
  +(sqrt(-1)*(-1)).*w2.^(-1).*hypergeom([1,x],1+x,sqrt(-1).*w3.^(-1) ...
  )+sqrt(-1).*w2.^(-1).*w3.^(-1).*(w2+w3).^(-1).*(w2.^2+w2.*w3+ ...
  w3.^2).*hypergeom([1,x],1+x,sqrt(-1).*(w2+w3).^(-1))+(w2+w3).^(-2) ...
  .*x.*((-1).*(w2+w3).*((sqrt(-1)*(-1))+w2+w3).^(-1)+x.*(1+x).^(-1) ...
  .*hypergeom([1,1+x],2+x,sqrt(-1).*(w2+w3).^(-1)));

end;

ind = find( w2s == -w3s );

if ( ~isempty( ind ) )
    
    w2 = w2s( ind );
    w3 = w3s( ind );
    
    Y_int( ind ) = 2.*((-1)+x).^(-1).*x+(sqrt(-1)*(-1)).*w3.^(-1).*hypergeom([1,x],1+ ...
  x,(sqrt(-1)*(-1)).*w3.^(-1))+sqrt(-1).*w3.^(-1).*hypergeom([1,x], ...
  1+x,sqrt(-1).*w3.^(-1));

end;



function X_int = X_int_function( x, w1, w2, w3 )

w1s = w1;
w2s = w2;
w3s = w3;

ind = find( ( w2s ~= -w3s ) & ( w1s ~= -w2s ) & ( w1s ~= -w3s ) );

if ( ~isempty( ind ) )

    w1 = w1s( ind );
    w2 = w2s( ind );
    w3 = w3s( ind );

    X_int( ind ) = (1/3).*((sqrt(-1)*(-1)).*w2.^(-1).*w3.^(-1).*(w2+w3).^(-1).*( ...
  w2.^2+3.*w2.*w3+w3.^2).*hypergeom([1,x],1+x,sqrt(-1).*w1.^(-1))+( ...
  sqrt(-1)*(-1)).*w1.^(-1).*w3.^(-1).*(w1+w3).^(-1).*(w1.^2+3.*w1.* ...
  w3+w3.^2).*hypergeom([1,x],1+x,sqrt(-1).*w2.^(-1))+sqrt(-1).*w1.^( ...
  -1).*w2.^(-1).*(w1+w2).^(-1).*w3.^(-1).*(w2.^2.*w3+w1.^2.*(2.*w2+ ...
  w3)+w1.*w2.*(2.*w2+w3)).*hypergeom([1,x],1+x,sqrt(-1).*(w1+w2).^( ...
  -1))+(sqrt(-1)*(-1)).*w1.^(-1).*w2.^(-1).*(w1+w2).^(-1).*(w1.^2+ ...
  3.*w1.*w2+w2.^2).*hypergeom([1,x],1+x,sqrt(-1).*w3.^(-1))+sqrt(-1) ...
  .*w1.^(-1).*w2.^(-1).*w3.^(-1).*(w1+w3).^(-1).*(w2.*w3.^2+w1.^2.*( ...
  w2+2.*w3)+w1.*w3.*(w2+2.*w3)).*hypergeom([1,x],1+x,sqrt(-1).*(w1+ ...
  w3).^(-1))+sqrt(-1).*w1.^(-1).*w2.^(-1).*w3.^(-1).*(w2+w3).^(-1).* ...
  (2.*w2.*w3.*(w2+w3)+w1.*(w2.^2+w2.*w3+w3.^2)).*hypergeom([1,x],1+ ...
  x,sqrt(-1).*(w2+w3).^(-1))+(sqrt(-1)*(-1)).*w1.^(-1).*w2.^(-1).*( ...
  w1+w2).^(-1).*w3.^(-1).*(w1+w3).^(-1).*(w2+w3).^(-1).*(2.*w2.^2.* ...
  w3.^2.*(w2+w3)+w1.^3.*(2.*w2.^2+3.*w2.*w3+2.*w3.^2)+w1.*w2.*w3.*( ...
  3.*w2.^2+5.*w2.*w3+3.*w3.^2)+w1.^2.*(2.*w2.^3+5.*w2.^2.*w3+5.*w2.* ...
  w3.^2+2.*w3.^3)).*hypergeom([1,x],1+x,sqrt(-1).*(w1+w2+w3).^(-1))+ ...
  (w1+w2).^(-2).*x.*((-1).*(w1+w2).*((sqrt(-1)*(-1))+w1+w2).^(-1)+ ...
  x.*(1+x).^(-1).*hypergeom([1,1+x],2+x,sqrt(-1).*(w1+w2).^(-1)))+( ...
  w1+w3).^(-2).*x.*((-1).*(w1+w3).*((sqrt(-1)*(-1))+w1+w3).^(-1)+x.* ...
  (1+x).^(-1).*hypergeom([1,1+x],2+x,sqrt(-1).*(w1+w3).^(-1)))+(w2+ ...
  w3).^(-2).*x.*((-1).*(w2+w3).*((sqrt(-1)*(-1))+w2+w3).^(-1)+x.*(1+ ...
  x).^(-1).*hypergeom([1,1+x],2+x,sqrt(-1).*(w2+w3).^(-1)))+3.*(w1+ ...
  w2+w3).^(-2).*x.*((w1+w2+w3).*((sqrt(-1)*(-1))+w1+w2+w3).^(-1)+( ...
  -1).*x.*(1+x).^(-1).*hypergeom([1,1+x],2+x,sqrt(-1).*(w1+w2+w3).^( ...
  -1))));

end;

ind = find( ( w2s == -w3s ) & ( w1s ~= -w2s ) & ( w1s ~= -w3s ) );

if ( ~isempty( ind ) )

    w1 = w1s( ind );
    w2 = w2s( ind );
    w3 = w3s( ind );

    X_int( ind ) = (1/3).*(2.*((-1)+x).^(-1).*x+(sqrt(-1)*2).*w3.^2.*(w1.^3+(-1).* ...
  w1.*w3.^2).^(-1).*hypergeom([1,x],1+x,sqrt(-1).*w1.^(-1))+sqrt(-1) ...
  .*w1.^(-1).*(w1+(-1).*w3).^(-1).*w3.^(-1).*(w1.^2+(-1).*w1.*w3+( ...
  -1).*w3.^2).*hypergeom([1,x],1+x,sqrt(-1).*(w1+(-1).*w3).^(-1))+( ...
  sqrt(-1)*(-1)).*w1.^(-1).*w3.^(-1).*(w1+w3).^(-1).*(w1.^2+3.*w1.* ...
  w3+w3.^2).*hypergeom([1,x],1+x,(sqrt(-1)*(-1)).*w3.^(-1))+sqrt(-1) ...
  .*w1.^(-1).*(w1+(-1).*w3).^(-1).*w3.^(-1).*(w1.^2+(-3).*w1.*w3+ ...
  w3.^2).*hypergeom([1,x],1+x,sqrt(-1).*w3.^(-1))+(sqrt(-1)*(-1)).* ...
  w1.^(-1).*w3.^(-1).*(w1+w3).^(-1).*(w1.^2+w1.*w3+(-1).*w3.^2).* ...
  hypergeom([1,x],1+x,sqrt(-1).*(w1+w3).^(-1))+4.*w1.^(-2).*x.*(w1.* ...
  ((sqrt(-1)*(-1))+w1).^(-1)+(-1).*x.*(1+x).^(-1).*hypergeom([1,1+ ...
  x],2+x,sqrt(-1).*w1.^(-1)))+(w1+(-1).*w3).^(-2).*x.*((w1+(-1).*w3) ...
  .*(sqrt(-1)+(-1).*w1+w3).^(-1)+x.*(1+x).^(-1).*hypergeom([1,1+x], ...
  2+x,sqrt(-1).*(w1+(-1).*w3).^(-1)))+(w1+w3).^(-2).*x.*((-1).*(w1+ ...
  w3).*((sqrt(-1)*(-1))+w1+w3).^(-1)+x.*(1+x).^(-1).*hypergeom([1,1+ ...
  x],2+x,sqrt(-1).*(w1+w3).^(-1))));
    
end;

ind = find( ( w2s ~= -w3s ) & ( w1s == -w2s ) & ( w1s ~= -w3s ) );

if ( ~isempty( ind ) )

    w1 = w1s( ind );
    w2 = w2s( ind );
    w3 = w3s( ind );

    X_int( ind ) = (1/3).*((-1).*(w2+(-1).*w3).^(-1).*(sqrt(-1)+w2+(-1).*w3).^(-1).* ...
  x+4.*w3.^(-1).*((sqrt(-1)*(-1))+w3).^(-1).*x+(-1).*(w2+w3).^(-1).* ...
  ((sqrt(-1)*(-1))+w2+w3).^(-1).*x+2.*((-1)+x).^(-1).*x+(sqrt(-1)*( ...
  -1)).*w2.^(-1).*w3.^(-1).*(w2+w3).^(-1).*(w2.^2+3.*w2.*w3+w3.^2).* ...
  hypergeom([1,x],1+x,(sqrt(-1)*(-1)).*w2.^(-1))+(sqrt(-1)*(-1)).* ...
  w2.^(-1).*(w2+(-1).*w3).^(-1).*w3.^(-1).*(w2.^2+(-3).*w2.*w3+ ...
  w3.^2).*hypergeom([1,x],1+x,sqrt(-1).*w2.^(-1))+(sqrt(-1)*2).* ...
  w2.^2.*((-1).*w2.^2.*w3+w3.^3).^(-1).*hypergeom([1,x],1+x,sqrt(-1) ...
  .*w3.^(-1))+sqrt(-1).*w2.^(-1).*(w2+(-1).*w3).^(-1).*w3.^(-1).*( ...
  w2.^2+w2.*w3+(-1).*w3.^2).*hypergeom([1,x],1+x,sqrt(-1).*((-1).* ...
  w2+w3).^(-1))+sqrt(-1).*w2.^(-1).*w3.^(-1).*(w2+w3).^(-1).*(w2.^2+ ...
  (-1).*w2.*w3+(-1).*w3.^2).*hypergeom([1,x],1+x,sqrt(-1).*(w2+w3) ...
  .^(-1))+(-4).*w3.^(-2).*x.^2.*(1+x).^(-1).*hypergeom([1,1+x],2+x, ...
  sqrt(-1).*w3.^(-1))+(w2+(-1).*w3).^(-2).*x.^2.*(1+x).^(-1).* ...
  hypergeom([1,1+x],2+x,sqrt(-1).*((-1).*w2+w3).^(-1))+(w2+w3).^(-2) ...
  .*x.^2.*(1+x).^(-1).*hypergeom([1,1+x],2+x,sqrt(-1).*(w2+w3).^(-1) ...
  ));

end;

ind = find( ( w2s ~= -w3s ) & ( w1s ~= -w2s ) & ( w1s == -w3s ) );

if ( ~isempty( ind ) )

    w1 = w1s( ind );
    w2 = w2s( ind );
    w3 = w3s( ind );

    X_int( ind ) = (1/3).*(2.*((-1)+x).^(-1).*x+(sqrt(-1)*2).*w3.^2.*(w2.^3+(-1).* ...
  w2.*w3.^2).^(-1).*hypergeom([1,x],1+x,sqrt(-1).*w2.^(-1))+sqrt(-1) ...
  .*w2.^(-1).*(w2+(-1).*w3).^(-1).*w3.^(-1).*(w2.^2+(-1).*w2.*w3+( ...
  -1).*w3.^2).*hypergeom([1,x],1+x,sqrt(-1).*(w2+(-1).*w3).^(-1))+( ...
  sqrt(-1)*(-1)).*w2.^(-1).*w3.^(-1).*(w2+w3).^(-1).*(w2.^2+3.*w2.* ...
  w3+w3.^2).*hypergeom([1,x],1+x,(sqrt(-1)*(-1)).*w3.^(-1))+sqrt(-1) ...
  .*w2.^(-1).*(w2+(-1).*w3).^(-1).*w3.^(-1).*(w2.^2+(-3).*w2.*w3+ ...
  w3.^2).*hypergeom([1,x],1+x,sqrt(-1).*w3.^(-1))+(sqrt(-1)*(-1)).* ...
  w2.^(-1).*w3.^(-1).*(w2+w3).^(-1).*(w2.^2+w2.*w3+(-1).*w3.^2).* ...
  hypergeom([1,x],1+x,sqrt(-1).*(w2+w3).^(-1))+4.*w2.^(-2).*x.*(w2.* ...
  ((sqrt(-1)*(-1))+w2).^(-1)+(-1).*x.*(1+x).^(-1).*hypergeom([1,1+ ...
  x],2+x,sqrt(-1).*w2.^(-1)))+(w2+(-1).*w3).^(-2).*x.*((w2+(-1).*w3) ...
  .*(sqrt(-1)+(-1).*w2+w3).^(-1)+x.*(1+x).^(-1).*hypergeom([1,1+x], ...
  2+x,sqrt(-1).*(w2+(-1).*w3).^(-1)))+(w2+w3).^(-2).*x.*((-1).*(w2+ ...
  w3).*((sqrt(-1)*(-1))+w2+w3).^(-1)+x.*(1+x).^(-1).*hypergeom([1,1+ ...
  x],2+x,sqrt(-1).*(w2+w3).^(-1))));

end;

ind = find( ( ( w3s == -w1s ) & ( w3s == -w2s ) ) | ( ( w2s == -w1s ) & ( w2s == -w3s ) ) | ( ( w1s == -w2s ) & ( w1s == -w3s ) ) );

if ( ~isempty( ind ) )

    w1 = w1s( ind );
    w2 = w2s( ind );
    w3 = w3s( ind );

    X_int( ind ) = (1/12).*(16.*((-1)+x).^(-1).*x+(sqrt(-1)*2).*w2.^(-1).*hypergeom( ...
  [1,x],1+x,(sqrt(-1)*(-1/2)).*w2.^(-1))+(sqrt(-1)*6).*w2.^(-1).* ...
  hypergeom([1,x],1+x,(sqrt(-1)*(-1)).*w2.^(-1))+(sqrt(-1)*10).* ...
  w2.^(-1).*hypergeom([1,x],1+x,sqrt(-1).*w2.^(-1))+(-1).*w2.^(-2).* ...
  x.*(2.*w2.*(sqrt(-1)+2.*w2).^(-1)+(-1).*x.*(1+x).^(-1).*hypergeom( ...
  [1,1+x],2+x,(sqrt(-1)*(-1/2)).*w2.^(-1)))+20.*w2.^(-2).*x.*(w2.*( ...
  sqrt(-1)+w2).^(-1)+(-1).*x.*(1+x).^(-1).*hypergeom([1,1+x],2+x,( ...
  sqrt(-1)*(-1)).*w2.^(-1))));

end;


