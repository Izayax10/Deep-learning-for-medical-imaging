function [Xrand,Xkacz,Xsymk,x] = gen()
%DEMO_ART Demonstrates the use of, and results from, the ART methods
%
% This script illustrates the use of the ART methods Kaczmarz, symmetric
% Kaczmarz, and randomized Kaczmarz.
%
% The script creates a parallel-beam CT test problem and solves it with the
% ART methods.  The exact image and the results from the methods are shown.
%
% See also: demo_cart, demo_constraints, demo_custom_all, demo_matrixfree,
% demo_relaxpar, demo_sirt, demo_stoprules, demo_training.

% Code written by: Per Christian Hansen, Jakob Sauer Jorgensen, and 
% Maria Saxild-Hansen, 2010-2017.

% This file is part of the AIR Tools II package and is distributed under
% the 3-Clause BSD License. A separate license file should be provided as
% part of the package. 
% 
% Copyright 2017 Per Christian Hansen, Technical University of Denmark and
% Jakob Sauer Jorgensen, University of Manchester.


fprintf(1,'Starting demo_art:\n\n');

% Set the parameters for the test problem.
N = 50;           % The image is N-times-N..
theta = 0:2:178;  % No. of used angles.
p = 75;           % No. of parallel rays.
k = 10;           % Number of iterations.

fprintf(1,'Creating a parallel-beam tomography test problem\n');
fprintf(1,'with N = %2.0f, theta = %1.0f:%1.0f:%3.0f, and p = %2.0f.',...
    [N,theta(1),theta(2)-theta(1),theta(end),p]);

% Create the test problem.
[A,b,x] = paralleltomo(N,theta,p);
x = reshape(x,N,N);

Xkacz = kaczmarz(A,b,k);
Xkacz = reshape(Xkacz,N,N);

Xsymk = symkaczmarz(A,b,k);
Xsymk = reshape(Xsymk,N,N);

Xrand = randkaczmarz(A,b,k);
Xrand = reshape(Xrand,N,N);

end

