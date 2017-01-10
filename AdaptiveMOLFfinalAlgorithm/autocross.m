function [R C] = autocross(x1,t,Nv) 
 
  %calculating R
  R = (x1' * x1) / Nv;
  %calculating C
  C = (x1' * t) / Nv;
  % Calculates output energies
  Et = sum(t .* t) / Nv;