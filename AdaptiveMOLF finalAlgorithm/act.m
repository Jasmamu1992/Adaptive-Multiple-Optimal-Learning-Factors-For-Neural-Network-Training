function [O]=act(net)
O=1./(1+exp(-net));