# requirement
import numpy as np
import matplotlib.pyplot as plt

clear all;
d = 200;
n = 180;
% we consider 5 groups where each group has 40 attributes
g = cell(5, 1);
for i = 1:length(g)
    g{i} = (i-1)*40+1:i*40;
end
x = randn(n, d);
noise = 0.5;
% we consider feature in group 1 and group 2 is activated.
w = [20 * randn(80, 1);
    zeros(120, 1);
    5 * rand];
x_tilde = [x, ones(n, 1)];
y = x_tilde * w + noise * randn(n, 1);
lambda = 1.0;
wridge = (x_tilde’*x_tilde + lambda * eye(d+1))\(x_tilde’ * y);
cvx_begin
variable west(d+1,1)
minimize( 0.5 / n * (x_tilde * west - y)’ * (x_tilde * west - y) + ...
    lambda * (norm(west(g{1}), 2.0) + ...
    norm(west(g{2}), 2.0) + ...
    norm(west(g{3}), 2.0) + ...
    norm(west(g{4}), 2.0) + ...
    norm(west(g{5}), 2.0) ))
cvx_end
x_test = randn(n, d);
x_test_tilde = [x_test, ones(n, 1)];
y_test = x_test_tilde * w + noise * randn(n, 1);
y_pred = x_test_tilde * west;
mean((y_pred - y_test) .ˆ2)
figure(1);
clf;
plot(west(1:d), ’r-o’)

hold on
plot(w, ’b-*’);
plot(wridge, ’g-+’);
legend(’group lasso’, ’ground truth’, ’ridge regression’)
figure(2);
clf;
plot(y_test, y_pred, ’bs’);
xlabel(’ground truth’)
ylabel(’prediction’)
fprintf(’carinality of w hat: %d\n’, length(find(abs(west) < 0.01)))
fprintf(’carinality of w ground truth: %d\n’, length(find(abs(w) < 0.01)))