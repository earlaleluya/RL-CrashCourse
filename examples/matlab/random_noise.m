function [r]=random_noise(low_limit, hi_limit)

r = (hi_limit-low_limit).*rand(1) + low_limit;