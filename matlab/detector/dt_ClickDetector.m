function [dets, time_taken] = dt_ClickDetector(data, Fs, threshold, fwhm_EC_s, debug_plot)
% function [dets, time_taken] = ...
%               dt_ClickDetector(data, Fs, threshold, fwhm_EC_s, debug_plot)
%
% Implementation of the echolocation-click detection algorithm described in
%   Madhusudhana, S., Gavrilov, A., Erbe, C. "Automatic detection of
%   echolocation clicks based on a Gabor model of their waveform."a J Acoust
%   Soc Am, 137 (6), p. 3077-3086, 2015. DOI: doi.org/10.1121/1.4921609
%
% Inputs:
%   data        Single channel time-series data as an array.
%   Fs          Sampling rate (scalar) of the 'data'.
%   threshold   Detector threshold, a scalar, typically within the range
%               [0.4, 0.95].
%   fwhm_EC_s   Full width at half-maximum of target echolocation click, in
%               seconds [default is 0.4 ms * sqrt(2)]. See section III.B in
%               article for details.
%   debug_plot  Setting this to true shows the results on a plot [default
%               is false]. Note: Avoid setting this to true for large data.
%
% Outputs:
%   dets        [N x 2] array where each row is a (start, end) pair of a
%               detected click's extents in sample indices.
%   time_taken  Algorithm's running time in seconds.
%
% Author: Shyam Madhusudhana
%         research@shyamblast.com
%


if nargin < 5
    debug_plot = false;

    if nargin < 4
        fwhm_EC_s = 0.0004 * sqrt(2);
    end
end

% Set this to a fixed value if absolutely necessary
min_click_duration = fwhm_EC_s / 2; % 30 / 1000000;  % 30 us

sigma_TK = fwhm_EC_s / (4 * sqrt(log(2)));  % Eq (11)
N = uint64(ceil(5 * sigma_TK * Fs));
min_click_samples = uint64(ceil(min_click_duration * Fs));

% Buid the smoothing/averaging filters. Eqs (12) and (13)
MAF1 = (1 / (Fs * sigma_TK * sqrt(2 * pi))) * ...
    exp(-(((1/Fs) * (-double(N):double(N))).^2) / (2 * sigma_TK^2));
MAF2 = repmat(mean(MAF1), 1, 2*N + 1);

FDR_threshold = threshold * ...
    (1 - (MAF2(N+1) / MAF1(N+1)));  % This second part is the FDR peak

data = data(:); % Make sure input is in column format

dets = zeros(0, 2); % Initialise result to "no detections"

my_timer = tic;
% Algorithm's processing start
% -------------------------------------------------------------------------

% Compute TKE
tke = (data .^ 2) - [0; (data(1:end-2) .* data(3:end)); 0];

% Apply filters. Keep only the valid points
smooth1 = conv(tke, MAF1, 'valid');
smooth2 = conv(tke, MAF2, 'valid');

% As per Eq (10), FDR is defined as :
%   FDR = (smooth1 - smooth2) ./ smooth1
%
% The conditional
%   click_pts = (FDR >= FDR_threshold)
% can be expressed equivalently as :
%             = ((1 - (smooth2 ./ smooth1)) >= FDR_threshold)
%             = ((1 - FDR_threshold) >= (smooth2 ./ smooth1))
%             = (((1 - FDR_threshold) * smooth1) >= smooth2)
% in order to speed up processing. Also, avoids divide-by-zero hassles.

click_pts = uint64(find((smooth2 > 0) & ...
    (((1 - FDR_threshold) * smooth1) >= smooth2)));
% With the way the above condition is expressed, we wouldn't need to
% explicitly also check for (smooth1 > 0) and (smooth1 > smooth2).

if ~isempty(click_pts)
    % Group together consecutive points beyond threshold and determine the extents
    temp = uint64(find(diff(click_pts) > 1));
    
    % The "+N" in the below statements is to offset indices owing to convolution artefacts
    if isempty(temp)
        dets = [click_pts(1) click_pts(end)] + N;
    else
        dets = [[click_pts(1); click_pts(temp + 1)], click_pts([temp; end])] + N;
    end

    % Remove short dets as they will most likely be dubious
    dets(dets(:, 2) - dets(:, 1) + 1 < min_click_samples, :) = [];
end

% =========================================================================
% End of algorithm
time_taken = toc(my_timer);


if debug_plot
    figure;
    
    ax = [0 0 0];
    ax(1) = subplot(3, 1, 1);
    plot((1:length(data)) ./ Fs, data, 'b');
    hold(ax(1), 'on');
    for temp = 1:size(dets, 1)
        plot(double(dets(temp, 1):dets(temp, 2)) ./ Fs, data(dets(temp, 1):dets(temp, 2)), 'r');
    end
    hold(ax(1), 'off');
    ylabel('Waveform', 'fontsize', 8);
    
    ax(2) = subplot(3, 1, 2);
    plot((1:length(data)) ./ Fs, tke, 'k', ...
        double((N+1):(length(data)-N)) ./ Fs, smooth2, 'g', ...
        double((N+1):(length(data)-N)) ./ Fs, smooth1, 'm');
    set(ax(2), 'ylim', 1.1 * [min(tke) max(tke)]);
    ylabel('Energy', 'fontsize', 8);
    legend({'TKE', 'h_{MAF2}', 'h_{MAF1}'}, 'fontsize', 7);
    
    ax(3) = subplot(3, 1, 3);
    FDR = zeros(length(data), 1);
    valid_idxs = uint64(find((smooth1 > 0) & (smooth2 > 0) & (smooth1 > smooth2)));
    FDR(valid_idxs + N) = 1 - (smooth2(valid_idxs) ./ smooth1(valid_idxs));
    plot((1:length(data)) ./ Fs, FDR, 'b', ...
        [1 length(data)] ./ Fs, repmat(FDR_threshold, 1, 2), 'r');
    set(ax(3), 'ylim', [0 1]);
    xlabel('Time (s)', 'fontsize', 8);
    ylabel('FDR', 'fontsize', 8);
    
    linkaxes(ax, 'x');
    set(ax(1:2), 'xticklabel', []);
    set(ax, 'xlim', [1 length(data)] ./ Fs, ...
        'tickdir', 'out', 'ticklen', [0.003 0.003], ...
        'fontsize', 8);
end

