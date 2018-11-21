
% Include the relevant directories
%addpath('matlab/detector')
%addpath('matlab/util')


% Load audio, ...
%[y, fs] = audioread('path/to/file.wav');
% ... or, synthesize signal.
fs = 32000;
t = 0:1/fs:(1.0 - 1/fs);
signal = [zeros(1, 2000), 1.0 * chirp(t(2001:end-2000), 1000, 0.8, 11000, 'quadratic'), zeros(1, 2000)] + ...
    [zeros(1, 13000), 0.3 * chirp(t(13001:end-10000), 19000, 0.65, 3000, 'quadratic'), zeros(1, 10000)] + ...
    [zeros(1, 12000), 0.5 * sin(2 * pi * 9500 * t(12001:end-14000)), zeros(1, 14000)];
noise = (0.2 + 0.2) .* rand(size(signal)) - 0.2;    % noise amplitude in the range [-0.2, 0.2]
y = signal + noise;
y = y ./ max(abs(y));  % normalize


% Compute spectrogram
win_len = 256;
win_ovrlp = 128;
nfft = 256;
[S, F, T, P] = spectrogram(y, hanning(win_len), win_ovrlp, nfft, fs);

% Note: If your audio is too long, you could iteratively load it in
% successive chunks, compute spectrograms of each chunk and have it
% processed by the tracker before loading the next chunk. This example,
% demostrates only the last part (of iterative processing) while the entire
% spectrogram is computed in one go above.


figure;
ax(1) = subplot(2, 1, 1);
imagesc(T, F, 10 * log10(P)); shading flat;
set(gca, 'ydir', 'normal');
ax(2) = subplot(2, 1, 2);
linkaxes(ax);


dT = T(2) - T(1);   % time resolution
dF = F(2) - F(1);   % frequency resolution

% STEP 1: Create a tracker object -
h_tracker = dt_SpectrogramRidgeTracker(dT, dF, F);

% STEP 2: Set the tracker's parameters -
h_tracker.Set_threshold_value(4);   % Set detection threshold.
                                    % An integer, in the range [1-8]

h_tracker.Set_min_intensity(10^(-80/10));  % Min spectral intensity.
                                    % To consider everything, pass '-Inf'.
                                    % Otherwise, pass in a value in linear
                                    % scale. For eg, to ignore points below
                                    % -80 dB, pass in '10 ^ (-80 / 10)'.

h_tracker.SetMinContourLength(15);  % Only TF contours spanning these many
                                    % frames or more will be reported. This
                                    % could be set as a function of time;
                                    % eg: specify 0.3 s as floor(0.3 / dT)

h_tracker.SetMaxContourInactivity(10);
                                    % Number of frames in a gap between two
                                    % temporally close TF contours for them
                                    % to be joined. Can also be specified
                                    % as a function of time as above.

h_tracker.SetTrackingStartTime(0);  % This is useful if you start the
                                    % "streaming mode" operation (see
                                    % below) mid-stream. By setting this
                                    % 'starting offset' time appropriately,
                                    % you can ensure that the reported TF
                                    % tracks will have valid time values.

% Step 3: How do you want to deal with the reported TF contours?
%         Setting a callback function here which plots the reported tracks.
% You may define and use your own callback functions. Eg: your function
% could to write the reported tracks to a file/database, etc.
h_tracker.SetCallback(@PlotTracks, ax(2));

% STEP 4: Process the spectrogram frames. Either pass in all of the frames
% at once, or process them as chunks in "streaming mode".
block_size = 50;    % deal with chunks of 50 frames at a time
block_start = 1;
while block_start+block_size-1 <= size(P, 2)
    h_tracker.ProcessFrames(P(:, block_start:(block_start+block_size-1)));
    
    block_start = block_start + block_size;
end
h_tracker.ProcessFrames(P(:, block_start:end));     % Last (partial) block
%h_tracker.ProcessFrames(P);    % Pass all at once, instead of streaming
                                % in chunks (Comment out the above lines)

% STEP 5: Flush internal buffers.
h_tracker.Flush();

% STEP 6: Delete the tracker handle, we're done with it.
delete(h_tracker);


function PlotTracks(ax, tracks)
    % Plot reported 'tracks' on the given axis.
    % 'tracks' variable is a cell array, and each cell is an Nx3 matrix
    % in which rows 1, 2 & 3 contain a TF contour's time, frequency and
    % spectral intensity (in dB) values, respectively.
    % 'ax' is the second parameter that was passed to SetCallback() above.
    
    % Functionality to roll color selection
    persistent tid;
    if isempty(tid)
        tid = 0;
    end
    line_colors = hsv(8);
    
    num_tracks = length(tracks);

    hold(ax, 'on');
    for track_idx = 1:num_tracks
        plot(ax, tracks{track_idx}(1, :), tracks{track_idx}(2, :), '.-', ...
            'color', line_colors(mod(tid, 8)+1, :));
        tid = tid + 1;  % Update next color idx
    end
    hold(ax, 'off');
    drawnow;
    
    if num_tracks > 0
        fprintf('Added %i tracks to axis\n', num_tracks);
    end
end

