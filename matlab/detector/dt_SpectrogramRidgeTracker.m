classdef dt_SpectrogramRidgeTracker < handle
    % dt_SpectrogramRidgeTracker - Class to encompass functionality for a
    % streaming interface ridge tracker in spectrograms. Internally uses
    % an instance of dt_SpectrogramRidgeDetector class for the detection of
    % ridge points. Ridge points across successive frames are associated
    % together with Kalman tracking.
    % See  http://greg.czerniak.info/guides/kalman1/  for a quick
    % explanation on the Kalman filter model parameters and sub-processes.
    % The terminology used here reflects those used on the above webpage.


% -------------------------------------------------------------------------
% The Kalman Filter is modeled as follows:
%    X(n)   =     A     *    X(n-1)   +     B     *           U
% 
% | F(n)  |   | 1 0 |   | F(n-1)  |   | 1 0 |   | tan(Beta(n-1))*dT |
% | P(n)  | = | 0 1 | * | P(n-1)  | + | 0 0 | * |         0         |
% 
% where, F(n) = the frequency (freq) of the point on the track at frame n
%        P(n) = the power of the point on the track at frame n & freq F(n)
% Note: Ridge strength measure of point at frame n & freq F(n) was also
% tried, but did not yield noticeable recognition performance improvement.
%
% For working with angles (Beta), the axes must be normalised. This is
% because, the angles obtained from dt_SpectrogramRidgeDetector are for
% unit axes values, not for scaled frequency and time axis values. We
% can normalise the quantity [F2 - F1] by dividing it by dF (spectrogram's
% frequency resolution) and similarly, a change in time can be normalised
% by dividing it by the spectrogram's time resolution. Effectively,
%                    [F2-F1] / dF
%       tan(Beta) = --------------
%                    [T2-T1] / dT
% For all computations done here, the quantity [T2 - T1] will always equal
% the spectrogram's time resolution (dT). Note that in the first matrix
% form equation above, "dT" is not necessarily the spectrogram's time
% resolution, it simply means 'a time increment'.
% So, the first element in U can instead be written as tan(Beta(n-1))*dF.
% This altered form is effected by setting the first element in U to just
% be tan(Beta(n-1)) and setting the appropriate element in B to the
% spectrogram's frequency resolution (see initialisation of B below).
% -------------------------------------------------------------------------


    properties (Constant)

        % Values obtained from training data.
        % ---------------------------
        std_IC_P = 4.13715;         % Std. Dev. of intra-contour spectral
                                    % power (95th percentile value from
                                    % all training data).
        std_IC_RS = 2.38607;        % Std. Dev. of intra-contour ridge
                                    % strength (95th percentile value from
                                    % all training data).
        std_Beta_diff = 0.186433;   % Std. Dev. of differences in the "in" 
                                    % and "out" angles at ridge points. The
                                    % "in" angle is the (adjusted) angle
                                    % computed at the previous ridge point
                                    % from state update, and the "out"
                                    % angle is the observed "beta" at the
                                    % current ridge point.
        % ---------------------------
        % ---------------------------


        % Settings for some heuristics checks
        % ---------------------------
        % When looking for possible extensions to a track at the current
        % frame, if the cost (Mahalanobis distance) of extending the track
        % with a detected ridge point is larger than
        max_transition_cost = ((3 ^ 2) * 3);
        % then the ridge point is not considered as a candidate for
        % extension. The value is based on this reasoning -
        % There are 4 dimensions in the mahalanobis distance computation.
        % All points must be within 3*sigma of the respective means for a
        % good candidate. sqrt((3^2) * 3) is the minimum cost for points
        % that lie outside of 3*sigma on all dimensions. Note, that we omit
        % the final sqrt() w.r.t costs throughout the implementation.

        % When looking for possible extensions to a track at the current
        % frame, if the difference in Beta between the track's frontier
        % point and that of an observation is larger than
        max_angle_disparity = (90 * pi/180);   % 90 degrees in radians
        % then the ridge point is not considered as a candidate for
        % extension.

        % The penalty added to the cost function (Mahalanobis distance) for
        % tracks that are yet short is computed as
        short_track_penalty_factor = ((2 ^ 2) * 3);
        % times [2 ^ -(age - consecutiveDormantCount + 1)]. The smaller the
        % age, the larger the penalty. As the age gets very large, the
        % effect of the num of non-observed points becomes negligible. The
        % value above is based on this reasoning - 
        % Since the cost function is a factor of how many times the std.
        % dev. (in any dimension) a given point is away from the
        % (respective) mean, 3 x sigma in each dimention would be a very
        % worse case. Our choice of 2 x sigma makes the worst case penalty
        % (where a track is only one frame old) smaller than
        % max_transition_cost but yet it remains a high value to deter
        % splinters from hijacking good contour tracks. Note, that we omit
        % the final sqrt() w.r.t costs throughout the implementation.
        % ---------------------------
        % ---------------------------


        % Defaults for user-settable algorithm parameters
        % ---------------------------
        default_critical_activity = single(0.60);
        default_min_contour_length = uint8(8);
        default_max_contour_inactivity = uint8(4);
        % ---------------------------
        % ---------------------------

        scales = 1:2;               % Scale values to be used in spatial
                                    % smoothing within the ridge_dt member.

        % The upward and downward DELTA_F functions.
        % 'freq' is the point where you want the DELTA_F computed at and
        % 'd_t' is the time increment (in seconds) as used in computing
        % slope.
%        DeltaF_up = @(freq, d_t) (freq .* log10(freq .* 10000000) .* d_t);
        DeltaF_up = @(freq, d_t) (10^5 .* (sqrt(freq./5000)) .* d_t);
%        DeltaF_dn = @(freq, d_t) (freq .* log10(freq .* 1000) .* d_t);
        DeltaF_dn = @(freq, d_t) (10^5 .* (sqrt(freq./5000)) .* d_t);
    end

    properties (GetAccess = 'public', SetAccess = 'private')
        last_frame_idx = uint32(0); % Index to the last frame that has been
                                    % processed so far.

        % User-settable algorithm parameters
        CriticalActivity = single(0);
        MinContourLength = uint8(0);
        MaxContourInactivity = uint8(0);

        tracking_start_time = 0.0;  % To help offset reported track times
    end

    properties (GetAccess = 'private', SetAccess = 'private')
        % Kalman filter elements (with prefix "kf_")
        kf_A = [];
        kf_B = [];
        kf_Q = [    0.703272	  0.00947098;
                  0.00947098	     2.50041];
                                    % Process errors from training

        kf_R = [     2.45445               0;
                           0         6.91595];
                                    % Measurement errors. The diagonal
                                    % values are the 95th percentile of the
                                    % intra-contour std dev of respective
                                    % quantities. The cross terms are
                                    % assumed to not play much of a role.
        kf_H = [];

        mahalanobis_dist_std = [];  % Std. dev. values for computation of
                                    % Mahalanobis distances

        PSD_dF = 0;                 % Frequency resolution of spectrogram
        PSD_dT = 0;                 % Time resolution of spectrogram

        f_angle_lim_up = [];        % Upward limiting angles at each
                                    % frequency point on frequency axis.

        f_angle_lim_dn = [];        % Downward limiting angles at each
                                    % frequency point on frequency axis.

        pwr_spec_f = [];            % 1D array containing the frequency
                                    % axis values (center bin frequencies)
                                    % of input spectrogram.

        num_tracks = uint32(0);     % Number of tracked contours at any
                                    % time

        % Initialise container to store active tracks.
        % Create an empty array of tracks.
        tracks = struct(...
            'age', {}, ...
            'totalActiveCount', {}, ...
            'consecutiveDormantCount', {}, ...
            'kf_x', {}, ...
            'kf_P', {}, ...
            'kf_u', {}, ...
            'path', {});
        % NOTE: Any change made to the above structure warrants similar
        % change in the CreateNewTracks() method below.

        ridge_dt = {};              % Handle to an instance of the
                                    % dt_SpectrogramRidgeDetector class.

        % Callback for reporting detected (completed/finished) tracks
        report_tracks_callback = NaN;
        report_tracks_callback_data = NaN;

        % Placeholders for anonymous functions (defined in constructor)
        f_bin_to_freq = @() (0);
        get_short_track_penalty = @() (0);
    end
    
    %======================================================================
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %------------------------------------------------------------------
        function obj = dt_SpectrogramRidgeTracker(PSD_dT, PSD_dF, f_bins)
        % Constructor

            % Set the defaults for user-settable algorithm parameters
            obj.CriticalActivity = obj.default_critical_activity;
            obj.MinContourLength = obj.default_min_contour_length;
            obj.MaxContourInactivity = obj.default_max_contour_inactivity;

            % Create the Kalman model's matrices, and ..
            obj.kf_A = eye(2);
            obj.kf_B = zeros(2, 2);
            obj.kf_H = eye(2);
            % .. set specific values appropriately.
            obj.kf_B(1, 1) = 1;

%             obj.mahalanobis_dist_std = [sqrt(obj.kf_Q(1, 1)), ...
%                     obj.std_IC_P, ...
%                     obj.std_IC_RS, ...
%                     obj.std_Beta_diff];
            obj.mahalanobis_dist_std = [sqrt(obj.kf_Q(1, 1)), ...
                    sqrt(obj.kf_Q(2, 2)), ...
                    obj.std_Beta_diff];

            obj.PSD_dF = PSD_dF;
            obj.PSD_dT = PSD_dT;

            % Pre-determine the upward and downward limiting angles at each
            % frequency point on the frequency axis.
            obj.f_angle_lim_up = atan((1 / PSD_dF) * obj.DeltaF_up(f_bins, PSD_dT));
            obj.f_angle_lim_dn = atan((-1 / PSD_dF) * obj.DeltaF_dn(f_bins, PSD_dT));
            
            obj.pwr_spec_f = f_bins;

            % Function to convert indexes to frequency bins into
            % corresponding frequency values.
            temp = f_bins(1);
            obj.f_bin_to_freq = @(bin_idxs) (temp + ((bin_idxs - 1) .* PSD_dF));

            % Function to compute penalties for infant tracks so that they
            % do not hijack points corresponding to long-running tracks.
            % The goal is to issue a penalty of
            % "short_track_penalty_factor" to a track that is one frame old
            % and reduce the penalty down to (1 std * N-d) at the 4th frame
            temp = -((1-4)^2) / log(length(obj.mahalanobis_dist_std) / obj.short_track_penalty_factor);
            obj.get_short_track_penalty = @(real_ages) (obj.short_track_penalty_factor .* exp(-((1 - real_ages) .^ 2) ./ temp));

            % Ridge detector instance
            obj.ridge_dt = dt_SpectrogramRidgeDetector(length(f_bins), obj.scales);

            obj.Init();
        end

        %------------------------------------------------------------------
        function delete(obj)
        % Destructor
            delete(obj.ridge_dt);
        end

        %------------------------------------------------------------------
        function Reset(obj, new_start_time)
        % Reset stuff. Useful when there are breaks in the input stream.

            % At this point, the active tracks should have all been already
            % reported by calling Flush(). If there still any tracks
            % remaining, simply chuck them now since we don't know what to
            % do with them.
            obj.tracks(:) = [];
            obj.num_tracks = uint32(0);

            % Reset the ridge detector
            obj.ridge_dt.Reset();

            obj.Init();

            if nargin > 1
                obj.SetTrackingStartTime(new_start_time);
            end
        end

        %------------------------------------------------------------------
        function Flush(obj)
        % Flush out existing buffers. To be called at the end of input
        % stream.

            % Invoke the ridge detector's Flush() functionality
            [Ridge_freq_idxs, Ridge_frame_idxs, spec_pwr, RS, Beta, num_processed_frames] = ...
                obj.ridge_dt.Flush();

            obj.ProcessRidgeDetectorResults(Ridge_freq_idxs, Ridge_frame_idxs, 10*log10(spec_pwr), 10*log10(RS), Beta, num_processed_frames);

            % Throw away very short tracks (if any)
            throwaway_mask = ([obj.tracks(:).age] < obj.MinContourLength);
            obj.tracks(throwaway_mask) = [];
            obj.num_tracks = obj.num_tracks - sum(throwaway_mask);

            if obj.num_tracks == 0
                return;
            end

            % If there still are any active tracks, report them now.
%             % Rather than repeating most of the operations already done in
%             % HandleInactiveTracks(), I perform a quick & dirty hack:
%             %   - Temporarily set the MaxContourInactivity to 0
%             %   - Invoke HandleInactiveTracks()
%             %   - Set MaxContourInactivity back to what it was
%             % With MaxContourInactivity = 0 within HandleInactiveTracks(),
%             % any remaining active tracks will get reported.
%             temp = obj.MaxContourInactivity;
%             obj.MaxContourInactivity = uint8(0);
            obj.HandleInactiveTracks(1:obj.num_tracks);
%             obj.MaxContourInactivity = temp;

%             % Now, throw away the remaining tracks
%             obj.tracks(:) = [];
%             obj.num_tracks = uint32(0);
        end

        %------------------------------------------------------------------
        function SetCallback(obj, callback_fn_handle, callback_data)
        % Set up the info necessary for reporting detected contour tracks.
        %   callback_fn_handle      Handle to a callback function
        %   callback_data           Data passed to the callback function
        %                           along with the detected tracks.

            if ~isa(callback_fn_handle, 'function_handle')
                error('dt_SpectrogramRidgeTracker:SetCallback', ...
                    'The callback parameter must be a ''function_handle''');
            end

            obj.report_tracks_callback = callback_fn_handle;
            obj.report_tracks_callback_data = callback_data;
        end

        %------------------------------------------------------------------
        function SetTrackingStartTime(obj, start_time)
            obj.tracking_start_time = double(start_time);
        end

        %------------------------------------------------------------------
        function ProcessFrames(obj, specgram_frames)
        % Process incoming spectrogram frames.

            % Invoke the ridge detector
            [Ridge_freq_idxs, Ridge_frame_idxs, spec_pwr, RS, Beta, num_processed_frames] = ...
                obj.ridge_dt.ProcessFrames(specgram_frames);

            obj.ProcessRidgeDetectorResults(Ridge_freq_idxs, Ridge_frame_idxs, 10*log10(spec_pwr), 10*log10(RS), Beta, num_processed_frames);
        end

        %------------------------------------------------------------------
        function SetCriticalActivity(obj, val)
            if ~(isnumeric(val))
                error('dt_SpectrogramRidgeTracker:SetCriticalActivity', ...
                    'CriticalActivity must be a numeric value')
            end
            obj.CriticalActivity = cast(val, class(obj.default_critical_activity));
        end % CriticalActivity
        function SetMinContourLength(obj, val)
            if ~isnumeric(val)
                error('dt_SpectrogramRidgeTracker:SetMinContourLength', ...
                    'MinContourLength must be a numeric value')
            end
            obj.MinContourLength = cast(val, class(obj.default_min_contour_length));
        end % MinContourLength
        function SetMaxContourInactivity(obj, val)
            if ~isnumeric(val)
                error('dt_SpectrogramRidgeTracker:SetMaxContourInactivity', ...
                    'MaxContourInactivity must be a numeric value')
            end
            obj.MaxContourInactivity = cast(val, class(obj.default_max_contour_inactivity));
        end % MaxContourInactivity
        
        %------------------------------------------------------------------
        function Set_min_intensity(obj, new_min_intensity)
            obj.ridge_dt.Set_min_intensity(new_min_intensity);
        end
        function Set_threshold_value(obj, new_threshold_value)
            obj.ridge_dt.Set_threshold_value(new_threshold_value);
        end

    end     % End public methods
    
    %======================================================================
    methods (Access = 'private')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Private Methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %------------------------------------------------------------------
        function Init(obj)
        % Initialise variables before start of streaming.

            obj.last_frame_idx = uint32(obj.ridge_dt.process_lag);  % To account for the initial lag
        end

        %------------------------------------------------------------------
        function ProcessRidgeDetectorResults(obj, Ridge_freq_idxs, Ridge_frame_idxs, spec_pwr, RS, Beta, num_processed_frames)
        % Handle the sparse data returned from the ridge detector.

            % Ignore points where the angle is too large. Such points
            % correspond to highly-broadband interferences (mostly clicks)
            % and hence can be safely ignored.
            good_angles_idxs = find((Beta <= obj.f_angle_lim_up(Ridge_freq_idxs)) & ...
                (Beta >= obj.f_angle_lim_dn(Ridge_freq_idxs)));
            good_angles_Ridge_frame_idxs = Ridge_frame_idxs(good_angles_idxs);

            for frame_idx = 1:num_processed_frames

                frame_time = obj.tracking_start_time + ...
                    (double(obj.last_frame_idx + frame_idx - 1) * obj.PSD_dT);

                % Ridge points with good angles in the current frame
                frame_pts_idxs = good_angles_idxs( ...
                    (good_angles_Ridge_frame_idxs == (obj.last_frame_idx + frame_idx)));
                num_frame_obs = length(frame_pts_idxs);

                if num_frame_obs > 0
                    [keep_mask, ignore_mask, local_max_idxs] = ...
                        obj.GroupSparseData(Ridge_freq_idxs(frame_pts_idxs), RS(frame_pts_idxs));

                    % Create 'observation' points from each valid ridge point
                    % in the current frame.
                    % Each row is one 'observation' vector -> [F_idx, P, Beta].
                    current_observations = [ ...
                        double(Ridge_freq_idxs(frame_pts_idxs)), ...
                        spec_pwr(frame_pts_idxs), ...
                        Beta(frame_pts_idxs)];
                else
                    ignore_mask = false(0, 1);
                    keep_mask = ignore_mask;
                    local_max_idxs = zeros(0, 1);
                    current_observations = zeros(0, 3);
                end

                if obj.num_tracks > 0
                    obj.PredictNewLocationsOfTracks();

                    if num_frame_obs == 0 %all(ignore_mask)
                        assignments = zeros(obj.num_tracks, 1, 'uint16');
                    else
                        assignments = obj.MapObservationsToTracks(current_observations, ignore_mask);

                        if any(assignments)
                            obj.UpdatePredictionsWithMappedObservations(frame_time, current_observations, assignments);

                            % After the below operation, groups of observations
                            % identified by the local maxima will be marked as "used"
                            % if at least one observation in the group was assigned to
                            % a track. Only the local maxima in any "unused" groups
                            % will be considered as "yet unused/unassigned" and will
                            % eventually be used in starting new tracks.
                            keep_mask(local_max_idxs(assignments((assignments > 0)))) = false;
                        end
                    end

                    if any(~assignments)%~all(assignments)
                        inactiveTrackIdxs = obj.UpdateUnassignedTracks(frame_time, assignments);

                        if ~isempty(inactiveTrackIdxs)
                            obj.HandleInactiveTracks(inactiveTrackIdxs);
                        end
                    end
                end

                % Create new tracks that start at the unused observations
                % (if any)
                if any(keep_mask)
                    obj.CreateNewTracks(frame_time, current_observations, keep_mask);
                end
            end
            
            obj.last_frame_idx = obj.last_frame_idx + num_processed_frames;
        end

        %------------------------------------------------------------------
        function PredictNewLocationsOfTracks(obj)
        % KALMAN FILTER STEP: State Prediction and Covariance Prediction.
        %
        % Predict the position of next point for each active track and
        % predict how much error.

            for i = 1:obj.num_tracks
                % Prediction for state vector
                % obj.tracks(i).kf_x = (obj.kf_A * obj.tracks(i).kf_x) + (obj.kf_B * obj.tracks(i).kf_u);
                % Shortcut, since kf_A is an identity matrix
                obj.tracks(i).kf_x = obj.tracks(i).kf_x + (obj.kf_B * obj.tracks(i).kf_u);
                
                % Prediction for covariance vector
                % obj.tracks(i).kf_P = (obj.kf_A * obj.tracks(i).kf_P * obj.kf_A') + obj.kf_Q;
                % Shortcut, since kf_A is an identity matrix
                obj.tracks(i).kf_P = obj.tracks(i).kf_P + obj.kf_Q;
            end
        end

        %------------------------------------------------------------------
        function UpdatePredictionsWithMappedObservations(obj, frame_time, observations, assignments)
		% KALMAN FILTER STEPS: Innovation, Innovation Covariance, Kalman
		%                      Gain, State Update, Covariance Update
		%
		% Based on the recommended global optimal "assignments", performs
		% the remaining steps of Kalman filtering. Performs housekeeping
		% on the tracks following the updation by the Kalman filtering
		% steps.

            % Deal with only those tracks for which extensions have been
            % identified.
            assignedTracksIdxs = find(assignments > 0);

            uint8_zero = uint8(0);
            for itr = 1:size(assignedTracksIdxs, 1)
                trackIdx = assignedTracksIdxs(itr);
                curr_track = obj.tracks(trackIdx);
                curr_obs = observations(assignments(trackIdx), 1:3)';
                curr_angle_tan = tan(observations(assignments(trackIdx), 3));

                % Kalman filter steps:
                % Correct the estimate of the track new position using the
                % new peak.
                % ---------------------------------------------------------
                % Innovation covariance ->   S = (H * P * H') + R
                % kf_S = (obj.kf_H * curr_track.kf_P * obj.kf_H') + obj.kf_R;
                % Shortcut, since kf_H is an identity matrix
                kf_S = curr_track.kf_P + obj.kf_R;

                % Kalman gain ->   K = P * H' * inv(S)
                % kf_K = (curr_track.kf_P * obj.kf_H') / kf_S;
                % Shortcut, since kf_H is an identity matrix
                kf_K = curr_track.kf_P / kf_S;

                % Correction based on observation ->
                %     Xn = Xp + (K * (zn - (H * Xp)))    and
                %     Pn = (I - (K * H)) * Pp)
                % new_kf_x = ...
                %     curr_track.kf_x + (kf_K * (curr_obs - (obj.kf_H * curr_track.kf_x)));
                % obj.tracks(trackIdx).kf_P = ...
                %     curr_track.kf_P - (kf_K * obj.kf_H * curr_track.kf_P);
                % Shortcut, since kf_H is an identity matrix
                new_kf_x = ...
                    curr_track.kf_x + (kf_K * (curr_obs(1:2) - curr_track.kf_x));
                obj.tracks(trackIdx).kf_P = ...
                    curr_track.kf_P - (kf_K * curr_track.kf_P);
                % ---------------------------------------------------------
                obj.tracks(trackIdx).kf_x = new_kf_x;

                % Housekeeping updates:
                % For the control U, rather than simply taking the
                % (tan(B)*dT), angle adjustment is made to alter the angle
                % to the next prediction point from the predicted-adjusted
                % point.
                obj.tracks(trackIdx).kf_u(1) = curr_angle_tan + ...
                    (curr_obs(1) - new_kf_x(1));    % Control vector for next prediction
                obj.tracks(trackIdx).path = [curr_track.path, [frame_time; new_kf_x; NaN]];

                % Increase track's age
                obj.tracks(trackIdx).age = curr_track.age + 1;

                % Update track's activity info
                obj.tracks(trackIdx).totalActiveCount = curr_track.totalActiveCount + 1;
                obj.tracks(trackIdx).consecutiveDormantCount = uint8_zero;
            end
        end

        %------------------------------------------------------------------
        function [inactive_tracks_idxs] = UpdateUnassignedTracks(obj, frame_time, assignments)
        % Perform housekeeping on tracks that couldn't be extended at the
        % current frame. Mark tracks for reporting and discarding as well.
        % -x-   No Kalman filtering steps performed in this function.   -x-

            unassignedTracksIdxs = find(assignments == 0);  % Get the tracks for which no extension were found
            working_set_size = length(unassignedTracksIdxs);

            % Gather tracks' data in the working set (ws)
            ws_ages = zeros(working_set_size, 1, 'uint32');
            ws_consecutiveDormantCounts = zeros(working_set_size, 1, 'uint8');
            ws_totalActiveCounts = zeros(working_set_size, 1, 'uint32');
%             pred_bin_idxs = zeros(working_set_size, 1);
%             pred_angles = zeros(working_set_size, 1);
            for i = 1:working_set_size
                trackIdx = unassignedTracksIdxs(i);

                ws_ages(i) = obj.tracks(trackIdx).age;
                ws_consecutiveDormantCounts(i) = obj.tracks(trackIdx).consecutiveDormantCount;
                ws_totalActiveCounts(i) = obj.tracks(trackIdx).totalActiveCount;
%                 pred_bin_idxs(i) = obj.tracks(trackIdx).kf_x(1);
%                 pred_angles(i) = obj.tracks(trackIdx).kf_u(1);    % Copy value now, apply atan() later
            end
%             pred_angles = atan(pred_angles);
            % Update age and dormancy counts
            ws_ages = ws_ages + 1;
            ws_consecutiveDormantCounts = ws_consecutiveDormantCounts + 1;
%            real_ages = ages - uint32(consecutiveDormantCounts);    % without the trailing predictions

            % A track's activity is measured as the fraction of its age
            % for which it has remained active ('active' meaning having
            % actual observations, not predicted extensions).
            % i.e., activity = track.totalActiveCount / track.age

            % We need to nip those bad ones in the bud. A "bad" track is
            % one that isn't long enough yet but already contains many
            % predicted extensions.
            % Find the tracks that are not "bad".
            ws_bad_starts_mask = ((ws_ages < obj.MinContourLength) & ...
                ((single(ws_totalActiveCounts) ./ single(ws_ages)) < obj.CriticalActivity));

            % Contours that were extended with predictions and straying
            % outside the spectrgram bandwidth without any possibility of
            % returning into the bandwidth need not be tracked any longer.
%             straying_tracks_mask = (consecutiveDormantCounts > 0) & ...
%                 (pred_bin_idxs < 1 | pred_bin_idxs > length(obj.pwr_spec_f));
%             [upper_bin_lim, lower_bin_lim] = ...
%                 obj.GetLenienceLimits(pred_angles(straying_tracks_mask));    % Get the freq bin lenience limits
%             straying_tracks_mask(straying_tracks_mask) = ...
%                 ((pred_bin_idxs(straying_tracks_mask) + upper_bin_lim) < 0) | ...
%                 ((pred_bin_idxs(straying_tracks_mask) + lower_bin_lim) > length(obj.pwr_spec_f));

            % Identify tracks that have been "dormant" for long, i.e., has
            % more than max allowed predicted extensions at the frontier.
            ws_long_dormant_tracks_mask = ...
                (ws_consecutiveDormantCounts >= obj.MaxContourInactivity);

            ws_active_tracks_mask = ~(ws_bad_starts_mask | ws_long_dormant_tracks_mask);
            ws_active_subset_idxs = find(ws_active_tracks_mask);
            active_tracks_idxs = unassignedTracksIdxs(ws_active_tracks_mask);
            working_set_size = length(active_tracks_idxs);

            for i = 1:working_set_size
                trackIdx = active_tracks_idxs(i);
                data_idx = ws_active_subset_idxs(i);

                % Housekeeping Updates:
                % Control vector U doesn't change, others are updated.
                % The assumption is that drastic amplitude modulation does
                % not occur when drastic frequency modulation is in
                % progress. So the unchanged control vector will enable
                % continued prediction following the last observed slope
                % for rapidly attenuating low-FM signals.
                obj.tracks(trackIdx).path = [obj.tracks(trackIdx).path, ...
                    [frame_time; obj.tracks(trackIdx).kf_x; obj.tracks(trackIdx).path(4, end)]];
                    %[frame_time; obj.tracks(trackIdx).kf_x(1); NaN; NaN]];

                % Increase track's age
                obj.tracks(trackIdx).age = ws_ages(data_idx);

                % Update track's activity info
                obj.tracks(trackIdx).consecutiveDormantCount = ws_consecutiveDormantCounts(data_idx);
            end

            inactive_tracks_idxs = unassignedTracksIdxs(~ws_active_tracks_mask);
        end

        %------------------------------------------------------------------
        function HandleInactiveTracks(obj, inactive_tracks_idxs)
        % Find and deal with tracks that are no longer "active".
        % Gather tracks that have been inactive for too many consecutive
        % frames and report them at once (invoke the callback).
        % Also delete recently created tracks that have little real
        % activity.
        % -x-   No Kalman filtering steps performed in this function.   -x-

            working_set_size = length(inactive_tracks_idxs);

            % Gather tracks' data in the working set (ws)
            ws_ages = zeros(working_set_size, 1, 'uint32');
            ws_consecutiveDormantCounts = zeros(working_set_size, 1, 'uint8');
            for i = 1:working_set_size
                trackIdx = inactive_tracks_idxs(i);

                ws_ages(i) = obj.tracks(trackIdx).age;
                ws_consecutiveDormantCounts(i) = obj.tracks(trackIdx).consecutiveDormantCount;
            end
            ws_real_ages = ws_ages - uint32(ws_consecutiveDormantCounts);

            % Create a container of tracks that are to be reported
            reportable_tracks_mask = ws_real_ages >= obj.MinContourLength;
            reportable_tracks_idxs = inactive_tracks_idxs(reportable_tracks_mask);
            ws_reportable_subset_idxs = find(reportable_tracks_idxs);
            outputs = cell(length(reportable_tracks_idxs), 1);
            for itr = 1:length(reportable_tracks_idxs)
                trackIdx = reportable_tracks_idxs(itr);
                % Copy the first three rows (T, F & P), up to the column
                % with the last non-predicted extension.
                num_good_pts = ws_real_ages(ws_reportable_subset_idxs(itr));
                outputs{itr} = zeros(3, num_good_pts);
                outputs{itr}([1 3], :) = obj.tracks(trackIdx).path([1 3], 1:num_good_pts);
                outputs{itr}(2, :) = obj.f_bin_to_freq(obj.tracks(trackIdx).path(2, 1:num_good_pts));
            end

            % Get rid of reported tracks and bad starts
%             not_throwaway_mask = ~(bad_starts_mask | long_dormant_tracks_mask);
%             % The above logical condition is equivalent to
%             %       ~(bad_mask | (good_mask & dormant_mask))
% %             obj.tracks(throwaway_mask) = [];
% %             obj.num_tracks = obj.num_tracks - sum(throwaway_mask);
%             obj.tracks = obj.tracks(not_throwaway_mask);
%             obj.num_tracks = sum(not_throwaway_mask);
            obj.num_tracks = obj.num_tracks - working_set_size;
            obj.tracks(inactive_tracks_idxs) = [];

            % Report the finished tracks
            if ~isempty(reportable_tracks_idxs) && isa(obj.report_tracks_callback, 'function_handle')
                obj.report_tracks_callback(obj.report_tracks_callback_data, outputs);
            end
        end

        %------------------------------------------------------------------
        function CreateNewTracks(obj, frame_time, observations, unused_obs_mask)
        % Create new tracks from unassigned observations.
        % -x-   No Kalman filtering steps performed in this function.   -x-

            unused_obs_idxs = find(unused_obs_mask);

            % Discard observations for which the lenience lookup lies
            % outside the spectrogram bandwidth
            start_bins = observations(unused_obs_idxs, 1);
            [upper_bin_lim, lower_bin_lim] = ...
                obj.GetLenienceLimits(observations(unused_obs_idxs, 3));    % Get the freq bin lenience limits
            unused_obs_idxs = unused_obs_idxs( ...
                (start_bins + upper_bin_lim) > 0 & (start_bins + lower_bin_lim) <= length(obj.pwr_spec_f));

            num_new_tracks = length(unused_obs_idxs);

            if num_new_tracks > 0
                existing_tracks = obj.num_tracks;

                % Create placeholders for as many new tracks
                % NOTE: Make sure that the structure fields are exactly the
                % same and are in the same order as that of the definition
                % of obj.tracks made in the 'class properties' above.
                obj.tracks((end+1):(end+num_new_tracks)) = struct(...
                    'age', uint32(1), ...
                    'totalActiveCount', uint32(1), ...
                    'consecutiveDormantCount', uint8(0), ...
                    'kf_x', [0; 0], ...
                    'kf_P', [0 0; 0 0], ...
                    'kf_u', [0; 0], ...
                    'path', [0; 0; 0; 0]);

                % Update the fields
                for itr = 1:num_new_tracks
                    curr_obs = observations(unused_obs_idxs(itr), :)';

                    new_track_idx = existing_tracks + itr;
                    obj.tracks(new_track_idx).kf_x = curr_obs(1:2);
                    obj.tracks(new_track_idx).kf_u(1) = tan(curr_obs(3));
                    obj.tracks(new_track_idx).path = [frame_time; curr_obs(1:2); NaN];
                end

                obj.num_tracks = obj.num_tracks + num_new_tracks;   % Update track count
            end
        end

        %------------------------------------------------------------------
        function [assignments] = MapObservationsToTracks(obj, observations, ignore_obs_mask)
        % Determine best extensions to active tracks based on prediction
        % information. Uses prediction information and other heuristics,
        % and eventually, the Hungarian Algorithm to find the global best
        % assignment of peaks to active tracks.
		% The returned 'assignments' is a column vector containing one
		% entry per track -> a zero indicates that no extension was found
		% for the corresponding track and a positive value is an index to
		% the observation that was found to be the best extension.
        % -x-   No Kalman filtering steps performed in this function.   -x-

            nTracks = obj.num_tracks;
            nPeaks = size(observations, 1); % Each row is supposed to be one observation vector -> [F_idx, P, Beta]

            last_freq_bin_idxs = zeros(nTracks, 1);
            adjusted_ages = zeros(nTracks, 1);

            % Build a list of prediction points, one row per track
            prediction_points = zeros(nTracks, 3);
            fake_extns_mask = false(nTracks, 1);    % True if the last extension was a predicted one
            for track_idx = 1:nTracks
                prediction_points(track_idx, 1:2) = obj.tracks(track_idx).kf_x;
                prediction_points(track_idx, 3) = obj.tracks(track_idx).kf_u(1);    % Copy value now, apply atan() later

                curr_track_path = obj.tracks(track_idx).path;
                last_freq_bin_idxs(track_idx) = curr_track_path(2, end);

%                 % Get the mean(s) of power and RS of the active contour
%                 % as the respective value of the prediction vector.
%                 prediction_points(track_idx, 2:3) = ...
%                     nanmean(curr_track_path(3:4, :), 2);

                adjusted_ages(track_idx) = double(obj.tracks(track_idx).age) - double(obj.tracks(track_idx).consecutiveDormantCount);

                fake_extns_mask(track_idx) = (obj.tracks(track_idx).consecutiveDormantCount > 0);
            end
            prediction_points(:, 3) = atan(prediction_points(:, 3));    % convert to angle

            % Convert bin indices to actual frequency values
            last_freq_vals = obj.f_bin_to_freq(last_freq_bin_idxs);

            % The max possible +ve & -ve frequency range are obtained
            % from DELTA_F, and the slope-specific lenience angles are
            % also determined. The more restrictive combination arising
            % out of the pairs of upper and lower limits are used for
            % rejection.
            [upper_bin_lim, lower_bin_lim] = ...
                obj.GetLenienceLimits(prediction_points(:, 3));    % Get the freq bin lenience limits
            upper_bin_lim = last_freq_bin_idxs + ...
                min(obj.DeltaF_up(last_freq_vals, obj.PSD_dT) .* (1/obj.PSD_dF), upper_bin_lim);
            lower_bin_lim = last_freq_bin_idxs + ...
                max(-obj.DeltaF_dn(last_freq_vals, obj.PSD_dT) .* (1/obj.PSD_dF), lower_bin_lim);

            % Computation of penalties
            penalties = obj.get_short_track_penalty(adjusted_ages);

            inv_ignore_obs_mask = ~ignore_obs_mask;

            % Identify valid track-observation pairs
            valid_pairs = false(nTracks, nPeaks);
            obs_col_1 = observations(inv_ignore_obs_mask, 1);
            obs_col_3 = observations(inv_ignore_obs_mask, 3);
            for track_idx = 1:nTracks
                % Where the observation points are beyond the allowed
                % ranges, ignore such observations.
                valid_pairs(track_idx, inv_ignore_obs_mask) = ...
                    obs_col_1 >= lower_bin_lim(track_idx) & obs_col_1 <= upper_bin_lim(track_idx) & ...
                    abs(obs_col_3 - prediction_points(track_idx, 3)) < obj.max_angle_disparity;
            end

            [track_idxs, obs_idxs] = find(valid_pairs);
            if ~isempty(track_idxs)
                % Initialize the costs to zeros to begin with.
                valid_costs = zeros(length(track_idxs), 1);

                % Expand the mask to match track_idxs & obs_idxs.
                fake_extns_mask = fake_extns_mask(track_idxs);

                % Compute the cost of assigning each peak to each track.
                valid_costs(~fake_extns_mask) = ...
                    dt_SpectrogramRidgeTracker.ComputeTransitionCosts( ...
                        obj.mahalanobis_dist_std, ...
                        prediction_points(track_idxs(~fake_extns_mask), :), ...
                        observations(obs_idxs(~fake_extns_mask), :));
                valid_costs(fake_extns_mask) = (2 ^ 2) + ...
                    dt_SpectrogramRidgeTracker.ComputeTransitionCosts( ...
                        obj.mahalanobis_dist_std([1 3]), ...
                        prediction_points(track_idxs(fake_extns_mask), [1 3]), ...
                        observations(obs_idxs(fake_extns_mask), [1 3]));

                % Add the per-track penalties
                valid_costs = valid_costs + penalties(track_idxs');

                % Mark higher cost ones for non-use
                valid_costs(valid_costs > obj.max_transition_cost) = Inf;

                cost = Inf(nTracks, nPeaks);
                cost(track_idxs + ((obs_idxs - 1) .* double(nTracks))) = valid_costs;

                % Solve the least-cost assignment problem
                assignments = ut_Munkres(cost);
            else
                assignments = zeros(nTracks, 1, 'uint16');
            end
        end

    end     % End private methods
    
    %======================================================================
    methods (Static, Access = 'public')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Private Static Methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %------------------------------------------------------------------
        function [peak_mask, valley_mask, local_peak_idxs] = GroupSparseData(x_data, y_data)
        % For sparse points specified with co-ordinates x_data and y_data,
        % both being 1D vectors, group together any contiguous-lying points
        % and identify each such group with its local maxima.
        % The returned peak_mask and valley_mask are logical arrays
        % identifying points that are local maxima and minima,
        % respectively. Contiguous points (if any) around a local maxima,
        % up to the next 'valley points', form a group. For each input
        % point, local_peak_idxs contains the index of the local maxima in
        % the group that the point is part of.
            num_pts = length(x_data);
            valley_mask = false(num_pts, 1);
            local_peak_idxs = (1:num_pts)';
            peak_mask = valley_mask; % Initialise also to all 'false's to start with

            if num_pts > 0
                x_diff = (diff(x_data) == 1);   % Identify contiguous pts from dx
                temp = diff(y_data);    % dy

                rise = [(temp > 0 & x_diff); false];  % Points from where there is a rise
                fall = [false; (temp < 0 & x_diff)];  % Points that lie at a fall

                % Mark local maxima, neither a rise nor a fall
                peak_mask = (~(rise | fall));

                if any(rise) || any(fall)
                    peaks_idxs_helper = find(peak_mask);

                    temp = zeros(num_pts+1, 2);
                    d_p_i = [peaks_idxs_helper(1); diff(peaks_idxs_helper)];
                    temp([1; peaks_idxs_helper + 1], 1) = [d_p_i; num_pts - peaks_idxs_helper(end)];
                    temp(peaks_idxs_helper + 1, 2) = d_p_i;
                    temp(1, 2) = 0;
                    temp = cumsum(temp);
                    local_peak_idxs(rise) = temp(rise, 1);
                    local_peak_idxs(fall) = temp(fall, 2);

                    % Mark valley points, if any
                    valley_mask(rise & fall) = true;
                end
            end
        end
        
        %------------------------------------------------------------------
        function [upper_limits, lower_limits] = GetLenienceLimits(in_angles)
        % Returns the number of pixels to be allowed as lenience limits, in
        % both increasing and decreasing angles, for the angle(s) supplied
        % in in_angles.
        %
        % Input-
        %   in_angles:  must either be a scalar or a vector, not a matrix.
        %

            %                 60 deg -  |in_angle|*2/3
            %                 pi/3   -  |in_angle|*2/3
            lenience_angles = ((2 / 3) * (pi/2 - abs(in_angles)));
            % Yields -
            %   ± 60 deg    for   in_angle = 0 deg
            %   ± 90 deg    for   in_angle = ± 90 deg

            upper_limits = fix(tan(in_angles + lenience_angles)) + 1;
            lower_limits = fix(tan(in_angles - lenience_angles)) - 1;
        end
        
        %------------------------------------------------------------------
        function [costs] = ComputeTransitionCosts(prior_stats_stds, predictions, candidates)
        % Returns transition costs as the Mahalanobis distances from the
        % prediction (single, row vector) to each of the candidates. The
        % 'prediction' vector is considered to be the "mean" of the
        % distribution for which standard deviatoins were determied
        % beforehand using training data (provided as prior_stats_stds).
        %
        % NOTE that since the 'costs' are only used in comparisons, without
        % much meaning associated with the actual values, computing the
        % final sqrt() before returning the values is omitted.

            num_cands = size(candidates, 1);
            temp = 1 ./ prior_stats_stds;
            if size(predictions, 1) ~= num_cands
                costs = sum( ...
                    ( ...
                    (predictions(ones(num_cands, 1), :) - candidates) .* ...
                    temp(ones(num_cands, 1), :) ...
                    ) .^ 2, ...
                    2);
            else
                costs = sum( ...
                    ( ...
                    (predictions - candidates) .* temp(ones(num_cands, 1), :) ...
                    ) .^ 2, ...
                    2);
            end
        end
        
    end     % End public static methods

end
