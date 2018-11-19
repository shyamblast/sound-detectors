classdef dt_SpectrogramRidgeDetector < handle
    % dt_SpectrogramRidgeDetector - Class to encompass functionality for a
    % streaming interface ridge detector in spectrograms.
    
    properties (Constant)
        gamma = 3/4;                % Scale normalisation (see Lindeberg's
                                    % paper)

        thld_to_limit_map = [ ...   % Thresholds obtained from training
            0.193802, 0.351763;     % data. First & second columns are
            0.212647, 0.433697;     % cutoffs for ridge strength
            0.225765, 0.488305;     % (|Lpp_y_norm_norm|) and ridge
            0.235129, 0.528894;     % narrowness ratio (|Lpp-Lqq| / |Lpp|)
            0.242184, 0.569218;     % respectively. Rows are in increasing
            0.248262, 0.604500;     % order of threshold settings 1-8
            0.258638, 0.662637;     % corresponding to cutoff percentiles
            0.267228, 0.716297];    % of [97.5:-2.5:85, 80 75].
        default_threshold = 4;

        process_lag = uint32(5);    % Accounting for a 4 frame lag in
                                    % determining Lxy and a one frame lag
                                    % from Rs/Beta/zero-crossing
                                    % determining.
    end

    properties (GetAccess = 'public', SetAccess = 'private')
        threshold_value = 0;        % [Tuneable property. Default:
                                    % [class]::default_threshold].

        min_intensity = 0;          % [Tuneable property. Default: 0]
                                    % Cutoff value for minimum spectrogram
                                    % power.

        num_scales = 0;             % Number of different scales in the
                                    % scale-space.

        frame_no = uint32(0);       % Number of the first frame for which a
                                    % result could be returned from the
                                    % next invocation of ProcessFrames().

        max_kernel_delay = uint32(0);
                                    % The delay (in num frames) caused by
                                    % the largest scale.
    end

    properties (GetAccess = 'private', SetAccess = 'private')
        num_specgram_rows = uint32(0);
                                    % Number of rows in the input stream of
                                    % sepctrogram frames.

        smoothing_kernel = {};      % Cell array containing as many
                                    % 2D smoothing kernels as 'num_scales'.

        kernel_delay = [];          % [Array] Latencies produced by each
                                    % smoothing kernel.

        t_pow_gamma = [];           % [Array] 't' is scale^2. This value is
                                    % never directly used, but t^gamma is
                                    % used in a couple of computations.

        buffer = [];                % To store incoming frames until enough
                                    % become available for processing. It
                                    % stores up to 
                                    % (max_kernel_delay + process_lag) x 2
                                    % columns.
    end
    
    %======================================================================
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Public Methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function obj = dt_SpectrogramRidgeDetector(num_f_bins, scales)
        % Constructor
        %   num_f_bins      Number of rows in the input spectrogram frames.
        %   scales          An array of scale values (in increasing order).
        %                   Could be a single value too for single-scale
        %                   ridge extraction.

            % Initialise default properties
            obj.threshold_value = dt_SpectrogramRidgeDetector.default_threshold;
            obj.min_intensity = 0;

            obj.num_specgram_rows = uint32(num_f_bins);
            obj.num_scales = numel(scales);

            t = scales(:).^2;
            obj.t_pow_gamma = t .^ dt_SpectrogramRidgeDetector.gamma;

            obj.smoothing_kernel = cell(1, obj.num_scales);
            obj.kernel_delay = zeros(size(scales), 'uint32');

            obj.max_kernel_delay = 0;
            for scale_idx = 1:obj.num_scales
                % Initialise the smoothing kernel(s)
                obj.smoothing_kernel{scale_idx} = ...
                    dt_SpectrogramRidgeDetector.GetSmoothingKernelAtScale(scales(scale_idx));

                obj.kernel_delay(scale_idx) = uint32( ...
                    floor(length(obj.smoothing_kernel{scale_idx}) / 2));
            end
            obj.max_kernel_delay = max(obj.kernel_delay);
            
            obj.Init();
        end

        %------------------------------------------------------------------
        function delete(obj)
        % Destructor
            % Nothing to destroy.
        end

        %------------------------------------------------------------------
        function Reset(obj)
        % Reset stuff. Useful when there are breaks in the input
        % stream.
            obj.Init();
        end

        %------------------------------------------------------------------
        function [Ridge_freq_idxs, Ridge_frame_idxs, Pwr, RS, Beta, processed_frames] = Flush(obj)
        % Flush out existing buffers. To be called at the end of input
        % stream.
            [Ridge_freq_idxs, Ridge_frame_idxs, Pwr, RS, Beta, processed_frames] = ...
                obj.ProcessFrames(zeros(obj.num_specgram_rows, obj.max_kernel_delay));
        end

        %------------------------------------------------------------------
        function [Ridge_freq_idxs, Ridge_frame_idxs, Pwr, RS, Beta, processed_frames] = ProcessFrames(obj, specgram_frames)
        % Process incoming spectrogram frames. Returns info about detected
        % ridge points. The four return parameters are all 1D arrays of the
        % same length containing as many elements as ridge points detected,
        % and the meaning of each bit of info is described below.
        % Processing is performed across all specified scales and the
        % results are combined.
        % Inputs:
        %  specgram_frames   [obj.num_specgram_rows x N] array of N frames.
        %
        % Outputs:
        %  Ridge_freq_idxs   Indices to rows in the input spectrogram
        %                    stream.
        %  Ridge_frame_idxs  Indices to frames in the input spectrogram
        %                    stream. Continuous and non-decreasing.
        %  Pwr               Spectral power at the ridge point.
        %  RS                Measure of ridge strength at the ridge point.
        %  Beta              Angle of the non-dominant curvature at the
        %                    ridge point. Indicates the direction of
        %                    ridge's progress in time.
        %  processed_frames  Total number of frames (including frames in
        %                    both buffer and specgram_frames) that were
        %                    completely processed.

            [in_r, in_c] = size(specgram_frames);
            if in_r ~= obj.num_specgram_rows
                error('dt_SpectrogramRidgeDetector:ProcessFrames', ...
                    'Expecting %u rows in ''specgram_frames'', %i given', ...
                    obj.num_specgram_rows, in_r);
            end
            cols_in_buf = size(obj.buffer, 2);

            % See how many interior frames will be valid after convolution,
            % derivitaves and zero-crossing operations.
            % Total frames (buf + input) needed to produce K output frames
            %  ->  max_kernel_delay + [process_lag + K + process_lag] + max_kernel_delay
            if (in_c + cols_in_buf) <= ((obj.max_kernel_delay + obj.process_lag) * 2)    % If enough frames aren't available
                % Save the current supply
                obj.buffer = [obj.buffer, specgram_frames];
                
                Ridge_freq_idxs = []; Ridge_frame_idxs = []; Pwr = []; RS = []; Beta = [];
                processed_frames = uint32(0);
                return;
            end

            processed_frames = (in_c + cols_in_buf) - ((obj.max_kernel_delay + obj.process_lag) * 2);

            scale_R_pts = cell(1, obj.num_scales);    % Container to store the detected ridge points (linear indexes)
            scale_Beta_q = cell(1, obj.num_scales);   % Container to store angles at ridge tops for points in r_pts
            scale_RS = cell(1, obj.num_scales);   % Container to store the ridge strengths (full frame)
            scale_points = cell(1, obj.num_scales); % Container to store the scale space values at the detected ridge points
            for scale_idx = 1:obj.num_scales
                % Convolution with the Gaussian kernel for smoothing.
                % buffer & specgram_frames are trimmed below in order to
                % ensure that the valid frames after processing
                % (convolution + derivatives + zero crossing) will be the
                % same set of interior frames at all scales.
                temp = obj.max_kernel_delay + obj.process_lag + processed_frames + obj.process_lag + obj.kernel_delay(scale_idx);
                current_scale_space = conv2(...
                    obj.smoothing_kernel{scale_idx}, obj.smoothing_kernel{scale_idx}, ...
                    [obj.buffer(:, (obj.max_kernel_delay+1-obj.kernel_delay(scale_idx)):min(cols_in_buf, temp)), specgram_frames(:, 1:max(0, (in_c - int32(obj.max_kernel_delay - obj.kernel_delay(scale_idx)))))], ...
                    'same');

                % The inputs to the convolution above already contains
                % extra columns on either side to cope for kernel delays.
                % Since the outputs are expected to be returned as the
                % 'same' size, we need to trim it out to retain only the
                % valid total_valid_cols frames here.
                [scale_R_pts{scale_idx}, scale_Beta_q{scale_idx}, scale_RS{scale_idx}] = ...
                    dt_SpectrogramRidgeDetector.ProcessScaleSpace( ...
                        current_scale_space(:, (obj.kernel_delay(scale_idx)+1):(end-obj.kernel_delay(scale_idx))), ...
                        obj.t_pow_gamma(scale_idx), ...
                        obj.threshold_value, obj.min_intensity);
                % scale_RS has one extra column on either side for use in
                % nonmax_suppress().
                % scale_R_pts are linear indices and start at the interior
                % columns in scale_RS.

                scale_points{scale_idx} = current_scale_space(((obj.kernel_delay(scale_idx)+4) * in_r) + scale_R_pts{scale_idx});
            end
            
            RS_full = zeros(obj.num_specgram_rows, processed_frames + 2);
            salient_scale_points = ones(obj.num_specgram_rows, processed_frames + 2);
            if obj.num_scales > 1
                % Scale-maximal checks: Compare against the neighbouring
                % scales to see if we have a local maximum in the scale
                % domain. If there are multiple local maxima, choose the
                % highest among them.

                best_pts_mask = false(obj.num_specgram_rows, processed_frames + 2);
                max_saliency_Beta_q = zeros(obj.num_specgram_rows, processed_frames + 2);

                % First one.
                % Check RS against next higher scale at current scale's
                % ridge points
                curr_scale_pts = scale_R_pts{1};
                keep_these = scale_RS{1}(curr_scale_pts) >= scale_RS{2}(curr_scale_pts);
                % Update
                RS_full(curr_scale_pts(keep_these)) = scale_RS{1}(curr_scale_pts(keep_these));
                salient_scale_points(curr_scale_pts(keep_these)) = scale_points{1}(keep_these);
                max_saliency_Beta_q(scale_R_pts{1}(keep_these)) = scale_Beta_q{1}(keep_these);
                best_pts_mask(scale_R_pts{1}(keep_these)) = true;

                % Inbetween ones.
                for scale_idx = 2:(obj.num_scales-1)
                    % At current scale's ridge points, compare RS against
                    % neighbouring scales and also against RS_full to
                    % ensure global maximum.
                    curr_scale_pts = scale_R_pts{scale_idx};
                    keep_these = scale_RS{scale_idx}(curr_scale_pts) >= scale_RS{scale_idx+1}(curr_scale_pts) & ...
                        scale_RS{scale_idx}(curr_scale_pts) > scale_RS{scale_idx-1}(curr_scale_pts) & ...
                        scale_RS{scale_idx}(curr_scale_pts) > RS_full(curr_scale_pts);
                    % Update
                    RS_full(curr_scale_pts(keep_these)) = scale_RS{scale_idx}(curr_scale_pts(keep_these));
                    salient_scale_points(curr_scale_pts(keep_these)) = scale_points{scale_idx}(keep_these);
                    max_saliency_Beta_q(scale_R_pts{scale_idx}(keep_these)) = scale_Beta_q{scale_idx}(keep_these);
                    best_pts_mask(scale_R_pts{scale_idx}(keep_these)) = true;
                end

                % Last one.
                % At current scale's ridge points, compare RS against the
                % previous scale and also against RS_full to ensure global
                % maximum.
                curr_scale_pts = scale_R_pts{end};
                keep_these = scale_RS{end}(curr_scale_pts) > scale_RS{end-1}(curr_scale_pts) & ...
                    scale_RS{end}(curr_scale_pts) > RS_full(curr_scale_pts);
                % Update
                RS_full(curr_scale_pts(keep_these)) = scale_RS{end}(curr_scale_pts(keep_these));
                salient_scale_points(curr_scale_pts(keep_these)) = scale_points{end}(keep_these);
                max_saliency_Beta_q(scale_R_pts{end}(keep_these)) = scale_Beta_q{end}(keep_these);
                best_pts_mask(scale_R_pts{end}(keep_these)) = true;

                % Combine scale-level comparison results
                ridge_pts_idxs = uint32(find(best_pts_mask));
                max_saliency_Beta_q = max_saliency_Beta_q(ridge_pts_idxs);
            else
                max_saliency_Beta_q = scale_Beta_q{1};
                ridge_pts_idxs = scale_R_pts{1};
                RS_full(ridge_pts_idxs) = scale_RS{1}(ridge_pts_idxs);
                salient_scale_points(ridge_pts_idxs) = scale_points{1};
            end

            % Mask of ridge points occuring in the interior frames
            pts_in_valid_cols_mask = (ridge_pts_idxs > obj.num_specgram_rows & ...
                ridge_pts_idxs <= (obj.num_specgram_rows * (1 + processed_frames)));

            % Perform non-maximum suppression on the ridge strength.
            % RS_full already contains one extra column on either side to
            % help with frame boundaries.
            keep_pts_mask = pts_in_valid_cols_mask & ...
                dt_SpectrogramRidgeDetector.nonmax_suppress(RS_full, ...
                ridge_pts_idxs, ...
                max_saliency_Beta_q);

%             % Perform scale normalisation
%             RS_full = RS_full ./ salient_scale_points;

            Beta = max_saliency_Beta_q(keep_pts_mask);   % Indexing is straightforward for this
            % For the below two operations, indexing must account for the
            % first "extra" column in RS_full
            ridge_pts_idxs = ridge_pts_idxs(keep_pts_mask) - obj.num_specgram_rows;
            RS = RS_full(ridge_pts_idxs + obj.num_specgram_rows);

%             % Gather power values from buffer and/or specgram_frames as
%             % appropriate.
%             % Find ridge points that were from buffer's frames (if any)
%             num_rpts_from_buf = obj.num_specgram_rows * uint32(max(min(int32(processed_frames), cols_in_buf - int32(obj.max_kernel_delay + obj.process_lag)), 0));
%             temp = (ridge_pts_idxs < (num_rpts_from_buf + 1));
%             Pwr = [ ...
%                 obj.buffer(((obj.max_kernel_delay + obj.process_lag) * obj.num_specgram_rows) + ridge_pts_idxs(temp)); ...
%                 specgram_frames((uint32(max(0, int32(obj.max_kernel_delay + obj.process_lag) - cols_in_buf)) * obj.num_specgram_rows) + ridge_pts_idxs(~temp) - num_rpts_from_buf)];
            Pwr = salient_scale_points(ridge_pts_idxs + obj.num_specgram_rows);

            % Convert linear indices to corresponding rows and cols
            Ridge_freq_idxs = mod(ridge_pts_idxs - 1, obj.num_specgram_rows) + 1;
            Ridge_frame_idxs = ((ridge_pts_idxs - Ridge_freq_idxs) ./ obj.num_specgram_rows) + 1 + obj.frame_no;  % Addding global running frame index here
            
            % Update stuff in preparation for next iteration
            obj.buffer = [obj.buffer(:, (processed_frames+1):end), specgram_frames(:, max(1, int32(processed_frames)-(cols_in_buf-1)):end)];
            obj.frame_no = obj.frame_no + processed_frames;
        end

        %------------------------------------------------------------------
        function Set_min_intensity(obj, new_min_intensity)
            if ~isnumeric(new_min_intensity)
                error('dt_SpectrogramRidgeDetector:Set_min_intensity', ...
                    'min_intensity must be a numeric value')
            end
            obj.min_intensity = new_min_intensity;
        end

        %------------------------------------------------------------------
        function Set_threshold_value(obj, new_threshold_value)
        % new_threshold_value must be in the range 1-num_rows(thld_to_limit_map).
            if ~isnumeric(new_threshold_value) || round(new_threshold_value) ~= new_threshold_value || ...
                    new_threshold_value < 1 || new_threshold_value > size(dt_SpectrogramRidgeDetector.thld_to_limit_map, 1)
                error('dt_SpectrogramRidgeDetector:Set_threshold_value', ...
                    'threshold_value must be an integer numeric value in the range [1-%i]', size(dt_SpectrogramRidgeDetector.thld_to_limit_map, 1));
            end
            obj.threshold_value = new_threshold_value;
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

            % Setting to one frame behind the first valid frame in results
            % so that the addition in ProcessFrames() sets the result to
            % the correct value.
            obj.frame_no = obj.process_lag;

            % Zeroes to start with
            obj.buffer = zeros(obj.num_specgram_rows, obj.max_kernel_delay);
        end

    end     % End private methods
    
    %======================================================================
    methods (Static, Access = 'private')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Private Static Methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %------------------------------------------------------------------
        function [kernel] = GetSmoothingKernelAtScale(sigma)
        % The necessary 2D filtering operation is separable as two
        % succesive 1D convolutions. So, returning only a 1D equivalent
        % here.
            % Construct the Gaussian kernel
            N = ceil(sigma * 4);    % 3*sigma covers >99% area, x4 just to be more inclusive
            kernel = (1 / (sqrt(2 * pi) * sigma)) * exp(-((-N:N).^2) / (2 * (sigma^2)));
        end
        
        %------------------------------------------------------------------
        function [Lx, Ly, Lxx, Lyy, Lxy] = SurfaceDerivatives(in_surface)
        % Determine partial derivitives of surface topography.
        % For input, size(in_surface, 2) must be >= 9.
        % All the return values will have dimensions
        %   size(in_surface, 1) rows   and   size(in_surface, 2)-8 columns.
        % NOTE: The crazy indexing done in both alternatives is necessary
        % for streaming mode operation so that values at frame boundaries
        % do not become 'invalid'.

            % Method 1 (traditional approach):
            % --------------------------------
            % [Lx, Ly] = gradient(in_surface(:, 3:(end-2)));
            % [Lxx, Lxy] = gradient(Lx(:, 2:(end-1)));
            % [Lyx, Lyy] = gradient(Ly(:, 2:(end-1)));
            % Lx = Lx(:, 3:(end-2));
            % Ly = Ly(:, 3:(end-2));
            % Lxx = Lxx(:, 2:(end-1));
            % Lyy = Lyy(:, 2:(end-1));
            % Lxy = Lxy(:, 2:(end-1));
            % ================================

            % Method 2:
            % --------------------------------
            % This approach is, apparently, more accurate than computing
            % with Method 1 at non-vertical & non-horizontal edges. My
            % testing showed that it is quite a bit faster too.
            % The method is described in -
            %     Farid, H., & Simoncelli, E. P. (2004). Differentiation of
            %     discrete multidimensional signals. Image Processing, IEEE
            %     Transactions on, 13(4), 496-508.

            p  = [0.030320  0.249724  0.439911  0.249724  0.030320];
            d1 = [0.104550  0.292315  0.000000 -0.292315 -0.104550];
            d2 = [0.232905  0.002668 -0.471147  0.002668  0.232905];

            temp = conv2(d1, p, in_surface(:, 3:(end-2)), 'same');
            Ly = temp(:, 3:(end-2));
            temp = conv2(p, d2, in_surface(:, 3:(end-2)), 'same');
            Lxx = temp(:, 3:(end-2));
            temp = conv2(d2, p, in_surface(:, 3:(end-2)), 'same');
            Lyy = temp(:, 3:(end-2));

            Lx = conv2(p, d1, in_surface);
            Lxy = conv2(d1, p, Lx, 'valid');

            Lx = Lx(3:(end-2), 7:(end-6));
            Lxy = Lxy(:, 5:(end-4));
            % ===============================
        end

        %------------------------------------------------------------------
        function [zero_crossing_idxs] = FindZeroCrossings(in_surface)
        % Find positions of zero-crossings in 'in_surface'. Adjust position
        % to the lower of the two points involved in the zero-crossing.
        % Only the interior N-2 columns (i.e., columns 2:(N-1)) of
        % in_surface will have valid outputs - the extents on either side
        % are needed for handling points at segment boundaries
        % appropriately. This means that size(in_surface, 2) (i.e., N) must
        % be >= 3.
            surface_abs = abs(in_surface);
            surface_signs = int8(sign(in_surface));
            [N, M] = size(in_surface);

%             zero_crossing_idxs = false(N, M);

            % Check with pixel above
%             zcs = ((in_surface(1:end-1, :) .* in_surface(2:end, :)) <= 0);
            zcs = ((surface_signs(1:end-1, :) ~= surface_signs(2:end, :)));
            lower_vals = (surface_abs(1:end-1, :) < surface_abs(2:end, :));
            zc_idxs_v = [(zcs & lower_vals); false(1, M)] | [false(1, M); (zcs & (~lower_vals))];
%             zero_crossing_idxs([(zcs & lower_vals); false(1, M)] | [false(1, M); (zcs & (~lower_vals))]) = true;

            % Check with pixel to right
%             zcs = ((in_surface(:, 1:end-1) .* in_surface(:, 2:end)) <= 0);
            zcs = ((surface_signs(:, 1:end-1) ~= surface_signs(:, 2:end)));
            lower_vals = (surface_abs(:, 1:end-1) < surface_abs(:, 2:end));
            zc_idxs_h = [(zcs & lower_vals), false(N, 1)] | [false(N, 1), (zcs & (~lower_vals))];
%             zero_crossing_idxs([(zcs & lower_vals), false(N, 1)] | [false(N, 1), (zcs & (~lower_vals))]) = true;

            % Check with pixel to above-right
%             zcs = ((in_surface(1:end-1, 1:end-1) .* in_surface(2:end, 2:end)) <= 0);
            zcs = ((surface_signs(1:end-1, 1:end-1) ~= surface_signs(2:end, 2:end)));
            lower_vals = (surface_abs(1:end-1, 1:end-1) < surface_abs(2:end, 2:end));
            zc_idxs_vh = [(zcs & lower_vals), false(N-1, 1); false(1, M)] | [false(1, M); false(N-1, 1), (zcs & (~lower_vals))];
%             zero_crossing_idxs([(zcs & lower_vals), false(N-1, 1); false(1, M)] | [false(1, M); false(N-1, 1), (zcs & (~lower_vals))]) = true;

            % Check with pixel to above-left
%             zcs = ((in_surface(1:end-1, 2:end) .* in_surface(2:end, 1:end-1)) <= 0);
            zcs = ((surface_signs(1:end-1, 2:end) ~= surface_signs(2:end, 1:end-1)));
            lower_vals = (surface_abs(1:end-1, 2:end) < surface_abs(2:end, 1:end-1));
            zc_idxs_hv = [false(N-1, 1), (zcs & lower_vals); false(1, M)] | [false(1, M); (zcs & (~lower_vals)), false(N-1, 1)];
%             zero_crossing_idxs([false(N-1, 1), (zcs & lower_vals); false(1, M)] | [false(1, M); (zcs & (~lower_vals)), false(N-1, 1)]) = true;

            % Combine all of the above
            zero_crossing_idxs = zc_idxs_v | zc_idxs_h | zc_idxs_vh | zc_idxs_hv;
        end

        %------------------------------------------------------------------
        function [RidgePts_Idxs, Beta_q, M_norm] = ProcessScaleSpace(scale_space, t_pow_gamma, threshold_value, min_intensity)
        % Identify ridge points in the given scale_space. t_pow_gamma is a
        % multiplying factor for the scale, derived from the algorithm
        % constant gamma value. curve_ratio_threshold and min_intensity are
        % tuneable algorithm parameters (thresholds). For their meaning,
        % see the description of the corresponding 'class properties'
        % above.
        % Of the N columns provided in scale_space, 4+1 columns on either
        % ends are considered overlaps (across segment boundaries). The
        % resulting detections, RidgePts_Idxs, are linear indices to the
        % interior N-8 frames starting at the 5th frame. Beta_q contains
        % the angles of the non-dominant curvatures at the ridge points
        % corresponding to RidgePts_Idxs. M_norm gives the ridge strength
        % estimates as a matrix with N-8 columns. The additional column on
        % either sides beyond what is considered valid (interior N-10) are
        % necessary for performing nonmax_suppress() later in the
        % algorithm.

            % Get partial derivatives of power topography
            [Lx, Ly, Lxx, Lyy, Lxy] = ...
                dt_SpectrogramRidgeDetector.SurfaceDerivatives(scale_space);
            % NOTE: 4 columns will be lost on either side here. The outputs
            % above will have 8 columns less than the input to
            % SurfaceDerivatives(). The below value is needed later to
            % handle offsets to linear indexes.
            lost_points_offset_1 = uint32(size(scale_space, 1) * 4);

            % The Hessian of the gradient is given by
            %   | Lxx Lxy |
            %   | Lyx Lyy |
            % The discriminant in the quadratic solutions to the
            % eigenvectors of the Hessian matrix is
            %   = ((Lxx + Lyy) .^ 2) - (4 * (Lxx .* Lyy) - (Lxy .* Lyx))
            %   = (Lxx.^2 + Lyy.^2 + 2*Lxx.*Lyy) - 4*Lxx.*Lyy + 4*Lxy.^2
            % (the last term is so because Lxy = Lyx)
            %   = ((Lxx - Lyy) .^ 2) + (4 * (Lxy .^ 2))
            temp1 = Lxx - Lyy;  % Intermediate value, reused again later
            discriminant_sqrt = sqrt(((temp1 .^ 2) + (4 * (Lxy .^ 2))));

            % Local derivitives in ridge transformed coordinates
            %    -> eigenvalues of the Hessian matrix of the brightness
            %       function at each pixel.
            %
            % For convenience in understanding, we'll ensure that Lpp is
            % always the dominant eigenvector  -->  |Lpp| >= |Lqq|.
            % Lpp is given by
            %     Lpp = ((Lxx + Lyy) ± sqrt(discriminant)) / 2
            % Read the ± as "plus" when (Lxx + Lyy) is non-negative and as
            % "minus" otherwise. This will ensure that the two addition
            % terms are either both positive or both negative, enabling Lpp
            % to have higher magnitude. Equivalently, Lqq is given by
            %     Lqq = ((Lxx + Lyy) ± (-sqrt(discriminant))) / 2
            %
            % Given the way Lpp is computed here, when (Lxx + Lyy) is
            % non-negative, Lpp will never become negative, and when
            % (Lxx + Lyy) is negative Lpp will always be negative. So, the
            % check "(Lxx + Lyy) < 0" is equivalent to "Lpp < 0".
            temp2 = Lxx + Lyy;  % Intermediate value, reused again later
            Lpp_negative_idxs = (temp2 < 0);

            % Direction of the non-dominant eigenvector (Lqq).
            % Using atan() instead of atan2() to restrict the angles to 1st
            % and 4th quadrants only.
            %   Beta_q = atan(Lxy / (Lqq - Lyy))
            %          = atan(Lxy / ((((Lxx + Lyy) ± (-sqrt(discriminant))) / 2) - Lyy))
            %          = atan(2*Lxy / (((Lxx + Lyy) ± (-sqrt(discriminant))) - 2*Lyy))
            %          = atan(2*Lxy / ((Lxx - Lyy) ± (-sqrt(discriminant))))
            scale_Beta_q = ...
                atan(2 * Lxy ./ (temp1 - ((1 - (2 * Lpp_negative_idxs)) .* discriminant_sqrt)));

            % Feature strength measures used in the algorithm are M-norm &
            % A-norm.
            %
            % M-norm (Eq 46): Magnitude of dominant curvature.
            %   = (t ^ gamma) * max(|Lpp|, |Lqq|)
            %   = (t ^ gamma) * |Lpp|
            %
            % A-norm (Eq 51): Difference in magnitudes of the curvatures.
            %   = (Lpp_gamma_norm - Lqq_gamma_norm) ^ 2
            %   = (t ^ (2 * gamma)) * (((Lxx - Lyy) ^ 2) + (4 * (Lxy ^ 2)))
            %   = (t ^ (2 * gamma)) * discriminant

            % M-norm or Gamma-normalised Lpp
            M_norm = (t_pow_gamma/2) * ...
                abs(temp2 + ((1 - (2 * Lpp_negative_idxs)) .* discriminant_sqrt));

            % Compute Lp and determine zero-crossings in it
            %   Lp = (-sin(Beta_q) * Lx) + (cos(Beta_q) * Ly)
            [zc_idxs] = dt_SpectrogramRidgeDetector.FindZeroCrossings( ...
                (sin(scale_Beta_q) .* Lx) - (cos(scale_Beta_q) .* Ly));

            % Definition of ridge: ((Lp == 0) & (Lpp < 0))
            %   Lp == 0  -> handled with zc_idxs
            %   Lpp < 0  -> handled with Lpp_negative_idxs
            % This definition is local to the current scale only.
            RidgePts_Idxs = uint32(find(zc_idxs & Lpp_negative_idxs));

            min_M_norm_norm = dt_SpectrogramRidgeDetector.thld_to_limit_map(threshold_value, 1);
            min_ridge_narrowness = dt_SpectrogramRidgeDetector.thld_to_limit_map(threshold_value, 2);
            % Check against thresholds:
            %   (i)    (|Lpp - Lqq| / |Lpp|) >= min_ridge_narrowness
            %            --> (|Lpp - Lqq| / (M_norm / t^gamma)) >= min_ridge_narrowness
            %            --> (|Lpp - Lqq| / M_norm) >= (min_ridge_narrowness / t^gamma)
            %            --> (discriminant_sqrt / M_norm) >= (min_ridge_narrowness / t^gamma)
            %            --> discriminant_sqrt >= (min_ridge_narrowness * M_norm / t^gamma)
            %   (ii)   (M_norm / scale_space) >= min_M_norm_norm
            %            --> M_norm >= min_M_norm_norm * scale_space
            %   (iii)  spec_pwr >= min_intensity        [if necessary]
            % Retain only those that pass the threshold checks.
            RidgePts_Idxs = RidgePts_Idxs( ...
                (discriminant_sqrt(RidgePts_Idxs) >= M_norm(RidgePts_Idxs) * (min_ridge_narrowness / t_pow_gamma)) & ...
                (M_norm(RidgePts_Idxs) >= min_M_norm_norm * scale_space(RidgePts_Idxs + lost_points_offset_1)));
            if min_intensity > 0
                RidgePts_Idxs = RidgePts_Idxs( ...
                    scale_space(RidgePts_Idxs + lost_points_offset_1) >= min_intensity);
            end
            
            Beta_q = scale_Beta_q(RidgePts_Idxs);
        end

        %------------------------------------------------------------------
        function [maximal_pts_mask] = nonmax_suppress(in_surface, mask_pts_idxs, mask_pts_thetas)
        % Perform non-max suppression on in_surface at points specified by
        % mask_pts_idxs, orthogonal to the corresponding angles specified
        % by mask_pts_thetas. 'mask_pts_idxs' must be linear indexes to
        % points in the 2D array in_surface and 
        % length(mask_pts_idxs) = length(mask_pts_thetas). The return value
        % will be a mask for maximal points in mask_pts_idxs.
        %
        % Code derived and partly copied from nonmax() function written by
        % David R. Martin (dmartin@eecs.berkeley.edu; March 2003).

            maximal_pts_mask = true(size(mask_pts_idxs));   % initialise

            % Do non-max suppression orthogonal to theta.
            thetas = mod(mask_pts_thetas(:) + pi/2, pi);
            
            % The following diagram depicts the 8 cases for non-max
            % suppression. Theta is valued in [0,pi), measured clockwise
            % from the positive x axis. The 'o' marks the pixel of
            % interest, and the eight neighboring pixels are marked with
            % '.'. The orientation is divided into 8 45-degree blocks.
            % Within each block, we interpolate the image value between the
            % two neighboring pixels.
            %
            %        .66.77.                                
            %        5\ | /8                                
            %        5 \|/ 8                                
            %        .--o--.-----> x-axis                     
            %        4 /|\ 1                                
            %        4/ | \1                                
            %        .33.22.                                
            %           |                                   
            %           |
            %           v
            %         y-axis                                  
            %
            % In the code below, d is always the distance from A, so the
            % distance to B is (1-d). A and B are the two neighboring
            % pixels of interest in each of the 8 cases. Note that the
            % clockwise ordering of A and B changes from case to case in
            % order to make it easier to compute d.

            % Determine which pixels belong to which cases.
            % Col 1 = cases 1 & 5, col 2 = cases 2 & 6, etc.
            case_mask = [(thetas >= 0 & thetas < pi/4), ...
                (thetas >= pi/4 & thetas < pi/2), ...
                (thetas >= pi/2 & thetas < pi*3/4), ...
                (thetas >= pi*3/4 & thetas < pi)];

            % Determine distances.
            d = zeros(size(case_mask)); % Cols are for same cases as of 'mask'
            d(case_mask(:, 1), 1) = tan(thetas(case_mask(:, 1)));
            d(case_mask(:, 2), 2) = tan(pi/2 - thetas(case_mask(:, 2)));
            d(case_mask(:, 3), 3) = tan(thetas(case_mask(:, 3)) - pi/2);
            d(case_mask(:, 4), 4) = tan(pi - thetas(case_mask(:, 4)));
            
            case_col_idx = [1 2 3 4 1 2 3 4];

            [h, w] = size(in_surface);

            % Per-case offsets to pixels A and B
            A_off = [h    , 1    , 1     , -h    , -h    , -1    , -1   , h    ];
            B_off = [h + 1, h + 1, -h + 1, -h + 1, -h - 1, -h - 1, h - 1, h - 1];

            mask_helper = [false(h, 1), true(h, w-2), false(h, 1)]; % Don't need to consider end columns
            mask_helper_flip_r = [h, h, h, h, 1, 1, 1, 1];

            for case_idx = 1:8
                % Only keep mask pts that are valid for the current case
                mask_helper(mask_helper_flip_r(case_idx), 2:end-1) = false;
                
                pertinent_pts = find(mask_helper(mask_pts_idxs) & case_mask(:, case_col_idx(case_idx)) & maximal_pts_mask);
                idx = mask_pts_idxs(pertinent_pts, 1);
                idxA = idx + A_off(case_idx);
                idxB = idx + B_off(case_idx);
                imI = (in_surface(idxA) .* (1 - d(pertinent_pts, case_col_idx(case_idx)))) + ...
                    (in_surface(idxB) .* d(pertinent_pts, case_col_idx(case_idx)));
                maximal_pts_mask(pertinent_pts(in_surface(idx) < imI)) = false;
                
                % Reset
                mask_helper(mask_helper_flip_r(case_idx), 2:end-1) = true;
            end

            %maximal_pts_idxs = mask_pts_idxs(maximal_pts_mask);
        end
        
    end     % End private static methods

end
