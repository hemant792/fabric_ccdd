import os
import time
import cv2
import numpy as np
import pandas as pd
from numba import njit
from multiprocessing import Pool, cpu_count
import util1000 as util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

input_dir = r"F:\test_samples"
output_dir = r"F:\output_img"
# calibration_img_path = r"F:\test_samples\IMG_0000.JPG"
calibration_img_path = None
ANGLE_STEP = 0.5
SMOOTH_KIND = "gaussian"
SMOOTH_WINDOW = 9
SMOOTH_SIGMA = SMOOTH_WINDOW / 6.0

model_path=r"F:\new_fabric work\thread_verification.pth"
class ThreadQualityPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        # checkpoint = torch.load(model_path, map_location=self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.feature_scaler = checkpoint['feature_scaler']
        self.sequence_scaler = checkpoint.get('sequence_scaler')
        config = checkpoint['model_config']

        # Recreate model
        if config['use_sequences']:
            from train_contruction import HybridThreadClassifier  # Import your model class
            self.model = HybridThreadClassifier(config['feature_dim'], config['sequence_dim'], True)
        else:
            from train_contruction import SimpleThreadClassifier
            self.model = SimpleThreadClassifier(config['feature_dim'])

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.use_sequences = config['use_sequences']
        self.sequence_dim = config.get('sequence_dim')

    def extract_features(self, thread_positions, array_length, projection_data=None):
        """Extract 12 features from thread data."""
        if len(thread_positions) < 2 or array_length == 0:
            return np.zeros(12)

        thread_count = len(thread_positions)
        thread_density = thread_count / array_length

        spacings = np.diff(thread_positions)
        spacing_mean = np.mean(spacings)
        spacing_std = np.std(spacings)
        spacing_cv = spacing_std / spacing_mean if spacing_mean > 0 else 0
        spacing_regularity = 1 / (1 + spacing_cv)

        large_gaps = np.sum(spacings > 1.5 * spacing_mean) if spacing_mean > 0 else 0
        gap_ratio = large_gaps / len(spacings)

        normalized_positions = [pos / array_length for pos in thread_positions]
        expected_spacing = 1.0 / len(normalized_positions)
        actual_spacings = np.diff(normalized_positions)
        uniformity_score = 1 / (1 + np.std(actual_spacings) / expected_spacing) if expected_spacing > 0 else 0

        if projection_data:
            proj_std = np.std(projection_data)
            proj_mean = np.mean(projection_data)
            signal_to_noise = proj_std / abs(proj_mean) if proj_mean != 0 else 0
            proj_range = np.max(projection_data) - np.min(projection_data)
        else:
            signal_to_noise = 0
            proj_range = 0

        edge_positions = sum(1 for pos in normalized_positions if pos < 0.1 or pos > 0.9)
        edge_ratio = edge_positions / len(normalized_positions)

        return np.array([thread_count, thread_density, spacing_mean, spacing_std,
                         spacing_cv, spacing_regularity, gap_ratio, uniformity_score,
                         signal_to_noise, proj_range, edge_ratio, spacing_regularity])

    def predict(self, thread_positions, array_length, projection_data=None):
        """Predict thread detection quality.

        Returns:
            prediction (int): 0 = under counting, 1 = correct counting
            probability (float): confidence score (0-1)
        """
        # Extract features
        features = self.extract_features(thread_positions, array_length, projection_data)
        features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)

        with torch.no_grad():
            if self.use_sequences:
                # Prepare normalized positions
                normalized_positions = [pos / array_length for pos in thread_positions] if array_length > 0 else []

                # Pad/truncate to required length
                if len(normalized_positions) > self.sequence_dim:
                    sequences = np.array(normalized_positions[:self.sequence_dim])
                else:
                    sequences = np.zeros(self.sequence_dim)
                    sequences[:len(normalized_positions)] = normalized_positions

                sequences_scaled = self.sequence_scaler.transform(sequences.reshape(1, -1))
                sequences_tensor = torch.FloatTensor(sequences_scaled).to(self.device)

                output = self.model(features_tensor, sequences_tensor)
            else:
                output = self.model(features_tensor)

            probability = output.item()
            prediction = 1 if probability > 0.5 else 0

        return prediction, probability

def load_quality_model():
    if os.path.exists(model_path):
        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            torch.serialization.add_safe_globals([StandardScaler, MinMaxScaler])
            return ThreadQualityPredictor(model_path)
        except Exception as e:
            print(f"Failed to load quality model: {e}")
            return None
    return None
quality_predictor = load_quality_model()

PROCESSING_CONFIG = {
    'rot_corr_tol': 0.1, 'shear_tol': 0.1, 'shear_limit': 5
    ,'rot_limit':30 ,'num_cores': 4,
    'shear_correction': 'yes', 'blur_kernel': 5, 'blur_sigma': 5 / 6.0,
    'orientation_correction': False,
    'h_ratio': 0.5772, 'v_ratio': 0.7616, 'cal_size': 'half'
}

OUTPUT_CONFIG = {
    'save_var_plots': False, 'save_binary_img': False, 'save_projection_plots': False,
    'annotated_img': True, 'PRINT_PER_IMAGE': True
}

def _gaussian_kernel_1d(window, sigma):
    if window < 1: window = 1
    if window % 2 == 0: window += 1
    x = np.arange(window) - (window // 2)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (k / (k.sum() + 1e-12)).astype(np.float64)


def _smooth_1d(y, kind="gaussian", window=7, sigma=1.5):
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0: return y.copy()
    if kind == "gaussian":
        k = _gaussian_kernel_1d(window, sigma)
    else:
        window = window if window % 2 == 1 else window + 1
        k = np.ones(window, dtype=np.float64) / float(window)
    pad = len(k) // 2
    y_pad = np.pad(y, pad_width=pad, mode='reflect')
    return np.convolve(y_pad, k, mode='valid')


def _compute_projection_variances_full(h_gray, angles_deg):
    h, w = h_gray.shape
    sum_v = np.zeros_like(angles_deg, dtype=np.float64)
    for idx, ang in enumerate(angles_deg):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(ang), 1.0)
        rot = cv2.warpAffine(h_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        sum_v[idx] = np.var(rot.sum(axis=1).astype(np.float64)) + np.var(rot.sum(axis=0).astype(np.float64))
    return sum_v


def find_optimal_angle(angles, sum_var_raw, sum_var_smooth):
    left_valid, right_valid = find_valid_range_excluding_slopes(sum_var_smooth)
    if right_valid <= left_valid:
        zero_idx = np.argmin(np.abs(angles))
        return angles[zero_idx]
    valid_smooth = sum_var_smooth[left_valid:right_valid]
    valid_indices = np.arange(left_valid, right_valid)
    peaks = []
    for i in range(1, len(valid_smooth) - 1):
        if valid_smooth[i] > valid_smooth[i - 1] and valid_smooth[i] > valid_smooth[i + 1]:
            actual_idx = valid_indices[i]
            peaks.append((actual_idx, sum_var_smooth[actual_idx]))
    if len(peaks) == 0:
        valid_angles = angles[left_valid:right_valid]
        zero_idx = left_valid + np.argmin(np.abs(valid_angles))
        return angles[zero_idx]
    peaks.sort(key=lambda x: x[1], reverse=True)
    if len(peaks) == 1:
        selected_peak_idx = peaks[0][0]
    else:
        peak1_idx, peak1_height = peaks[0]
        peak2_idx, peak2_height = peaks[1]
        height_diff_ratio = abs(peak1_height - peak2_height) / peak1_height
        if height_diff_ratio <= 0.05:  # Within 5%
            angle1, angle2 = angles[peak1_idx], angles[peak2_idx]
            if abs(angle1) <= abs(angle2):
                selected_peak_idx = peak1_idx
            else:
                selected_peak_idx = peak2_idx
        else:
            selected_peak_idx = peak1_idx
    peak_width = get_peak_width(sum_var_smooth, selected_peak_idx)
    left_bound = max(0, selected_peak_idx - peak_width // 2)
    right_bound = min(len(angles), selected_peak_idx + peak_width // 2 + 1)
    raw_segment = sum_var_raw[left_bound:right_bound]
    local_max_idx = np.argmax(raw_segment)
    final_angle_idx = left_bound + local_max_idx
    return angles[final_angle_idx]


def find_valid_range_excluding_slopes(smooth_curve):
    n = len(smooth_curve)
    for i in range(1, n):
        if smooth_curve[i] > smooth_curve[i - 1]:  # Found uphill - end of downhill slope
            left_valid = i - 1  # Include the valley point
            break
    else:
        left_valid = n // 3
    for i in range(n - 2, -1, -1):
        if smooth_curve[i] > smooth_curve[i + 1]:  # Found uphill (going backwards) - end of downhill slope
            right_valid = i + 2  # Include the valley point
            break
    else:
        right_valid = (2 * n) // 3
    min_range = max(3, n // 5)  # At least 20% of the range
    if right_valid - left_valid < min_range:
        center = n // 2
        left_valid = max(0, center - min_range // 2)
        right_valid = min(n, center + min_range // 2)
    return left_valid, right_valid


def get_peak_width(smooth_curve, peak_idx):
    peak_value = smooth_curve[peak_idx]
    n = len(smooth_curve)
    left_min = peak_value
    right_min = peak_value
    for i in range(peak_idx - 1, -1, -1):
        if smooth_curve[i] < left_min:
            left_min = smooth_curve[i]
        if i > 0 and smooth_curve[i] > smooth_curve[i - 1]:
            break
    for i in range(peak_idx + 1, n):
        if smooth_curve[i] < right_min:
            right_min = smooth_curve[i]
        if i < n - 1 and smooth_curve[i] > smooth_curve[i + 1]:
            break
    baseline = max(left_min, right_min)
    half_height = baseline + 0.5 * (peak_value - baseline)
    left_width = 0
    right_width = 0
    for i in range(peak_idx, -1, -1):
        if smooth_curve[i] <= half_height:
            left_width = peak_idx - i
            break
    for i in range(peak_idx, n):
        if smooth_curve[i] <= half_height:
            right_width = i - peak_idx
            break
    total_width = left_width + right_width
    return max(3, min(total_width, len(smooth_curve) // 4))


def improved_deskew(img_bgr, img_name=None, ANGLE_MAX=10.0, ANGLE_MIN=-10.0, modify_angle_limit=True):
    r, c = img_bgr.shape[:2]
    bgr = img_bgr[int(0.25 * r):int(0.75 * r), int(0.25 * c):int(0.75 * c)] if modify_angle_limit else img_bgr
    if PROCESSING_CONFIG['orientation_correction'] and modify_angle_limit:
        ANGLE_MAX, ANGLE_MIN = PROCESSING_CONFIG['rot_limit'], -PROCESSING_CONFIG['rot_limit']
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    if modify_angle_limit:
        clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(50, 50))
        gray = clahe.apply(gray)
    angles = np.arange(ANGLE_MIN, ANGLE_MAX + 1e-9, ANGLE_STEP, dtype=np.float64)
    sum_var= _compute_projection_variances_full(gray, angles)
    sum_sm = _smooth_1d(sum_var, kind=SMOOTH_KIND, window=SMOOTH_WINDOW, sigma=SMOOTH_SIGMA)
    selected_angle = find_optimal_angle(angles, sum_var, sum_sm)
    return float(selected_angle)


def get_thread_tilt(binary):
    weft_height, warp_width = np.array(binary.shape[:2]) // 20
    y_shear_weft = [
        improved_deskew(binary[i:i + weft_height, :], img_name=None, ANGLE_MAX=PROCESSING_CONFIG['shear_limit'],
                        ANGLE_MIN=-PROCESSING_CONFIG['shear_limit'], modify_angle_limit=False)
        for i in range(5, 10 * weft_height, weft_height)]
    x_shear_warp = [
        improved_deskew(binary.T[i:i + warp_width, :], img_name=None, ANGLE_MAX=PROCESSING_CONFIG['shear_limit'],
                        ANGLE_MIN=-PROCESSING_CONFIG['shear_limit'], modify_angle_limit=False)
        for i in range(5, 10 * warp_width, warp_width)]
    return -np.mean(y_shear_weft), -np.mean(x_shear_warp)


def fill_binary_by_density(binary_img, kernel_size=10):
    square_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    square_density = cv2.filter2D((binary_img ==255).astype(np.float32), -1, square_kernel)
    # Horizontal kernel (wider horizontally)
    h_kernel_size = max(3, kernel_size // 2)
    h_kernel = np.ones((h_kernel_size, kernel_size * 2), dtype=np.float32) / (h_kernel_size * kernel_size * 2)
    h_density = cv2.filter2D((binary_img ==255).astype(np.float32), -1, h_kernel)
    # Vertical kernel (taller vertically)
    v_kernel_size = max(3, kernel_size // 2)
    v_kernel = np.ones((kernel_size * 2, v_kernel_size), dtype=np.float32) / (kernel_size * 2 * v_kernel_size)
    v_density = cv2.filter2D((binary_img ==255).astype(np.float32), -1, v_kernel)
    # Combine densities - take maximum to favor horizontal/vertical filling
    combined_density = np.maximum(np.maximum(square_density, h_density), v_density)
    output_img = np.zeros_like(binary_img, dtype=np.uint8)
    fill_threshold = 1 / (1 + np.exp(-kernel_size * (combined_density - 0.5)))
    output_img[fill_threshold > 0.5] = 255
    output_img[binary_img ==255] = 255
    return cv2.medianBlur(output_img, 3*PROCESSING_CONFIG['blur_kernel'])

def count_peaks_and_zero_crossing(projection0, meann=0.0, direction="vertical", img_name=None, slice=None):
    projection = util.unbiased_autocorr(projection0)
    original_projection = projection.copy()  # Keep original for saving
    if OUTPUT_CONFIG['save_projection_plots'] and img_name:
        os.makedirs(output_dir, exist_ok=True)
        corrected_proj_dir = os.path.join(output_dir, "corrected_projections")
        os.makedirs(corrected_proj_dir, exist_ok=True)
        base_name = os.path.splitext(img_name)[0]
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(original_projection, 'b-', alpha=0.7)
        plt.title(f'Original Projection - {direction} - {img_name} - Slice {slice}')
        plt.xlabel('Index')
        plt.ylabel('ACF Value')
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 1, 2)
        plt.plot(projection, 'r-', alpha=0.7)
        plt.title(f'Corrected Projection - {direction} - {img_name} - Slice {slice}')
        plt.xlabel('Index')
        plt.ylabel('ACF Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        corrected_filename = f"{base_name}_{direction}_corrected_slice_{slice}.png"
        corrected_path = os.path.join(corrected_proj_dir, corrected_filename)
        plt.savefig(corrected_path, dpi=120, bbox_inches='tight')
        plt.close()
    meann = np.mean(projection) if meann is None else meann
    count, sub_count, thread_pos = 0, 0, []
    zero_points,thickness = [],[]
    value_pos = 1 if projection[0] >= meann else 0
    last_local_minima = projection[0]
    for idx, value in enumerate(projection):
        if value >= meann and not value_pos:
            if not zero_points or idx - zero_points[-1] >= 1:
                count += 1
                value_pos = 1
                zero_points.append(idx)
                if len(zero_points) >= 2:
                    local_projection=projection[zero_points[-2]:idx]
                    current_minima=np.min(local_projection)
                    if current_minima > 0.15*last_local_minima and idx <= 0.75*len(projection):# or (current_minima > 0.1*last_local_minima and idx > 0.85*len(projection)):
                        count-=2
                        zero_points = zero_points[:-2]
                        thread_pos = thread_pos[:-1]
                    else:
                        last_local_minima = current_minima
        elif value < meann and value_pos:
            if not zero_points or idx - zero_points[-1] >= 1:
                count += 1
                value_pos = 0
                zero_points.append(idx)
                thickness.append(idx-zero_points[-2])
                if len(zero_points) >= 2:
                    start, end = zero_points[-2], zero_points[-1]
                    local = projection[start:end]
                    pos = int(np.argmax(local))
                    thread_pos.append(start + pos - (zero_points[0] // 2))
    if calibration_img_path is None:
        count = count//2
    count = 2*(count//2) + 2 if PROCESSING_CONFIG['cal_size'] =='half' else (count // 2) + 2
    gap = np.diff(zero_points)
    thickness=util.remove_outliers(np.array(thickness))
    return count, np.std(gap), np.mean(gap),np.array(thread_pos), projection,np.mean(thickness)


@njit
def denoise_by_majority_voting_fast(img, threshold=50, std_multiplier=1.0, blur_kernel=3):
    h, w = img.shape
    result = img.copy()
    threshold = threshold * std_multiplier
    kernel_size = 3 * blur_kernel
    for row in range(0, h, kernel_size):
        for col in range(0, w, kernel_size):
            row_end = min(row + kernel_size, h)
            col_end = min(col + kernel_size, w)
            black_pixels = 0
            white_pixels = 0
            for i in range(row, row_end):
                for j in range(col, col_end):
                    if img[i, j] < threshold:
                        black_pixels += 1
                    else:
                        white_pixels += 1
            if black_pixels >= white_pixels:
                for i in range(row, row_end):
                    for j in range(col, col_end):
                        result[i, j] = 0
    return result

@njit
def stretch_contrast_adjustable_fast(grayscale_img, strength=5.0, midpoint=50.0):
    h, w = grayscale_img.shape
    result = np.empty((h, w), dtype=np.uint8)
    mid_normalized = midpoint / 255.0
    for i in range(h):
        for j in range(w):
            normalized = grayscale_img[i, j] / 255.0
            if normalized < mid_normalized:
                stretched = mid_normalized * (normalized / mid_normalized) ** strength
            else:
                stretched = 1.0 - (1.0 - mid_normalized) * ((1.0 - normalized) / (1.0 - mid_normalized)) ** strength
            result[i, j] = np.uint8(stretched * 255.0)
    return result


def denoise_by_majority_voting(img, threshold=50, std_multiplier=1.0, processing_config=None):
    if processing_config is not None:
        blur_kernel = processing_config.get('blur_kernel', 3)
    else:
        blur_kernel = 3  # Default value
    return denoise_by_majority_voting_fast(img, threshold, std_multiplier, blur_kernel)


def stretch_contrast_adjustable(grayscale_img, strength=5.0, midpoint=50):
    return stretch_contrast_adjustable_fast(grayscale_img, strength, float(midpoint))

def analyse_projection(h_proj_img, corrected_color_img, direction='vertical', img_name=None, retry_factor=2,all_attempts=None):
    if all_attempts is None:
        all_attempts = []
    h_projection = h_proj_img[:, -1] - h_proj_img[:, 0]
    h_projection = h_projection.astype(np.float64) - np.mean(h_projection.astype(np.float64))
    indexs, _ = util.periodicity_extraction(util.get_acf(h_projection), 0, factor=retry_factor)
    threads = []
    predictions = []
    pos_counter=0
    for i in range(1, len(indexs) - 1, 2*retry_factor) if len(indexs) > 2 else range(len(indexs)):
        pos_counter += 1
        idx1 = indexs[i]
        idx2 =indexs[len(indexs) - 1]
        slice_img = corrected_color_img[idx1:idx2, :] if direction == 'vertical' else corrected_color_img[:, idx1:idx2]
        l_channel, _, _ = cv2.split(cv2.cvtColor(slice_img, cv2.COLOR_BGR2LAB))
        r, c = slice_img.shape[:2]
        limit = min(int(np.sqrt(r)), int(np.sqrt(c)))
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(limit, limit))
        l_channel = clahe.apply(l_channel)
        std = np.std(l_channel)
        slice_enhanced = denoise_by_majority_voting(stretch_contrast_adjustable(l_channel, midpoint=std), threshold=std)
        slice_enhanced = cv2.GaussianBlur(slice_enhanced,(PROCESSING_CONFIG['blur_kernel'], PROCESSING_CONFIG['blur_kernel']),PROCESSING_CONFIG['blur_sigma'])
        if OUTPUT_CONFIG['save_projection_plots'] and img_name:
            os.makedirs(output_dir, exist_ok=True)
            slice_dir = os.path.join(output_dir, "enhanced_slices")
            os.makedirs(slice_dir, exist_ok=True)
            slice_filename = f"{os.path.splitext(img_name)[0]}_{direction}_slice_{i}_factor_{retry_factor}_enhanced.png"
            slice_path = os.path.join(slice_dir, slice_filename)
            cv2.imwrite(slice_path, slice_enhanced)
        projection = np.sum(slice_enhanced, axis=0 if direction == 'vertical' else 1).astype(np.float64)
        projection -= np.mean(projection)
        count, std, avg_gap, thread_pos, acf, median_width = count_peaks_and_zero_crossing(projection, meann=0.0,direction=direction,img_name=img_name, slice=i)
        prediction, confidence = quality_predictor.predict(thread_positions=thread_pos, array_length=len(projection))
        predictions.append(prediction)
        confidence_score, scores = util.cnf_scores(projection, std, avg_gap, acf)
        regularity_score, peak_prominence, snr_score = scores
        threads.append(
            (count, confidence_score, regularity_score, peak_prominence, snr_score, thread_pos, 0, median_width))
    # Process threads for current attempt
    final_result = util.process_threads(threads, mode='max_confidence')
    # Store current attempt result
    current_attempt = {
        'factor': retry_factor,
        'predictions': predictions.copy(),
        'correct_predictions': sum(predictions),
        'total_predictions': len(predictions),
        'result': final_result,
        'final_confidence': final_result[1] if len(final_result) > 1 else 0.0
    }
    all_attempts.append(current_attempt)
    if sum(predictions) >= 0.1*len(predictions) and len(predictions) > 0:
        # print(f"{img_name} achieved correct counting of {yarn_dir} with factor {retry_factor} ({sum(predictions)}/{len(predictions)} slices correct).")
        return final_result
    next_factor = retry_factor *2
    if next_factor <= len(h_projection):
        return analyse_projection(h_proj_img, corrected_color_img, direction=direction, img_name=img_name,retry_factor=next_factor, all_attempts=all_attempts)
    good_attempts = [attempt for attempt in all_attempts if attempt['correct_predictions'] >= 2]
    if good_attempts:
        best_attempt = max(good_attempts, key=lambda x: x['final_confidence'])
    else:
        best_attempt = max(all_attempts, key=lambda x: x['final_confidence'])
    return best_attempt['result']


def enhanced_count_threads(img, img_name, output_dir_bin, h_ratio=None, v_ratio=None):
    r, c = img.shape[:2]
    img = cv2.GaussianBlur(img, (PROCESSING_CONFIG['blur_kernel'], PROCESSING_CONFIG['blur_kernel']),PROCESSING_CONFIG['blur_sigma'])
    rotation_angle = improved_deskew(img.copy(), img_name=img_name)
    deskewed_col_img = util.rotate_image(img, rotation_angle, same_size=True) if abs(rotation_angle) >= PROCESSING_CONFIG['rot_corr_tol'] else img
    l_channel, _, _ = cv2.split(cv2.cvtColor(deskewed_col_img, cv2.COLOR_BGR2LAB))
    clahe = cv2.createCLAHE(clipLimit=max(int(np.sqrt(c)), int(np.sqrt(r))), tileGridSize=(int(np.sqrt(c)), int(np.sqrt(r))))
    binary_temp = util.remove_if_neighbors_disconnected(util.get_otsu_binary(clahe.apply(l_channel)))
    binary= fill_binary_by_density(binary_temp)
    binary_copy = binary.copy()
    if OUTPUT_CONFIG['save_binary_img'] and output_dir_bin is not None:
        cv2.imwrite(os.path.join(output_dir_bin, f"light_corrected_rot_{img_name}"), binary)
    applied_weft_tilt = applied_warp_tilt = 0.0
    if PROCESSING_CONFIG['shear_correction'] == 'yes':
        weft_tilt, warp_tilt = get_thread_tilt(binary,)
        if ((abs(weft_tilt) > PROCESSING_CONFIG['shear_tol'] or abs(warp_tilt) >= PROCESSING_CONFIG['shear_tol'])
                and abs(weft_tilt) <= PROCESSING_CONFIG['shear_limit']
                and abs(warp_tilt) <= PROCESSING_CONFIG['shear_limit']):
            binary = util.shear_image(binary_copy, x_shear_angle=warp_tilt, y_shear_angle=weft_tilt)
            deskewed_col_img = util.shear_image(deskewed_col_img, x_shear_angle=warp_tilt, y_shear_angle=weft_tilt)
            applied_weft_tilt, applied_warp_tilt = float(weft_tilt), float(warp_tilt)
        if h_ratio is not None and v_ratio is not None:
            binary = util.extract_center_square(binary, h_ratio, v_ratio)
            deskewed_col_img = util.extract_center_square(deskewed_col_img, h_ratio, v_ratio)
        else:
            binary = util.extract_center_square(binary, PROCESSING_CONFIG['h_ratio'], PROCESSING_CONFIG['v_ratio'])
            deskewed_col_img = util.extract_center_square(deskewed_col_img, PROCESSING_CONFIG['h_ratio'],PROCESSING_CONFIG['v_ratio'])
        if OUTPUT_CONFIG['save_binary_img'] and output_dir_bin is not None:
            cv2.imwrite(os.path.join(output_dir_bin, f"sheared_rot_{img_name}"), binary)
    h_proj_img, v_proj_img = np.cumsum(binary, axis=1), np.cumsum(binary, axis=0)
    warp_count, warp_conf, warp_thread_pos = analyse_projection(h_proj_img, deskewed_col_img,direction='vertical', img_name=img_name)
    weft_count, weft_conf, weft_thread_pos = analyse_projection(v_proj_img.T, deskewed_col_img,direction='horizontal', img_name=img_name)
    return int(1.0*warp_count), int(1.0*weft_count), deskewed_col_img, rotation_angle, warp_conf, weft_conf,np.array(warp_thread_pos).tolist(), np.array(weft_thread_pos).tolist(),applied_warp_tilt, applied_weft_tilt


def process_single_image(args):
    img_path, out_dir, output_dir_bin, h_ratio, v_ratio = args
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    t0 = time.time()
    (warp_const, weft_const, processed_img, ang, warp_conf, weft_conf,warp_thread_pos, weft_thread_pos, warp_shear_deg, weft_shear_deg) = enhanced_count_threads(img, img_name, output_dir_bin, h_ratio=h_ratio, v_ratio=v_ratio)
    print(f"\n=== {img_name} ===")
    print(f"\nConstruction Analysis")
    print(f"Detected: Warp={warp_const}, Weft={weft_const}")
    print(f"time elapsed for construction: {(time.time() - t0) / 60} minutes.")
    if OUTPUT_CONFIG['annotated_img']:
        output_path = os.path.join(out_dir, img_name)
        result_img = util.write_text(processed_img, warp_const, weft_const, ang, weft_rows=weft_thread_pos, warp_cols=warp_thread_pos,shear_x=warp_shear_deg, shear_y=weft_shear_deg)
        cv2.imwrite(output_path, result_img)
    return {
        'image_name': img_name,
        'vertical/warp': warp_const,
        'horizontal/weft': weft_const,
        'angle(degrees)': ang,
        'warp_confidence': warp_conf,
        'weft_confidence': weft_conf,
        'shear_x(deg)': warp_shear_deg,
        'shear_y(deg)': weft_shear_deg
    }

def parallel_process_images(input_dir, output_dir, parallel=True, h_ratio=None, v_ratio=None):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    output_dir_bin = util.create_unique_directory(output_dir, "binary") if OUTPUT_CONFIG['save_binary_img'] else None
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    all_images = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if f.lower().endswith(image_extensions)]
    args_list = [(img_path, output_dir, output_dir_bin, h_ratio, v_ratio) for img_path in all_images]
    if parallel and len(all_images) > 0:
        with Pool(processes=min(cpu_count(), PROCESSING_CONFIG['num_cores'])) as pool:
            results = pool.map(process_single_image, args_list)
    else:
        results = [process_single_image(args) for args in args_list]
    results = [res for res in results if res is not None]
    if len(results) > 0:
        df = pd.DataFrame(results, columns=['image_name', 'vertical/warp', 'horizontal/weft','angle(degrees)','shear_x(deg)', 'shear_y(deg)'])
        excel_file = os.path.join(output_dir, "thread_count.xlsx")
        df.to_excel(excel_file, index=False)
    print(f"\nProcessing complete! Total time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    if calibration_img_path is None:
        h_ratio, v_ratio = 0.74,0.94
    else:
        h_ratio, v_ratio = util.get_calibration_ratio(calibration_img_path)
    print(f"Calibration ratios - h_ratio: {h_ratio:.4f}, v_ratio: {v_ratio:.4f}")
    parallel_process_images(input_dir, output_dir, parallel=True, h_ratio=h_ratio, v_ratio=v_ratio)