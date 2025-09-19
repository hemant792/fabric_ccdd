import os
import time
import cv2
import numpy as np
import pandas as pd
from numba import njit
from multiprocessing import Pool, cpu_count
import util1000 as util
from scipy.signal import savgol_filter, find_peaks
import matplotlib
from collections import Counter
import math
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

input_dir = r"E:\count_test\sample_true_data.xlsx"
output_dir = r"E:\output_img"
# calibration_img_path = r"E:\C_C\C_C_4_oxford\New folder\IMG_0001.JPG"
calibration_img_path = r"E:\count_test\IMG_0000.JPG"
# Processing parameters
ANGLE_STEP = 0.5
SMOOTH_KIND = "gaussian"
SMOOTH_WINDOW = 9
SMOOTH_SIGMA = SMOOTH_WINDOW / 6.0
gsm=217
# print(f"GSM is : {gsm}")
# reader = WeightReader(port='COM3')
# if reader.connect():
#     weight = reader.read_weight()
weave_type="plain"


from cr_predictor_crossvit5 import UniversalFabricModel
cr_inference_model = UniversalFabricModel(checkpoint_path=r"D:\running_code\checkpoints\efficientvit_regression_best.pth")

model_path=r"D:\construction\thread_detection_model_20250917_171804.pth"
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

FABRIC_CONSTRUCTION_FACTORS = {
    'plain_weave': {
        'name': 'Plain Weave',
        'crimp_factors': {'warp': 1.06, 'weft': 1.06},
        'coverage_factor': 0.98,
        'visibility_corrections': {'warp': 1.00, 'weft': 1.00},
        'thickness_factor': 1.00,
        'yarn_tension': {'warp': 'high', 'weft': 'high'},
        'detection_difficulty': 'easy'
    },

    'twill_2_1': {
        'name': 'Twill 2/1',
        'crimp_factors': {'warp': 1.12, 'weft': 1.08},
        'coverage_factor': 0.92,
        'visibility_corrections': {'warp': 0.95, 'weft': 0.98},
        'thickness_factor': 1.08,
        'yarn_tension': {'warp': 'medium-high', 'weft': 'medium'},
        'detection_difficulty': 'medium'
    },

    'twill_3_1': {
        'name': 'Twill 3/1',
        'crimp_factors': {'warp': 1.15, 'weft': 1.07},
        'coverage_factor': 0.88,
        'visibility_corrections': {'warp': 0.90, 'weft': 0.95},
        'thickness_factor': 1.12,
        'yarn_tension': {'warp': 'medium', 'weft': 'low'},
        'detection_difficulty': 'medium-hard'
    },

    'twill_2_2': {
        'name': 'Twill 2/2 (Balanced)',
        'crimp_factors': {'warp': 1.10, 'weft': 1.10},
        'coverage_factor': 0.91,
        'visibility_corrections': {'warp': 0.93, 'weft': 0.93},
        'thickness_factor': 1.09,
        'yarn_tension': {'warp': 'medium', 'weft': 'medium'},
        'detection_difficulty': 'medium'
    },

    'satin_5': {
        'name': 'Satin 5-Harness',
        'crimp_factors': {'warp': 1.03, 'weft': 1.12},
        'coverage_factor': 0.84,
        'visibility_corrections': {'warp': 0.85, 'weft': 0.90},
        'thickness_factor': 1.20,
        'yarn_tension': {'warp': 'low', 'weft': 'medium-high'},
        'detection_difficulty': 'hard'
    },

    'satin_8': {
        'name': 'Satin 8-Harness',
        'crimp_factors': {'warp': 1.02, 'weft': 1.15},
        'coverage_factor': 0.80,
        'visibility_corrections': {'warp': 0.80, 'weft': 0.88},
        'thickness_factor': 1.25,
        'yarn_tension': {'warp': 'very low', 'weft': 'high'},
        'detection_difficulty': 'very hard'
    },

    'oxford_weave': {
        'name': 'Oxford Weave',
        'crimp_factors': {'warp': 1.08, 'weft': 1.06},
        'coverage_factor': 0.95,
        'visibility_corrections': {'warp': 1.05, 'weft': 1.00},  # May over-count grouped warps
        'thickness_factor': 1.10,
        'yarn_tension': {'warp': 'medium', 'weft': 'medium'},
        'detection_difficulty': 'medium'
    },

    'basket_2x2': {
        'name': 'Basket 2x2',
        'crimp_factors': {'warp': 1.07, 'weft': 1.07},
        'coverage_factor': 0.88,
        'visibility_corrections': {'warp': 0.95, 'weft': 0.95},
        'thickness_factor': 1.08,
        'yarn_tension': {'warp': 'medium', 'weft': 'medium'},
        'detection_difficulty': 'medium'
    },

    'hopsack_2x2': {
        'name': 'Hopsack 2x2',
        'crimp_factors': {'warp': 1.06, 'weft': 1.06},
        'coverage_factor': 0.80,
        'visibility_corrections': {'warp': 0.90, 'weft': 0.90},
        'thickness_factor': 0.98,
        'yarn_tension': {'warp': 'low', 'weft': 'low'},
        'detection_difficulty': 'medium-hard'
    },

    'herringbone': {
        'name': 'Herringbone',
        'crimp_factors': {'warp': 1.10, 'weft': 1.08},
        'coverage_factor': 0.91,
        'visibility_corrections': {'warp': 0.92, 'weft': 0.96},
        'thickness_factor': 1.09,
        'yarn_tension': {'warp': 'medium', 'weft': 'medium'},
        'detection_difficulty': 'medium-hard'
    },

    'dobby_simple': {
        'name': 'Simple Dobby',
        'crimp_factors': {'warp': 1.08, 'weft': 1.08},
        'coverage_factor': 0.90,
        'visibility_corrections': {'warp': 0.93, 'weft': 0.93},
        'thickness_factor': 1.06,
        'yarn_tension': {'warp': 'medium', 'weft': 'medium'},
        'detection_difficulty': 'medium'
    },

    'dobby_complex': {
        'name': 'Complex Dobby',
        'crimp_factors': {'warp': 1.12, 'weft': 1.10},
        'coverage_factor': 0.85,
        'visibility_corrections': {'warp': 0.88, 'weft': 0.90},
        'thickness_factor': 1.12,
        'yarn_tension': {'warp': 'variable', 'weft': 'variable'},
        'detection_difficulty': 'hard'
    },

    'leno_gauze': {
        'name': 'Leno/Gauze',
        'crimp_factors': {'warp': 1.08, 'weft': 1.04},
        'coverage_factor': 0.68,
        'visibility_corrections': {'warp': 0.85, 'weft': 0.95},
        'thickness_factor': 0.90,
        'yarn_tension': {'warp': 'high', 'weft': 'low'},
        'detection_difficulty': 'very hard'
    }
}

# ============================== UTILITY FUNCTIONS ==============================
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

# ============================== MAIN DESKEW ALGORITHM ==============================
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


# ============================== THREAD ANALYSIS FUNCTIONS ==============================

# def get_thread_tilt(binary):
#     h_projection=util.get_acf(np.sum(binary,axis=0))
#     h_projection=h_projection.astype(np.float64)
#     h_projection -= np.mean(h_projection)
#     index_v, _ = util.periodicity_extraction(h_projection, 0,factor=4)
#     v_projection = util.get_acf(np.sum(binary, axis=1))
#     v_projection =v_projection.astype(np.float64)
#     v_projection -= np.mean(v_projection)
#     index_h, _ = util.periodicity_extraction(v_projection, 0,factor=4)
#     y_shear_weft = [improved_deskew(binary[index_v[i]:index_v[i+1], :], img_name=None,ANGLE_MAX=PROCESSING_CONFIG['shear_limit'],ANGLE_MIN=-PROCESSING_CONFIG['shear_limit'], modify_angle_limit=False)
#                         for i in range(2,len(index_v)-3,2)]
#     x_shear_warp = [improved_deskew(binary[:,index_h[i]:index_h[i+1]], img_name=None,ANGLE_MAX=PROCESSING_CONFIG['shear_limit'],ANGLE_MIN=-PROCESSING_CONFIG['shear_limit'], modify_angle_limit=False)
#                         for i in range(2,len(index_h)-3,2)]
#     return -np.mean(y_shear_weft), -np.mean(x_shear_warp)

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


def plot_projection(projection,direction=None,img_name=None,slice_no=None):
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "projection_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(projection)
    plt.title(f'ACF Projection - {direction} - {img_name} - Slice {slice_no - 1}')
    plt.xlabel('Index')
    plt.ylabel('ACF Value')
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(plot_dir, f"{os.path.splitext(img_name)[0]}_{direction}_acf_slice_{slice_no - 1}.png")
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()

def count_peaks_and_zero_crossing(projection0, meann=0.0, direction="vertical", img_name=None, slice=None):
    projection = util.unbiased_autocorr(projection0)
    original_projection = projection.copy()  # Keep original for saving
    if OUTPUT_CONFIG['save_projection_plots'] and img_name:
        os.makedirs(output_dir, exist_ok=True)
        corrected_proj_dir = os.path.join(output_dir, "corrected_projections")
        os.makedirs(corrected_proj_dir, exist_ok=True)

        base_name = os.path.splitext(img_name)[0]

        # Plot original vs corrected projection
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
    count = 2*(count // 2 + sub_count) + 4 if PROCESSING_CONFIG['cal_size'] =='half' else (count // 2 + sub_count) + 2
    gap = np.diff(zero_points)
    thickness=util.remove_outliers(np.array(thickness))
    # print(np.mean(thickness))
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

def analyse_projection(h_proj_img, corrected_color_img, direction='vertical', img_name=None, retry_factor=2,
                       all_attempts=None):
    """
    Analyze projection with recursive retry mechanism using increasing factors.

    Args:
        retry_factor: Current factor value (starts at 4, multiplies by 2 each retry)
        all_attempts: List to store all attempt results for comparison
    """
    if all_attempts is None:
        all_attempts = []
    h_projection = h_proj_img[:, -1] - h_proj_img[:, 0]
    h_projection = h_projection.astype(np.float64) - np.mean(h_projection.astype(np.float64))
    # Use current retry_factor
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
    yarn_dir = "warp" if direction == 'vertical' else "weft"
    # Check if current attempt has good predictions (at least 2 correct)
    if sum(predictions) >= 0.1*len(predictions) and len(predictions) > 0:
        # print(f"{img_name} achieved correct counting of {yarn_dir} with factor {retry_factor} ({sum(predictions)}/{len(predictions)} slices correct).")
        return final_result
    next_factor = retry_factor *2
    # Check if we should continue recursion
    if next_factor <= len(h_projection):
        # print(f"{img_name} has wrong counting of {yarn_dir} with factor {retry_factor} ({sum(predictions)}/{len(predictions)} correct). Retrying with factor {next_factor}.")
        return analyse_projection(h_proj_img, corrected_color_img, direction=direction, img_name=img_name,retry_factor=next_factor, all_attempts=all_attempts)
    # Recursion limit reached, select best attempt
    # print(f"{img_name} reached maximum factor {retry_factor} for {yarn_dir}. Selecting best attempt from {len(all_attempts)} tries.")

    # First try to find any attempt with good predictions (>= 2 correct)
    good_attempts = [attempt for attempt in all_attempts if attempt['correct_predictions'] >= 2]
    if good_attempts:
        # Select the one with highest confidence among good attempts
        best_attempt = max(good_attempts, key=lambda x: x['final_confidence'])
        # print(f"Selected attempt with factor {best_attempt['factor']} (confidence: {best_attempt['final_confidence']:.3f}, predictions: {best_attempt['correct_predictions']}/{best_attempt['total_predictions']})")
    else:
        best_attempt = max(all_attempts, key=lambda x: x['final_confidence'])
        # print(f"No good predictions found. Selected attempt with highest confidence: factor {best_attempt['factor']} (confidence: {best_attempt['final_confidence']:.3f})")
    # print(f"{img_name} ended with suboptimal counting of {yarn_dir}.")
    return best_attempt['result']

def analyse_projection00(h_proj_img, corrected_color_img, direction='vertical', img_name=None):
    h_projection = h_proj_img[:, -1] - h_proj_img[:, 0]
    h_projection = h_projection.astype(np.float64) - np.mean(h_projection.astype(np.float64))
    indexs,_ = util.periodicity_extraction(util.get_acf(h_projection), 0)
    threads = []
    for i in range(1,len(indexs)-1) if len(indexs) >2 else range(len(indexs)):
        idx1,idx2 = indexs[i],indexs[i+1]
        slice_img = corrected_color_img[idx1:idx2, :] if direction == 'vertical' else corrected_color_img[:, idx1:idx2]
        l_channel, _, _ = cv2.split(cv2.cvtColor(slice_img, cv2.COLOR_BGR2LAB))
        r, c = slice_img.shape[:2]
        limit = min(int(np.sqrt(r)), int(np.sqrt(c)))
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(limit, limit))
        l_channel = clahe.apply(l_channel)
        std=np.std(l_channel)
        slice_enhanced =denoise_by_majority_voting(stretch_contrast_adjustable(l_channel,midpoint= std),threshold=std)
        slice_enhanced =cv2.GaussianBlur(slice_enhanced, (PROCESSING_CONFIG['blur_kernel'],PROCESSING_CONFIG['blur_kernel']), PROCESSING_CONFIG['blur_sigma'])
        if OUTPUT_CONFIG['save_projection_plots'] and img_name:
            os.makedirs(output_dir, exist_ok=True)
            slice_dir = os.path.join(output_dir, "enhanced_slices")
            os.makedirs(slice_dir, exist_ok=True)
            slice_filename = f"{os.path.splitext(img_name)[0]}_{direction}_slice_{i}_enhanced.png"
            slice_path = os.path.join(slice_dir, slice_filename)
            cv2.imwrite(slice_path, slice_enhanced)
        projection = np.sum(slice_enhanced, axis=0 if direction == 'vertical' else 1).astype(np.float64)
        projection -= np.mean(projection)
        count, std, avg_gap, thread_pos,acf,median_width = count_peaks_and_zero_crossing(projection, meann=0.0,direction=direction,img_name=img_name,slice=i)
        confidence, scores = util.cnf_scores(projection, std, avg_gap, acf)
        regularity_score, peak_prominence, snr_score = scores
        threads.append((count, confidence, regularity_score, peak_prominence, snr_score, thread_pos, 0,median_width))
    return util.process_threads(threads, mode='max_confidence')


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
    warp_count, warp_conf, warp_thread_pos,warp_width = analyse_projection(h_proj_img, deskewed_col_img,direction='vertical', img_name=img_name)
    weft_count, weft_conf, weft_thread_pos, weft_width = analyse_projection(v_proj_img.T, deskewed_col_img,direction='horizontal', img_name=img_name)
    # print(f"thickness ratio: {warp_width/weft_width}, warp_count {warp_width}, weft_count {weft_width}")
    return int(1.0*warp_count), int(1.0*weft_count), deskewed_col_img, rotation_angle, warp_conf, weft_conf,np.array(warp_thread_pos).tolist(), np.array(weft_thread_pos).tolist(),applied_warp_tilt, applied_weft_tilt


def central_crop(image_path, crop_size=2048, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    if isinstance(crop_size, int):
        ch, cw = crop_size, crop_size
    else:
        ch, cw = crop_size
    ch = min(ch, h)
    cw = min(cw, w)
    start_y = (h - ch) // 2
    start_x = (w - cw) // 2
    cropped = img[start_y:start_y + ch, start_x:start_x + cw]
    return cropped


def extract_multiple_crops(image, crop_size=2048, overlap_ratio=0.1, min_crops=10, max_crops=30):
    """
    Extract multiple crops from an image for robust ply prediction.

    Args:
        image: Input image (numpy array)
        crop_size: Size of each crop (default 2048x2048)
        overlap_ratio: Overlap between adjacent crops (0.5 = 50% overlap)
        min_crops: Minimum number of crops to extract
        max_crops: Maximum number of crops to extract

    Returns:
        List of cropped images
    """
    h, w = image.shape[:2]

    # If image is smaller than crop_size, return resized version
    if h < crop_size or w < crop_size:
        resized = cv2.resize(image, (crop_size, crop_size))
        return [resized]

    crops = []
    step_size = int(crop_size * (1 - overlap_ratio))

    # Calculate grid dimensions
    rows = max(1, (h - crop_size) // step_size + 1)
    cols = max(1, (w - crop_size) // step_size + 1)

    # If we can't get enough crops with current step size, reduce step size
    total_possible = rows * cols
    if total_possible < min_crops:
        step_size = max(crop_size // 4, min(
            (h - crop_size) // (int(math.sqrt(min_crops)) - 1) if h > crop_size else crop_size,
            (w - crop_size) // (int(math.sqrt(min_crops)) - 1) if w > crop_size else crop_size
        ))
        rows = max(1, (h - crop_size) // step_size + 1)
        cols = max(1, (w - crop_size) // step_size + 1)

    # Extract crops in grid pattern
    positions = []
    for i in range(rows):
        for j in range(cols):
            y = min(i * step_size, h - crop_size)
            x = min(j * step_size, w - crop_size)
            positions.append((y, x))

    # If we have too many positions, select a subset
    if len(positions) > max_crops:
        # Select positions more evenly distributed
        indices = np.linspace(0, len(positions) - 1, max_crops, dtype=int)
        positions = [positions[i] for i in indices]

    # Extract crops
    for y, x in positions:
        crop = image[y:y + crop_size, x:x + crop_size]
        if crop.shape[0] == crop_size and crop.shape[1] == crop_size:
            crops.append(crop)

    # If still not enough crops, add center crop and corner crops
    if len(crops) < min_crops:
        # Center crop
        center_y = (h - crop_size) // 2
        center_x = (w - crop_size) // 2
        if center_y >= 0 and center_x >= 0:
            center_crop = image[center_y:center_y + crop_size, center_x:center_x + crop_size]
            crops.append(center_crop)

        # Corner crops if space allows
        corners = [
            (0, 0),  # Top-left
            (0, w - crop_size),  # Top-right
            (h - crop_size, 0),  # Bottom-left
            (h - crop_size, w - crop_size)  # Bottom-right
        ]

        for y, x in corners:
            if y >= 0 and x >= 0 and len(crops) < min_crops:
                corner_crop = image[y:y + crop_size, x:x + crop_size]
                if corner_crop.shape[0] == crop_size and corner_crop.shape[1] == crop_size:
                    # Check if this crop is significantly different from existing crops
                    is_unique = True
                    for existing_crop in crops:
                        if np.array_equal(corner_crop, existing_crop):
                            is_unique = False
                            break
                    if is_unique:
                        crops.append(corner_crop)

    return crops[:max_crops] if len(crops) > max_crops else crops


def predict_ply_with_multiple_crops(image, ply_inference_model, crop_size=2048,
                                    confidence_threshold=0.7, min_agreement=0.8):
    """
    Predict ply using multiple crops for robust prediction.

    Args:
        image: Input image (numpy array)
        ply_inference_model: Trained ply prediction model
        crop_size: Size of crops to extract
        confidence_threshold: Minimum confidence for a prediction to be considered reliable
        min_agreement: Minimum fraction of crops that must agree for final decision

    Returns:
        dict with final prediction, confidence, and detailed results
    """
    # Extract multiple crops
    crops = extract_multiple_crops(image, crop_size=crop_size)

    if not crops:
        return {
            'predicted_class': 1,  # Default to 1-ply
            'confidence': 0.0,
            'num_crops': 0,
            'crop_predictions': [],
            'agreement_ratio': 0.0,
            'reliable_predictions': 0,
            'prediction_distribution': {}  # Add this missing key
        }

    # Get predictions for each crop
    crop_predictions = []
    reliable_predictions = []

    for i, crop in enumerate(crops):
        try:
            # Pass image array directly to model, not file path
            pred_result = ply_inference_model.predict(crop)
            crop_predictions.append({
                'crop_index': i,
                'predicted_class': pred_result['predicted_class'],
                'confidence': pred_result['confidence'],
                'probabilities': pred_result['probabilities']
            })

            # Consider prediction reliable if confidence is above threshold
            if pred_result['confidence'] >= confidence_threshold:
                reliable_predictions.append(pred_result['predicted_class'])

        except Exception as e:
            print(f"Error predicting crop {i}: {e}")
            continue

    if not crop_predictions:
        return {
            'predicted_class': 1,  # Default to 1-ply
            'confidence': 0.0,
            'num_crops': len(crops),
            'crop_predictions': [],
            'agreement_ratio': 0.0,
            'reliable_predictions': 0,
            'prediction_distribution': {}  # Add this missing key
        }

    # Analyze predictions
    all_predictions = [pred['predicted_class'] for pred in crop_predictions]
    all_confidences = [pred['confidence'] for pred in crop_predictions]

    # Count predictions
    prediction_counts = Counter(all_predictions)
    most_common_prediction, most_common_count = prediction_counts.most_common(1)[0]

    # Calculate agreement ratio
    agreement_ratio = most_common_count / len(all_predictions)

    # Decision logic
    if reliable_predictions:
        # Use reliable predictions if available
        reliable_counts = Counter(reliable_predictions)
        reliable_prediction, reliable_count = reliable_counts.most_common(1)[0]
        reliable_agreement = reliable_count / len(reliable_predictions)

        if reliable_agreement >= min_agreement:
            final_prediction = reliable_prediction
            # Calculate average confidence for the chosen prediction
            final_confidence = np.mean([
                pred['confidence'] for pred in crop_predictions
                if pred['predicted_class'] == final_prediction and pred['confidence'] >= confidence_threshold
            ])
        else:
            # Fallback to majority vote among all predictions
            final_prediction = most_common_prediction
            final_confidence = np.mean([
                pred['confidence'] for pred in crop_predictions
                if pred['predicted_class'] == final_prediction
            ])
    else:
        # Use majority vote among all predictions
        if agreement_ratio >= min_agreement:
            final_prediction = most_common_prediction
            final_confidence = np.mean([
                pred['confidence'] for pred in crop_predictions
                if pred['predicted_class'] == final_prediction
            ])
        else:
            # Low agreement - use weighted average or default to more conservative choice
            if len(prediction_counts) == 2:
                # If close split, use confidence-weighted decision
                class_0_preds = [pred['confidence'] for pred in crop_predictions if pred['predicted_class'] == 0]
                class_1_preds = [pred['confidence'] for pred in crop_predictions if pred['predicted_class'] == 1]

                class_0_conf = np.mean(class_0_preds) if class_0_preds else 0.0
                class_1_conf = np.mean(class_1_preds) if class_1_preds else 0.0

                final_prediction = 0 if class_0_conf > class_1_conf else 1
                final_confidence = max(class_0_conf, class_1_conf)
            else:
                final_prediction = most_common_prediction
                final_confidence = np.mean(all_confidences)

    return {
        'predicted_class': final_prediction,
        'confidence': final_confidence,
        'num_crops': len(crops),
        'crop_predictions': crop_predictions,
        'agreement_ratio': agreement_ratio,
        'reliable_predictions': len(reliable_predictions),
        'prediction_distribution': dict(prediction_counts)
    }


def predict_binary_class_with_multiple_crops(image, binary_inference_model, crop_size=2048,
                                             confidence_threshold=0.7, min_agreement=0.8):
    """
    Predict binary classification (ratio=1 vs ratio≠1) using multiple crops for robust prediction.

    Args:
        image: Input image (numpy array)
        binary_inference_model: Trained binary classification model
        crop_size: Size of crops to extract
        confidence_threshold: Minimum confidence for a prediction to be considered reliable
        min_agreement: Minimum fraction of crops that must agree for final decision

    Returns:
        dict with final prediction, confidence, and detailed results
    """
    # Extract multiple crops
    crops = extract_multiple_crops(image, crop_size=crop_size)

    if not crops:
        return {
            'class': 0,  # Default to ratio≠1
            'probability': 0.5,
            'logit': 0.0,
            'confidence': 0.0,
            'num_crops': 0,
            'crop_predictions': [],
            'agreement_ratio': 0.0,
            'reliable_predictions': 0,
            'prediction_distribution': {}
        }

    # Get predictions for each crop
    crop_predictions = []
    reliable_predictions = []

    for i, crop in enumerate(crops):
        try:
            # Get prediction from binary classification model
            pred_result = binary_inference_model.predict_single_image(crop)

            crop_predictions.append({
                'crop_index': i,
                'class': pred_result['class'],
                'probability': pred_result['probability'],
                'logit': pred_result['logit'],
                'confidence': abs(pred_result['probability'] - 0.5) * 2  # Convert to 0-1 confidence
            })

            # Consider prediction reliable if confidence is above threshold
            confidence = abs(pred_result['probability'] - 0.5) * 2
            if confidence >= confidence_threshold:
                reliable_predictions.append(pred_result['class'])

        except Exception as e:
            print(f"Error predicting crop {i}: {e}")
            continue

    if not crop_predictions:
        return {
            'class': 0,  # Default to ratio≠1
            'probability': 0.5,
            'logit': 0.0,
            'confidence': 0.0,
            'num_crops': len(crops),
            'crop_predictions': [],
            'agreement_ratio': 0.0,
            'reliable_predictions': 0,
            'prediction_distribution': {}
        }

    # Analyze predictions
    all_predictions = [pred['class'] for pred in crop_predictions]
    all_confidences = [pred['confidence'] for pred in crop_predictions]
    all_probabilities = [pred['probability'] for pred in crop_predictions]
    all_logits = [pred['logit'] for pred in crop_predictions]

    # Count predictions
    prediction_counts = Counter(all_predictions)
    most_common_prediction, most_common_count = prediction_counts.most_common(1)[0]

    # Calculate agreement ratio
    agreement_ratio = most_common_count / len(all_predictions)

    # Decision logic
    if reliable_predictions:
        # Use reliable predictions if available
        reliable_counts = Counter(reliable_predictions)
        reliable_prediction, reliable_count = reliable_counts.most_common(1)[0]
        reliable_agreement = reliable_count / len(reliable_predictions)

        if reliable_agreement >= min_agreement:
            final_prediction = reliable_prediction
            # Calculate average metrics for the chosen prediction from reliable crops
            reliable_crop_data = [
                pred for pred in crop_predictions
                if pred['class'] == final_prediction and pred['confidence'] >= confidence_threshold
            ]
            final_confidence = np.mean([pred['confidence'] for pred in reliable_crop_data])
            final_probability = np.mean([pred['probability'] for pred in reliable_crop_data])
            final_logit = np.mean([pred['logit'] for pred in reliable_crop_data])
        else:
            # Fallback to majority vote among all predictions
            final_prediction = most_common_prediction
            matching_crops = [pred for pred in crop_predictions if pred['class'] == final_prediction]
            final_confidence = np.mean([pred['confidence'] for pred in matching_crops])
            final_probability = np.mean([pred['probability'] for pred in matching_crops])
            final_logit = np.mean([pred['logit'] for pred in matching_crops])
    else:
        # Use majority vote among all predictions
        if agreement_ratio >= min_agreement:
            final_prediction = most_common_prediction
            matching_crops = [pred for pred in crop_predictions if pred['class'] == final_prediction]
            final_confidence = np.mean([pred['confidence'] for pred in matching_crops])
            final_probability = np.mean([pred['probability'] for pred in matching_crops])
            final_logit = np.mean([pred['logit'] for pred in matching_crops])
        else:
            # Low agreement - use confidence-weighted decision
            if len(prediction_counts) == 2:
                # Calculate weighted averages for each class
                class_0_crops = [pred for pred in crop_predictions if pred['class'] == 0]
                class_1_crops = [pred for pred in crop_predictions if pred['class'] == 1]

                class_0_conf = np.mean([pred['confidence'] for pred in class_0_crops]) if class_0_crops else 0.0
                class_1_conf = np.mean([pred['confidence'] for pred in class_1_crops]) if class_1_crops else 0.0

                if class_0_conf > class_1_conf:
                    final_prediction = 0
                    final_confidence = class_0_conf
                    final_probability = np.mean([pred['probability'] for pred in class_0_crops])
                    final_logit = np.mean([pred['logit'] for pred in class_0_crops])
                else:
                    final_prediction = 1
                    final_confidence = class_1_conf
                    final_probability = np.mean([pred['probability'] for pred in class_1_crops])
                    final_logit = np.mean([pred['logit'] for pred in class_1_crops])
            else:
                final_prediction = most_common_prediction
                final_confidence = np.mean(all_confidences)
                final_probability = np.mean(all_probabilities)
                final_logit = np.mean(all_logits)

    return {
        'class': final_prediction,
        'probability': final_probability,
        'logit': final_logit,
        'confidence': final_confidence,
        'num_crops': len(crops),
        'crop_predictions': crop_predictions,
        'agreement_ratio': agreement_ratio,
        'reliable_predictions': len(reliable_predictions),
        'prediction_distribution': dict(prediction_counts)
    }


def predict_ratio_with_multiple_crops(image, ratio_inference_model, crop_size=2048,
                                      confidence_threshold=0.05, min_agreement=0.6):
    """
    Predict warp-to-weft ratio using multiple crops for robust prediction.

    Args:
        image: Input image (numpy array)
        ratio_inference_model: Trained ratio prediction model
        crop_size: Size of crops to extract
        confidence_threshold: Maximum standard deviation for predictions to be considered reliable
        min_agreement: Minimum fraction of crops that must be within acceptable range

    Returns:
        dict with final prediction, confidence, and detailed results
    """
    # Extract multiple crops
    crops = extract_multiple_crops(image, crop_size=crop_size)

    if not crops:
        return {
            'predicted_ratio': 1.0,  # Default to balanced ratio
            'confidence': 0.0,
            'std_deviation': 0.0,
            'num_crops': 0,
            'crop_predictions': [],
            'agreement_ratio': 0.0,
            'reliable_predictions': 0,
            'prediction_stats': {}
        }

    # Get predictions for each crop
    crop_predictions = []
    all_ratios = []

    for i, crop in enumerate(crops):
        try:
            # Get ratio prediction
            predicted_ratio = ratio_inference_model.predict(crop)

            crop_predictions.append({
                'crop_index': i,
                'predicted_ratio': predicted_ratio
            })

            all_ratios.append(predicted_ratio)

        except Exception as e:
            print(f"Error predicting crop {i}: {e}")
            continue

    if not crop_predictions or len(all_ratios) == 0:
        return {
            'predicted_ratio': 1.0,  # Default to balanced ratio
            'confidence': 0.0,
            'std_deviation': 0.0,
            'num_crops': len(crops),
            'crop_predictions': [],
            'agreement_ratio': 0.0,
            'reliable_predictions': 0,
            'prediction_stats': {}
        }

    # Calculate statistics
    all_ratios = np.array(all_ratios)
    mean_ratio = np.mean(all_ratios)
    median_ratio = np.median(all_ratios)
    std_ratio = np.std(all_ratios)
    min_ratio = np.min(all_ratios)
    max_ratio = np.max(all_ratios)

    # Calculate confidence based on consistency (lower std = higher confidence)
    confidence = max(0.0, 1.0 - (std_ratio / 0.2))  # Normalize std to 0-1 confidence

    # Identify outliers using IQR method
    q1 = np.percentile(all_ratios, 25)
    q3 = np.percentile(all_ratios, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter out outliers
    reliable_ratios = all_ratios[(all_ratios >= lower_bound) & (all_ratios <= upper_bound)]
    reliable_predictions = len(reliable_ratios)

    # Calculate agreement ratio (percentage of non-outlier predictions)
    agreement_ratio = reliable_predictions / len(all_ratios)

    # Decision logic for final prediction
    if reliable_predictions >= max(1, len(all_ratios) * min_agreement):
        # Use reliable predictions if we have enough agreement
        if std_ratio <= confidence_threshold:
            # High confidence - use mean
            final_ratio = np.mean(reliable_ratios)
            final_confidence = confidence
        else:
            # Lower confidence - use median for robustness
            final_ratio = np.median(reliable_ratios)
            final_confidence = confidence * 0.8  # Reduce confidence due to high variance
    else:
        # Low agreement - use median of all predictions
        final_ratio = median_ratio
        final_confidence = confidence * 0.5  # Significantly reduce confidence

    # Additional confidence adjustments
    if len(all_ratios) < 4:
        final_confidence *= 0.7  # Reduce confidence if too few crops

    if agreement_ratio < 0.5:
        final_confidence *= 0.6  # Reduce confidence if low agreement

    # Ensure confidence is between 0 and 1
    final_confidence = max(0.0, min(1.0, final_confidence))

    prediction_stats = {
        'mean': float(mean_ratio),
        'median': float(median_ratio),
        'std': float(std_ratio),
        'min': float(min_ratio),
        'max': float(max_ratio),
        'q1': float(q1),
        'q3': float(q3),
        'outliers_removed': len(all_ratios) - reliable_predictions
    }

    return {
        'predicted_ratio': float(final_ratio),
        'confidence': float(final_confidence),
        'std_deviation': float(std_ratio),
        'num_crops': len(crops),
        'crop_predictions': crop_predictions,
        'agreement_ratio': float(agreement_ratio),
        'reliable_predictions': reliable_predictions,
        'prediction_stats': prediction_stats
    }


# ============================== IMAGE PROCESSING PIPELINE ==============================
def process_single_image(args):
    img_path, out_dir, output_dir_bin, h_ratio, v_ratio,gsm,no_of_ply = args
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    t0 = time.time()
    (warp_const, weft_const, processed_img, ang, warp_conf, weft_conf,warp_thread_pos, weft_thread_pos, warp_shear_deg, weft_shear_deg) = enhanced_count_threads(img, img_name, output_dir_bin, h_ratio=h_ratio, v_ratio=v_ratio)
    print(f"\n=== {img_name} ===")
    print(f"\nConstruction Analysis")
    print(f"Detected: Warp={warp_const}, Weft={weft_const}")
    print(f"Confidence: Warp={warp_conf:.2f}, Weft={weft_conf:.2f}")
    print(f"\nCount Analysis(Ne):")
    ratio_result = predict_ratio_with_multiple_crops(
        processed_img,  # Use the processed image directly
        cr_inference_model,
        crop_size=2048,
        confidence_threshold=0.05,  # Max std deviation for reliable predictions
        min_agreement=0.6
    )

    # Print detailed results for debugging
    print(f"Ratio prediction details:")
    print(f"  Final prediction: {ratio_result['predicted_ratio']:.4f}")
    print(f"  Confidence: {ratio_result['confidence']:.3f}")
    print(f"  Standard deviation: {ratio_result['std_deviation']:.4f}")
    print(f"  Crops analyzed: {ratio_result['num_crops']}")
    print(f"  Agreement ratio: {ratio_result['agreement_ratio']:.3f}")
    print(f"  Reliable predictions: {ratio_result['reliable_predictions']}")
    print(f"  Stats: mean={ratio_result['prediction_stats']['mean']:.4f}, "
          f"median={ratio_result['prediction_stats']['median']:.4f}, "
          f"range=[{ratio_result['prediction_stats']['min']:.4f}, {ratio_result['prediction_stats']['max']:.4f}]")

    sample_weight = gsm / 1550
    # factors = FABRIC_CONSTRUCTION_FACTORS['plain_weave']
    # sample_weight = gsm * factors["coverage_factor"] / 1550
    warp_to_weft_count=ratio_result['predicted_ratio']
    lhs = (no_of_ply*warp_const + warp_to_weft_count * weft_const) * 0.0254
    weft_count = sample_weight / lhs
    warp_count = weft_count/warp_to_weft_count
    warp_count,weft_count= 595.5/(warp_count*1000),595.5/(weft_count*1000)
    print(f"warp count (Ne):{warp_count}")
    print(f"weft count (Ne):{weft_count}")
    print(f"time elapsed for construction and count analysis: {(time.time() - t0) / 60} minutes.")
    if OUTPUT_CONFIG['annotated_img']:
        output_path = os.path.join(out_dir, img_name)
        result_img = util.write_text(processed_img, warp_const, weft_const, ang, warp_conf, weft_conf, weft_rows=weft_thread_pos, warp_cols=warp_thread_pos,shear_x=warp_shear_deg, shear_y=weft_shear_deg)
        cv2.imwrite(output_path, result_img)
    return {
        'image_name': img_name,
        'gsm':gsm,
        'vertical/warp': warp_const,
        'horizontal/weft': weft_const,
        'angle(degrees)': ang,
        'warp_confidence': warp_conf,
        'weft_confidence': weft_conf,
        'shear_x(deg)': warp_shear_deg,
        'shear_y(deg)': weft_shear_deg,
        'warp_count': warp_count,
        'weft_count': weft_count,
        'no_of_ply' : no_of_ply
    }


def parallel_process_images(excel_file, output_dir, parallel=True, h_ratio=None, v_ratio=None):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    output_dir_bin = util.create_unique_directory(output_dir, "binary") if OUTPUT_CONFIG['save_binary_img'] else None
    df_input = pd.read_excel(excel_file)
    all_images = [(str(path).strip().strip('"').strip("'"), gsm,no_of_ply)
                  for path, gsm,no_of_ply in df_input[['Path to sample', 'Gsm','no_of_ply']].dropna().values.tolist()]
    args_list = [(img_path, output_dir, output_dir_bin, h_ratio, v_ratio, gsm,no_of_ply)
                 for img_path, gsm,no_of_ply in all_images]
    if parallel and len(all_images) > 0:
        with Pool(processes=min(cpu_count(), PROCESSING_CONFIG['num_cores'])) as pool:
            results = pool.map(process_single_image, args_list)
    else:
        results = [process_single_image(args) for args in args_list]
    results = [res for res in results if res is not None]
    if len(results) > 0:
        df = pd.DataFrame(results,
                          columns=['image_name', 'gsm', 'vertical/warp', 'horizontal/weft',
                                   'angle(degrees)', 'warp_confidence', 'weft_confidence',
                                   'shear_x(deg)', 'shear_y(deg)','warp_count','weft_count','no_of_ply'])
        excel_output = os.path.join(output_dir, "thread_count.xlsx")
        df.to_excel(excel_output, index=False)
    print(f"\nProcessing complete! Total time: {(time.time() - start_time) / 60:.2f} minutes")



def parallel_process_images0(input_dir, output_dir, parallel=True, h_ratio=None, v_ratio=None):
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
        df = pd.DataFrame(results, columns=['image_name', 'vertical/warp', 'horizontal/weft','angle(degrees)', 'warp_confidence', 'weft_confidence','shear_x(deg)', 'shear_y(deg)'])
        excel_file = os.path.join(output_dir, "thread_count.xlsx")
        df.to_excel(excel_file, index=False)
    print(f"\nProcessing complete! Total time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    print("Available fabric constructions:")
    for const_type in list(FABRIC_CONSTRUCTION_FACTORS.keys()):
        info = FABRIC_CONSTRUCTION_FACTORS[const_type]
        print(f"  {const_type}: {info['name']} ")
    h_ratio, v_ratio = util.get_calibration_ratio(calibration_img_path)
    print(f"Calibration ratios - h_ratio: {h_ratio:.4f}, v_ratio: {v_ratio:.4f}")
    parallel_process_images(input_dir, output_dir, parallel=True, h_ratio=h_ratio, v_ratio=v_ratio)