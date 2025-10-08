import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops, label, shannon_entropy
from skimage import io, color, filters, img_as_ubyte, exposure
from skimage.transform import resize
from scipy.stats import skew, kurtosis
import os
import joblib


def extract_enhanced_features(image_path):
    """
    Extract comprehensive features:
    - Morphological (7)
    - Texture GLCM (6)
    - Color histogram (30)
    - LBP texture (10)
    - Statistical (8)
    Total: ~60+ features
    """
    img = io.imread(image_path)

    # Convert to grayscale
    if img.ndim == 3:
        img_gray = color.rgb2gray(img)
        img_color = img.copy()
    else:
        img_gray = img.astype(np.float32)
        if img_gray.max() > 1.0:
            img_gray = img_gray / 255.0
        img_color = None

    # Resize for consistency
    img_resized = resize(img_gray, (128, 128), anti_aliasing=True)
    img_u8 = img_as_ubyte(img_resized)

    features = []

    # ==========================================
    # 1. MORPHOLOGICAL FEATURES (7)
    # ==========================================
    thresh = filters.threshold_otsu(img_resized)
    mask = img_resized > thresh
    lbl = label(mask)
    props = regionprops(lbl)

    if props:
        p0 = max(props, key=lambda r: r.area)
        features.extend(
            [
                float(p0.area),
                float(p0.perimeter),
                float(p0.eccentricity),
                float(p0.solidity),
                float(p0.extent),
                float(p0.major_axis_length),
                float(p0.minor_axis_length),
            ]
        )
    else:
        features.extend([0.0] * 7)

    # ==========================================
    # 2. GLCM TEXTURE FEATURES (6)
    # ==========================================
    glcm = graycomatrix(
        img_u8,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    features.extend(
        [
            float(graycoprops(glcm, "contrast").mean()),
            float(graycoprops(glcm, "dissimilarity").mean()),
            float(graycoprops(glcm, "homogeneity").mean()),
            float(graycoprops(glcm, "energy").mean()),
            float(graycoprops(glcm, "correlation").mean()),
            float(graycoprops(glcm, "ASM").mean()),
        ]
    )

    # ==========================================
    # 3. COLOR HISTOGRAM FEATURES (30)
    # ==========================================
    if img_color is not None and img_color.ndim == 3:
        img_color_resized = resize(img_color, (128, 128), anti_aliasing=True)

        # RGB histograms (10 bins each = 30 features)
        for ch in range(3):
            channel = (img_color_resized[:, :, ch] * 255).astype(np.uint8)
            hist, _ = np.histogram(channel, bins=10, range=(0, 256))
            hist = hist.astype(float) / (hist.sum() + 1e-7)  # Normalize
            features.extend(hist.tolist())
    else:
        features.extend([0.0] * 30)

    # ==========================================
    # 4. LOCAL BINARY PATTERN (LBP) (10)
    # ==========================================
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img_u8, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-7)
    features.extend(lbp_hist.tolist())

    # ==========================================
    # 5. STATISTICAL FEATURES (8)
    # ==========================================
    flat = img_resized.ravel()
    features.extend(
        [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.median(flat)),
            float(np.min(flat)),
            float(np.max(flat)),
            float(skew(flat)),
            float(kurtosis(flat)),
            float(shannon_entropy(img_u8)),
        ]
    )

    return features


def extract_all_features(dataset_dir):
    X, y, paths = [], [], []

    for label_name in sorted(os.listdir(dataset_dir)):
        folder = os.path.join(dataset_dir, label_name)
        if not os.path.isdir(folder):
            continue

        print(f"Processing class: {label_name}")
        imgs = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ]

        for fn in imgs:
            p = os.path.join(folder, fn)
            try:
                feats = extract_enhanced_features(p)
                X.append(feats)
                y.append(label_name)
                paths.append(p)
            except Exception as e:
                print(f"[WARN] Skipped {p}: {e}")

    return np.array(X, dtype=np.float32), np.array(y), paths


if __name__ == "__main__":
    TRAIN_DIR = r"C:\Users\htagi\bacterias_splited\train"
    VAL_DIR = r"C:\Users\htagi\bacterias_splited\val"
    SAVE_DIR = r"C:\Users\htagi\bacteria_features"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Extract train features
    print("Extracting TRAIN features...")
    X_train, y_train, paths_train = extract_all_features(TRAIN_DIR)
    print(f"Train: {X_train.shape}")
    np.save(os.path.join(SAVE_DIR, "features_train.npy"), X_train)
    np.save(os.path.join(SAVE_DIR, "labels_train.npy"), y_train)
    joblib.dump(paths_train, os.path.join(SAVE_DIR, "paths_train.pkl"))

    # Extract validation features
    print("\nExtracting VAL features...")
    X_val, y_val, paths_val = extract_all_features(VAL_DIR)
    print(f"Val: {X_val.shape}")
    np.save(os.path.join(SAVE_DIR, "features_val.npy"), X_val)
    np.save(os.path.join(SAVE_DIR, "labels_val.npy"), y_val)
    joblib.dump(paths_val, os.path.join(SAVE_DIR, "paths_val.pkl"))

    print(f"\nSaved to: {SAVE_DIR}")
    print(f"Feature dimension: {X_train.shape[1]}")
