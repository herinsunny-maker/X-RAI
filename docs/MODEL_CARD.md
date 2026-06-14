# X-rai — Model Card (v1 Champion)

**Model:** `knee_oa_efficientnet_last_stand.h5`  
**Task:** Knee Osteoarthritis (OA) grading on knee X-rays (Kellgren-Lawrence scale 0–4)  
**Overall Accuracy:** 66.12% | **QWK Score:** 0.8194  
**Status:** ✅ Production Stage 

---

## 1. Technical Evolution (The Path to v1)

The "v1 Champion" is the result of multiple technical iterations to overcome the specific challenges of medical X-ray imaging:

*   **Iteration 1 (Baseline):** Standard CNNs with Categorical Cross-Entropy. *Result: <50% accuracy.* It treated Grade 0 vs Grade 1 errors the same as Grade 0 vs Grade 4.
*   **Iteration 2 (Backbone Shift):** Moved to **EfficientNetV2-S**. This provided a much stronger feature extraction foundation than DenseNet or ResNet for subtle bone textures.
*   **Iteration 3 (Ordinal Logic):** Switched to **Ordinal Regression**. Instead of 5 independent classes, the model now predicts the *progression* of the disease (Grade $\ge$ X). This mimics clinical logic and boosted accuracy significantly.
*   **Iteration 4 (Image Pre-processing):** Integrated **CLAHE** (Contrast Limited Adaptive Histogram Equalization). This standardizes X-rays from different machines, making the model robust to varying exposures.

---

## 2. Technical Architecture

### Backbone
- **Architecture:** EfficientNetV2-S (ImageNet pre-trained)
- **Input Size:** 224 x 224 pixels

### Classification Head (Ordinal Regression)
- **Logic:** The model uses **Ordinal Labeling** (e.g., Grade 2 is encoded as `[1, 1, 0, 0]`).
- **Layers:** GlobalAveragePooling2D → BatchNormalization → Dropout (0.5) → Dense (4 units, Sigmoid)
- **Decoding:** The KL Grade (0-4) is the count of probabilities $> 0.5$.

---

## 3. Performance Results (1,656 Test Images)

| Clinical KL-Grade | Recall (Sensitivity) | Description |
| :--- | :--- | :--- |
| **Grade 0 (Normal)** | 78% | Success in screening out healthy knees |
| **Grade 1 (Doubtful)** | 40% | The "Bottleneck" — high clinical subjectivity |
| **Grade 2 (Mild)** | 60% | Definite findings (Osteophytes) |
| **Grade 3 (Moderate)** | 74% | Clear joint space narrowing |
| **Grade 4 (Severe)** | 88% | Bone-on-bone contact |

**Quadratic Weighted Kappa (QWK): 0.82**  
*Interpreted as: "Strong clinical agreement." Errors are mostly 1-grade adjacent, which mimics human radiologist variance.*

---

## 4. Interpretability (Grad-CAM)

The system uses **Grad-CAM** to visualize clinical evidence:
- **Source:** `top_activation` layer of the EfficientNet backbone.
- **Method:** Logit-based gradients ensure heatmaps remain focused even on high-confidence cases.
- **Visual:** Smooth heatmaps centered on joint compartments, highlighting narrowing and osteophytes.

---

## 5. Usage & Integration

- **Binary Pre-check:** A separate CNN (`binary_xray_validator.h5`) verifies if the upload is a knee X-ray (Accuracy: >95%).
- **Inference Entry Point:** `ModelHandler.predict(image_path)` in `backend/model_handler.py`.
- **Environment:** TensorFlow 2.10 (Optimized for GPU).

---
*Developed for the X-rai Radiology Support Platform.*
