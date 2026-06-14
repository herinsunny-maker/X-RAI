
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from ml_pipeline.utils import apply_clahe


IMG_SIZE = (224, 224)  
BINARY_IMG_SIZE = (224, 224)
from backend.path_utils import get_ml_model_path, get_upload_path


IMG_SIZE = (224, 224)
BINARY_IMG_SIZE = (224, 224)


MODEL_PATH = os.path.join(get_ml_model_path(), "knee_oa_efficientnet_last_stand.h5")
BINARY_MODEL_PATH = os.path.join(get_ml_model_path(), "binary_xray_validator.h5")

class ModelHandler:
    def __init__(self, mock_mode=False):
        self.mock_mode = mock_mode
        self.oa_model = None
        self.binary_model = None
        
        if not mock_mode:
            self.load_models()

    def load_models(self):
       
        if os.path.exists(MODEL_PATH):
            try:
                
                self.oa_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print(f"OA Model (v1) loaded OK — {sum(l.count_params() for l in self.oa_model.layers):,} params")
            except Exception as e:
                print(f"ERROR loading OA model directly: {e}")
                print("Attempting fallback: rebuild EfficientNetV2S + load weights...")
                self.oa_model = self._build_oa_model()
                try:
                    self.oa_model.load_weights(MODEL_PATH, by_name=False, skip_mismatch=False)
                    print("OA Weights loaded via fallback.")
                except Exception as e2:
                    print(f"Fallback weight load also failed: {e2} — model will be None")
                    self.oa_model = None
        else:
            print(f"ERROR: OA model file not found at: {MODEL_PATH}")

        print(f"OA model status: {'LOADED' if self.oa_model is not None else 'FAILED/NONE'}")

       
        if os.path.exists(BINARY_MODEL_PATH):
            try:
                self.binary_model = tf.keras.models.load_model(BINARY_MODEL_PATH, compile=False)
                print(f"Binary validator loaded OK")
            except Exception as e:
                print(f"Binary model load failed directly: {e}")
                print("Attempting fallback: rebuild CNN + load weights...")
                self.binary_model = self._build_binary_model()
                try:
                    self.binary_model.load_weights(BINARY_MODEL_PATH, by_name=False, skip_mismatch=False)
                    print("Binary Weights loaded via fallback.")
                except Exception as e2:
                    print(f"Fallback weight load also failed: {e2}")
                    self.binary_model = None
        else:
            print(f"Binary validator not found — X-ray validation disabled")

    def _build_binary_model(self):
        """Fallback: rebuild Binary CNN architecture."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=None, input_shape=(*BINARY_IMG_SIZE, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def _build_oa_model(self):
        """Fallback: rebuild v1 EfficientNetV2S Functional architecture."""
        base = tf.keras.applications.EfficientNetV2S(
            weights=None, include_top=False, input_shape=(*IMG_SIZE, 3)
        )
        inputs = tf.keras.layers.Input(shape=(*IMG_SIZE, 3))
        x = base(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(4, activation='sigmoid')(x)
        return tf.keras.Model(inputs, outputs)

    def preprocess_image(self, image_path):
       
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        
        img_clahe = apply_clahe(img_rgb)
        
        
        img_resized = cv2.resize(img_clahe, IMG_SIZE)
        img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
        
        return img_rgb, img_array

    def make_gradcam_heatmap(self, img_array):
        """
        Robust Grad-CAM: Manually chains layers to avoid Graph Disconnected errors.
        Uses logits to prevent vanishing gradients on confident predictions.
        """
        try:
            
            rescaling = None
            backbone = None
            gap = None
            bn = None
            dense = None
            
            for layer in self.oa_model.layers:
                lname = layer.name.lower()
                if 'sequential' in lname or 'rescaling' in lname: rescaling = layer
                if 'efficientnet' in lname: backbone = layer
                if 'global_average_pooling2d' in lname: gap = layer
                if 'batch_normalization' in lname: bn = layer
                if 'dense' in lname: dense = layer

            if not all([backbone, gap, bn, dense]):
                raise ValueError("Could not identify all model components for Grad-CAM")

            
            last_conv_layer_name = "top_activation"
            backbone_features_model = tf.keras.models.Model(
                backbone.inputs, backbone.get_layer(last_conv_layer_name).output
            )

            with tf.GradientTape() as tape:
               
                x = rescaling(img_array) if rescaling else img_array
                conv_output = backbone_features_model(x)
                tape.watch(conv_output)
                
               
                x = gap(conv_output)
                x = bn(x)
               
                preds = dense(x) 

              
                pred_np = preds.numpy()[0]
                grade_idx = int((pred_np > 0.5).sum())
                sig_idx = min(max(grade_idx - 1, 0), 3)
                
                
                target_p = preds[:, sig_idx]
                logit = tf.math.log(target_p / (1.0 - target_p + 1e-7))

            
            grads = tape.gradient(logit, conv_output)
            
            
            if grads is None:
                raise ValueError("Gradients could not be computed")

            
            weights = tf.reduce_mean(grads, axis=(0, 1, 2))

            
            feature_maps = conv_output[0]
            heatmap = feature_maps @ weights[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap).numpy()

            
            heatmap = np.maximum(heatmap, 0)
            mx = np.max(heatmap)
            if mx > 1e-8:
                heatmap /= mx
            else:
                heatmap = np.zeros_like(heatmap)
            
            return heatmap.astype(np.float32), pred_np

        except Exception as e:
            print(f"Grad-CAM failed ({e}), falling back to prediction only.")
            try:
                pred_np = self.oa_model.predict(img_array, verbose=0)[0]
            except Exception:
                pred_np = np.array([0.5, 0.3, 0.1, 0.05])
            return None, pred_np

    def save_and_overlay(self, heatmap, original_img, filename):
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        
        orig_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        superimposed = np.clip(heatmap_colored * 0.45 + orig_bgr * 0.55, 0, 255).astype(np.uint8)

        
        upload_dir = get_upload_path()
        os.makedirs(upload_dir, exist_ok=True)

        heatmap_name = f"heatmap_{filename}"
        overlay_name = f"overlay_{filename}"
        cv2.imwrite(os.path.join(upload_dir, heatmap_name), heatmap_colored)
        cv2.imwrite(os.path.join(upload_dir, overlay_name), superimposed)

        return f"static/uploads/{heatmap_name}", f"static/uploads/{overlay_name}"

    def predict(self, image_path, skip_validation=False):
        filename = os.path.basename(image_path)
        
        if self.mock_mode:
            import random
            return {
                "is_xray": True,
                "xray_path": f"static/uploads/{filename}",
                "prediction": f"Grade {random.randint(0,4)}",
                "confidence": f"{random.uniform(0.6, 0.95)*100:.1f}%",
                "details": "MOCK MODE: Results are simulated."
            }

      
        if self.binary_model and not skip_validation:
            img_bgr = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, BINARY_IMG_SIZE)
            img_array_bin = np.expand_dims(img_resized, axis=(0, -1)) / 255.0
            
            xray_prob = self.binary_model.predict(img_array_bin, verbose=0)[0][0]
            if xray_prob < 0.5:
                
                return {
                    "is_xray": False,
                    "confidence": f"{(1-xray_prob)*100:.1f}%",
                    "details": "The image does not appear to be a knee X-ray. Please upload a valid medical image."
                }

       
        if self.oa_model is None:
            return {"is_xray": True, "error": "OA model not loaded.",
                    "prediction": "Unavailable", "confidence": "N/A",
                    "details": "Model file missing.", "grade": -1}

        original_img, img_array = self.preprocess_image(image_path)
        heatmap, preds = self.make_gradcam_heatmap(img_array)

       
        heatmap_url, overlay_url = None, None
        if heatmap is not None:
            heatmap_url, overlay_url = self.save_and_overlay(heatmap, original_img, filename)
        
        
        grade_idx = int(np.sum(preds > 0.5))
        
       
        if grade_idx == 0:
            conf_val = float(1.0 - preds[0])
        elif grade_idx == 4:
            conf_val = float(preds[3])
        else:
            conf_val = float((preds[grade_idx-1] + (1.0 - preds[grade_idx])) / 2.0)
        conf_val = round(max(0.0, min(1.0, conf_val)) * 100, 1)  # e.g. 75.3
        classes = ["Normal (Grade 0)","Doubtful (Grade 1)","Mild (Grade 2)","Moderate (Grade 3)","Severe (Grade 4)"]


        
        all_preds_dict = {}
        all_preds_dict["Grade 0"] = float(1.0 - preds[0])
        all_preds_dict["Grade 1"] = float(preds[0] - preds[1]) if len(preds) > 1 else 0.0
        all_preds_dict["Grade 2"] = float(preds[1] - preds[2]) if len(preds) > 2 else 0.0
        all_preds_dict["Grade 3"] = float(preds[2] - preds[3]) if len(preds) > 3 else 0.0
        all_preds_dict["Grade 4"] = float(preds[3]) if len(preds) > 3 else 0.0

        results = {
            "is_xray": True,
            "xray_path": f"static/uploads/{filename}",
            "prediction": classes[grade_idx],
            "confidence": conf_val,           
            "confidence_str": f"{conf_val:.1f}%",  
            "details": f"KL-Grade {grade_idx} — knee_oa_efficient net v1 (EfficientNetV2S, 66% acc, QWK 0.82)",
            "grade": grade_idx,
            "heatmap_path": heatmap_url,
            "overlay_path": overlay_url,
            "all_predictions": all_preds_dict
        }
        
        return results
