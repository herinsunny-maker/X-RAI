# Python Functions & Classes — X-rai
> **Purpose:** Quick reference for invigilators and students.  
> Each entry shows: what it does, where to find it, and what it takes/returns.

---

## `app.py` — Flask Web Application (Routes & Logic)

> **Run with:** `python app.py` from the `backend/` folder  
> **Base URL:** `http://localhost:5000`

| # | Route / Function | Method | Location | Purpose |
|---|---|---|---|---|
| 1 | `login()` | GET, POST | `app.py : 58` | **GET** renders the login page. **POST** checks email+password against the `users` table. On success, stores `user_id`, `name`, `role` in the Flask session and redirects to `/dashboard`. |
| 2 | `logout()` | GET | `app.py : 84` | Calls `session.clear()` to wipe all session data, then redirects to `/login`. |
| 3 | `dashboard()` | GET | `app.py : 110` | Main landing page. Redirects patients to `/patient/dashboard`. For staff, queries last 20 reports and KPI counts. Renders `staff_dashboard.html`. |
| 4 | `patient_login()` | POST | `app.py : 101` | Validates patient phone number against the `patients` table. On success, stores `patient_phone` in session and redirects to patient dashboard. |
| 5 | `patient_dashboard()` | GET | `app.py : 513` | Fetches all reports linked to the logged-in patient's phone number. Renders `patient_dashboard.html`. |
| 6 | `new_patient()` | GET, POST | `app.py : 162` | **GET** renders the registration form. **POST** saves uploaded X-ray, runs AI, inserts `patients` and `reports` rows, and redirects to `case_preview`. |
| 5 | `case_preview(report_id)` | GET | `app.py : 196` | Fetches a single report + linked patient from DB. Packages `xray_path`, `overlay_path`, `prediction`, `confidence` into a `results` dict and renders `case_preview.html`. |
| 8 | `send_to_doctor(report_id)` | POST | `app.py : 316` | Updates patient details (name, age, etc.) if they were edited in preview, and updates `reports.status` to `'pending'`. Redirects to dashboard. |
| 9 | `view_report(report_id)` | GET | `app.py : 547` | Renders a detailed, A4-style `report_detail.html` for printing. Includes X-ray, heatmap, and final diagnosis. |
| 7 | `delete_patient(patient_id)` | POST | `app.py : 240` | Finds all reports for the patient, **deletes physical image files** from disk (xray, heatmap, overlay), then deletes the `reports` and `patients` rows from the database. |
| 8 | `quick_check()` | GET | `app.py : 278` | Staff-only route that simply renders `quick_check.html` (no DB query needed). |
| 9 | `quick_predict()` | POST | `app.py : 290` | **API endpoint** called by JavaScript fetch. Accepts a multipart file + `skip_validation` flag. Saves to `TEMP_FOLDER`, calls `handler.predict()`, moves to `UPLOAD_FOLDER` if valid. Returns JSON. |
| 10 | `quick_save()` | POST | `app.py : 321` | **API endpoint** called by JavaScript fetch. Accepts JSON with prediction results. Creates an anonymous `patients` row (name = "Quick Test — Grade X") and a `draft` report row in the DB. |
| 11 | `serve_uploads(filename)` | GET | `app.py : 353` | Static file server for uploaded images. Flask's built-in `/static/` doesn't cover the dynamic upload folder path, so this explicit route handles it. |
| 12 | `pending_cases()` | GET | `app.py : 365` | Doctor route. Queries all reports with `status='pending'` and renders them in `staff_dashboard.html`. |
| 16 | `review_case(report_id)` | GET, POST | `app.py : 590` | **GET** renders `case_review.html`. **POST** processes doctor's decision. `approve` sets `status='reviewed'` and `final_grade`. |

---

## `model_handler.py` — AI Model Wrapper Class

> **Class:** `ModelHandler`  
> **File:** `backend/model_handler.py`  
> Used by `app.py` as a singleton: `handler = ModelHandler()`

| # | Method | Location | Purpose |
|---|---|---|---|
| 1 | `__init__(mock_mode=False)` | `model_handler.py : 16` | Constructor. If `mock_mode=True`, skips loading real models and returns random predictions (useful for UI testing without GPU). Otherwise calls `load_models()`. |
| 2 | `load_models()` | `model_handler.py : 24` | Loads two Keras `.h5` files: **OA model** (EfficientNetV2S, classifies knee OA grade) and **binary validator** (lightweight CNN, checks if image is a knee X-ray). Falls back to weight-only loading if full-model load fails. |
| 3 | `_build_oa_model()` | `model_handler.py : 57` | Private helper — rebuilds the EfficientNetV2S architecture in Python (used only if the `.h5` cannot be loaded directly due to Keras version issues). |
| 4 | `preprocess_image(image_path)` | `model_handler.py : 70` | Reads the image with OpenCV, converts BGR→RGB, applies **CLAHE** (contrast enhancement matching training pipeline), resizes to 224×224, expands to batch (1,224,224,3). Returns `(original_img, img_array)`. |
| 5 | `make_gradcam_heatmap(img_array)` | `model_handler.py : 84` | **Saliency/Grad-CAM** — wraps the image in a `tf.Variable`, runs a forward pass under `tf.GradientTape` to record gradients, computes which pixels influenced the prediction most, returns a normalised float heatmap `(H, W)` and the raw sigmoid predictions array. |
| 6 | `save_and_overlay(heatmap, original_img, filename)` | `model_handler.py : 125` | Resizes the heatmap to match the original image, applies `COLORMAP_JET` to colour it, blends heatmap (45%) + original (55%) to create an overlay. Writes both `heatmap_<name>.jpg` and `overlay_<name>.jpg` to `static/uploads/`. Returns `(heatmap_path, overlay_path)`. |
| 7 | `predict(image_path, skip_validation=False)` | `model_handler.py : 145` | **Main entry point** called by Flask routes. Pipeline: ① binary validation (unless skipped), ② preprocess, ③ Grad-CAM, ④ ordinal regression decoding, ⑤ assemble result dict. Returns a dict with `is_xray`, `prediction`, `confidence`, `grade`, `xray_path`, `heatmap_path`, `overlay_path`, `all_predictions`. |

---

## `database.py` — SQLite Connection Helper

> **File:** `backend/database.py`

| # | Function | Location | Purpose |
|---|---|---|---|
| 1 | `get_db_connection()` | `database.py : ~10` | Opens a connection to `knee_oa.db` using `sqlite3`. Sets `row_factory = sqlite3.Row` so query results can be accessed like dictionaries (e.g. `row['name']`). Returns the connection object — caller must close it. |

---

## Data Decode Logic — Ordinal Regression

The OA model (`knee_oa_efficient net v1`) uses **ordinal regression** (4 sigmoid outputs).

| Sigmoid Output | Meaning |
|---|---|
| `preds[0]` | P(grade ≥ 1) |
| `preds[1]` | P(grade ≥ 2) |
| `preds[2]` | P(grade ≥ 3) |
| `preds[3]` | P(grade ≥ 4) |

**Grade decoding:** `grade = count of preds > 0.5`  
- 0 thresholds passed → Grade 0 (Normal)  
- 4 thresholds passed → Grade 4 (Severe)

---

## Database Schema (reference)

```
patients  (id, name, age, gender, phone, created_by, created_at)
reports   (id, patient_id, xray_path, heatmap_path, overlay_path,
           ai_grade, ai_confidence, final_grade, doctor_notes,
           staff_notes, status [draft|pending|reviewed], created_at)
users     (id, name, email, password_hash, role [staff|doctor])
```
