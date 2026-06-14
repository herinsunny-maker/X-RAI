# JavaScript Functions — X-rai
> **Purpose:** Maps every client-side JS function/handler to what it does, where it lives, and what DOM elements it reads or writes.  
> Use this during demos: find the function an invigilator asks about → open the template → show the code.

---

## `quick_check.html` — JS Functions

Located in: `backend/templates/quick_check.html` — inside the `<script>` block.

### Variables (module-level)

| Variable | Type | What it holds |
|---|---|---|
| `dropZone` | DOM element | The dashed upload area (`#drop-zone`). |
| `fileInput` | DOM element | Hidden `<input type="file">` (`#file-input`). |
| `form` | DOM element | The `<form id="quick-check-form">`. |
| `cnnToggle` | DOM element | The CNN guardrail checkbox (`name="cnn_enabled"`). |
| `resContainer` | DOM element | The `#result-container` div (results panel). |
| `loading` | DOM element | The spinner `#loading` div. |
| `previewWrap` | DOM element | The `#preview-wrap` image preview container. |
| `previewImg` | DOM element | The `<img id="preview-img">` thumbnail. |
| `lastResults` | Object | Stores the last JSON response from `/api/quick_predict` so it can be re-sent to `/api/quick_save`. Starts as `null`. |

---

### Functions

#### `showPreview(file)`
| Attribute | Value |
|---|---|
| **Template** | `quick_check.html` |
| **Triggered by** | `fileInput.onchange` or `dropZone.ondrop` |
| **Purpose** | Reads the selected `File` object using the browser's `FileReader` API, converts it to a base64 data URL, and sets it as the `src` of `#preview-img`, making the thumbnail visible before the form is submitted. Also displays the file name in the preview bar. |
| **Reads** | `file` (File object from the input) |
| **Writes** | `previewImg.src`, `#preview-filename.textContent`, `previewWrap.style.display`, `resContainer.style.display` |

---

#### `clearFile()`
| Attribute | Value |
|---|---|
| **Template** | `quick_check.html` |
| **Triggered by** | `#remove-img-btn` click |
| **Purpose** | Resets the file input by setting `fileInput.value = ''` (which browser engines interpret as "no file selected"), clears the preview image src, and hides `#preview-wrap`. Also nullifies `lastResults`. |
| **Reads** | — |
| **Writes** | `fileInput.value`, `previewImg.src`, `previewWrap.style.display`, `lastResults` |

---

#### `resetCheck()`
| Attribute | Value |
|---|---|
| **Template** | `quick_check.html` |
| **Triggered by** | "Discard & New Check" button `onclick` |
| **Purpose** | Calls `location.reload()` — reloads the entire page, giving the user a completely clean state without any stale result data. |

---

#### `form.onsubmit` (anonymous async handler)
| Attribute | Value |
|---|---|
| **Template** | `quick_check.html` |
| **Triggered by** | Form submission (clicking "Run AI Analysis") |
| **Purpose** | Prevents the default HTML form POST. Builds a `FormData` object from the form (which automatically picks up the `cnn_enabled` checkbox) and sends it to `/api/quick_predict` via **`fetch()`** (AJAX — no page reload). Displays the spinner during the request. On success: populates both result images and the metric card. On error: shows the error message panel. |
| **Reads** | `fileInput.files`, `cnnToggle.checked` |
| **Writes** | `loading.style.display`, `resContainer.style.display`, `#res-xray.src`, `#res-heatmap.src`, `#res-prediction.textContent`, `#res-confidence.textContent`, `#res-error.textContent`, `lastResults` |

---

#### `#quick-save-btn.onclick` (anonymous async handler)
| Attribute | Value |
|---|---|
| **Template** | `quick_check.html` |
| **Triggered by** | Clicking "Save to Database (Draft)" button |
| **Purpose** | POSTs `lastResults` (the stored JSON from the last prediction) to `/api/quick_save` as a JSON body. On success: shows a confirmation alert and redirects to `/dashboard`. On failure: alerts the user to check server logs. |
| **Reads** | `lastResults` |
| **Writes** | `window.location.href` (on success) |

---

### Event Handlers (inline, no named function)

| Handler | Element | What it does |
|---|---|---|
| `dropZone.onclick` | `#drop-zone` | Calls `fileInput.click()` to open the OS file picker |
| `dropZone.ondragover` | `#drop-zone` | Prevents default browser behaviour and adds `.active` class to highlight the zone |
| `dropZone.ondragleave` | `#drop-zone` | Removes `.active` class when the dragged file leaves the zone |
| `dropZone.ondrop` | `#drop-zone` | Assigns `e.dataTransfer.files` to `fileInput.files`, then calls `showPreview()` |
| `fileInput.onchange` | `#file-input` | Calls `showPreview(fileInput.files[0])` when file is picked from dialog |
| `#remove-img-btn.onclick` | Remove button | Calls `clearFile()` |

---

## `new_patient.html` — JS Functions

Located in: `backend/templates/new_patient.html` — inside the `<script>` block.

### Variables (module-level)

| Variable | Type | What it holds |
|---|---|---|
| `dropZone` | DOM element | The upload drop zone (`#drop-zone`) |
| `fileInput` | DOM element | Hidden `<input type="file" id="file">` |
| `previewWrap` | DOM element | `#preview-wrap` — the image preview container |
| `previewImg` | DOM element | `<img id="preview-img">` — the thumbnail |

---

### Functions

#### `showPreview(file)`
| Attribute | Value |
|---|---|
| **Template** | `new_patient.html` |
| **Purpose** | Same as in `quick_check.html` — reads the image with `FileReader`, shows a local thumbnail so staff can confirm the correct X-ray is selected before submitting the registration form. Also changes the drop zone border to the primary colour as visual confirmation. |
| **Reads** | `file` (File object) |
| **Writes** | `previewImg.src`, `#preview-filename.textContent`, `previewWrap.style.display`, `dropZone.style.borderColor` |

---

#### `clearPreview()`
| Attribute | Value |
|---|---|
| **Template** | `new_patient.html` |
| **Triggered by** | `#remove-img-btn` click, or the Cancel/Reset button `onclick` |
| **Purpose** | Clears the file input value, resets the preview image src, hides `#preview-wrap`, and resets the drop zone border colour back to the default. |
| **Writes** | `fileInput.value`, `previewImg.src`, `previewWrap.style.display`, `dropZone.style.borderColor` |

---

### Event Handlers (inline)

| Handler | Element | What it does |
|---|---|---|
| `dropZone.onclick` | `#drop-zone` | Opens the file picker via `fileInput.click()` |
| `fileInput.onchange` | `#file` | Calls `showPreview(fileInput.files[0])` |
| `#remove-img-btn.onclick` | Remove button | Calls `clearPreview()` |
| `dropZone.ondragover` | `#drop-zone` | Highlights border with primary colour |
| `dropZone.ondragleave` | `#drop-zone` | Resets border to default |
| `dropZone.ondrop` | `#drop-zone` | Sets `fileInput.files` from drag data, calls `showPreview()` |
| Cancel button `onclick` | Reset button | Calls `clearPreview()` additionally to the natural form reset |

---

## `staff_dashboard.html` — JS

No JavaScript is used in `staff_dashboard.html`. All interactions (View, Delete, New Patient) are standard HTML links or form POSTs handled entirely by the server.

---

## `case_preview.html` — JS

No JavaScript is used in `case_preview.html`. The "Send to Doctor" button is a plain HTML `<form method="POST">` that triggers the `send_to_doctor` Flask route.

---

## `login.html` — JS Functions

Located in: `backend/templates/login.html` — inside the `<script>` block.

### Functions

#### `togglePatientLogin(showPatient)`
| Attribute | Value |
|---|---|
| **Template** | `login.html` |
| **Purpose** | Toggles visibility between the Staff/Doctor login form and the Patient Portal login form. Updates the card subtitle text. |
| **Reads** | `showPatient` (boolean) |
| **Writes** | `staff-login-form.style.display`, `patient-login-form.style.display`, `#login-subtitle.textContent` |

---

#### `showOtpField()`
| Attribute | Value |
|---|---|
| **Template** | `login.html` |
| **Purpose** | Validates that a phone number is entered, then hides the phone input section and reveals the OTP input section and "Verify" button. |
| **Reads** | `#phone.value` |
| **Writes** | `#patient-phone-section.style.display`, `#patient-otp-section.style.display`, `#send-otp-btn.style.display`, `#verify-otp-btn.style.display` |

---

## `base.html` — JS
...

---

## Key Browser APIs Used (for invigilator explanation)

| API | Where used | What it does |
|---|---|---|
| `FileReader.readAsDataURL()` | `showPreview()` in both templates | Reads a local File from the OS (not yet uploaded) and encodes it as a base64 string that can be used as an `<img>` `src` — so the preview appears without any server round-trip. |
| `fetch()` | `form.onsubmit`, `quick-save-btn.onclick` in `quick_check.html` | Sends HTTP requests to the Flask API from JavaScript (AJAX) without reloading the page. Returns a `Promise` — `await` pauses execution until the response arrives. |
| `FormData` | `form.onsubmit` in `quick_check.html` | Serialises the form fields (including the `cnn_enabled` checkbox and binary file) into a multipart request body, the same format as a normal HTML form POST. |
| `DataTransfer.files` | `dropZone.ondrop` | Browser drag-and-drop API — `e.dataTransfer.files` gives the dropped `FileList`, which is assigned directly to `fileInput.files`. |
