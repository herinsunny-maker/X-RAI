
import os
import sys
import time

from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, session, flash, send_from_directory
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.model_handler import ModelHandler
from backend.database import get_db_connection

from backend.path_utils import get_upload_path, get_temp_path


app = Flask(__name__)
app.secret_key = "d03b07044453a985e135548d1c68f6041accd5f187796d11e5927c735d465352"
CORS(app)


UPLOAD_FOLDER = get_upload_path()
TEMP_FOLDER   = get_temp_path()

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER,   exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


handler = ModelHandler()


def calculate_bmi(height, weight):
    """Returns (bmi_value, bmi_category, healthy_weight_range) or (None, None, None)"""
    if not height or not weight:
        return None, None, None
    try:
        height_m = float(height) / 100
        weight_kg = float(weight)
        if height_m <= 0:
            return None, None
        bmi = round(weight_kg / (height_m * height_m), 1)
        
        if bmi < 18.5:
            cat = "Underweight"
        elif bmi < 25:
            cat = "Normal"
        elif bmi < 30:
            cat = "Overweight"
        else:
            cat = "Obese"
            
        min_weight = round(18.5 * height_m * height_m, 1)
        max_weight = round(24.9 * height_m * height_m, 1)
        healthy_range = f"{min_weight} - {max_weight} kg"
        
        return bmi, cat, healthy_range
    except (ValueError, TypeError):
        return None, None, None



@app.route("/login", methods=["GET", "POST"])
def login():
    """
    GET  /login  — render the login form.
    POST /login  — validate credentials from the 'users' table.
                   On success: store user_id, name, role in session
                   and redirect to /dashboard.
                   On failure: flash an error and re-render the form.
    """
    if request.method == "POST":
        email    = request.form.get("email")
        password = request.form.get("password")

        db   = get_db_connection()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        db.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["name"]    = user["name"]
            session["role"]    = user["role"]
            flash(f"Welcome back, {user['name']}!", "success")
            if user["role"] == "doctor":
                return redirect(url_for("pending_cases"))
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    """
    Clears the user session and redirects to /login.
    Any sensitive session data (user_id, role, name, patient_phone) is removed.
    """
    session.clear()
    return redirect(url_for("login"))


@app.route("/patient_login", methods=["POST"])
def patient_login():
    """
    POST /patient_login  — validate patient from the 'patients' table by phone.
                       On success: store patient_phone in session
                       and redirect to /patient/dashboard.
                       On failure: flash an error and redirect back to login.
    """
    phone = request.form.get("phone")
    otp = request.form.get("otp") # Currently ignoring actual OTP validation for demo purposes

    if not phone:
        flash("Mobile number is required.", "error")
        return redirect(url_for("login"))

    db = get_db_connection()
    
    patient = db.execute("SELECT * FROM patients WHERE phone = ?", (phone,)).fetchone()
    db.close()

    if patient:
        session["patient_phone"] = phone
        session["role"] = "patient"
        session["name"] = patient["name"]
        flash(f"Welcome to your Patient Portal, {patient['name']}!", "success")
        return redirect(url_for("patient_dashboard"))
    else:
        flash("No patient found with this mobile number.", "error")
        return redirect(url_for("login"))


@app.route("/")
@app.route("/dashboard")
def dashboard():
    """
    GET /dashboard — main landing page after login.

    Queries the database for:
      - reports: last 10 cases joined with patient names (for table display)
      - stats:   total patients, pending reviews, reviewed cases (for KPI cards)

    Renders staff_dashboard.html with {reports, stats}.
    Non-staff users are redirected to /login.
    """
    if "user_id" not in session and "patient_phone" not in session:
        return redirect(url_for("login"))
    
   
    if session.get("role") == "patient":
        return redirect(url_for("patient_dashboard"))

   
    if session["role"] != "staff":
        flash("Access restricted to staff accounts.", "error")
        return redirect(url_for("login"))

    db = get_db_connection()
    reports = db.execute("""
        SELECT r.*, p.name AS patient_name, p.height, p.weight
        FROM   reports r
        JOIN   patients p ON r.patient_id = p.id
        ORDER  BY r.created_at DESC
        LIMIT  20
    """).fetchall()

    reports_list = []
    for r in reports:
        r_dict = dict(r)
        bmi, bmi_cat, healthy_weight = calculate_bmi(r["height"], r["weight"])
        r_dict["bmi"] = bmi
        r_dict["bmi_cat"] = bmi_cat
        reports_list.append(r_dict)

    stats = {
        "total_patients":  db.execute("SELECT COUNT(*) FROM patients").fetchone()[0],
        "pending_review":  db.execute("SELECT COUNT(*) FROM reports WHERE status='pending'").fetchone()[0],
        "approved_cases":  db.execute("SELECT COUNT(*) FROM reports WHERE status='reviewed'").fetchone()[0],
    }
    db.close()

    return render_template("staff_dashboard.html", reports=reports_list, stats=stats)



@app.route("/staff/new_patient", methods=["GET", "POST"])
def new_patient():
    """
    GET  /staff/new_patient — render the patient registration form.
    POST /staff/new_patient — process the form:
      1. Save the uploaded X-ray to TEMP_FOLDER.
      2. Run AI prediction via handler.predict().
      3. If valid X-ray: insert patient + report into DB,
         move file to UPLOAD_FOLDER, redirect to case_preview.
      4. If invalid image: flash error and re-render the form.

    Form fields: name, age, gender, phone, notes, file, skip_validation.
    """
    if "user_id" not in session or session["role"] != "staff":
        return redirect(url_for("login"))

    if request.method == "POST":
        name   = request.form.get("name")
        age    = request.form.get("age")
        gender = request.form.get("gender")
        phone  = request.form.get("phone")
        height = request.form.get("height")
        weight = request.form.get("weight")
        notes  = request.form.get("notes")
        file   = request.files.get("file")

        if not file or file.filename == "":
            flash("Please upload an X-ray image.", "error")
            return redirect(request.url)

        
        filename  = f"{int(time.time())}_{secure_filename(file.filename)}"
        temp_path = os.path.join(TEMP_FOLDER, filename)
        file.save(temp_path)

        try:
            
            cnn_enabled = request.form.get("cnn_enabled") == "on"
            results = handler.predict(temp_path, skip_validation=not cnn_enabled)

            if not results.get("is_xray"):
              
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                flash(results.get("error", "Invalid image — not a knee X-ray."), "error")
                return redirect(request.url)

            
            db     = get_db_connection()
            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO patients (name, age, gender, phone, height, weight, created_by) VALUES (?,?,?,?,?,?,?)",
                (name, age, gender, phone, height, weight, session["user_id"])
            )
            patient_id = cursor.lastrowid

           
            final_path = os.path.join(UPLOAD_FOLDER, filename)
            os.rename(temp_path, final_path)

           
            grade_num = results.get("grade", 0)

          
            cursor.execute("""
                INSERT INTO reports
                    (patient_id, xray_path, heatmap_path, overlay_path,
                     ai_grade, ai_confidence, staff_notes, status)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                patient_id,
                f"static/uploads/{filename}",
                results.get("heatmap_path"),
                results.get("overlay_path"),
                grade_num,
                results.get("confidence"),
                notes,
                "draft"
            ))
            report_id = cursor.lastrowid
            db.commit()
            db.close()

            return redirect(url_for("case_preview", report_id=report_id))

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            flash(f"Error processing case: {e}", "error")
            return redirect(request.url)

    return render_template("new_patient.html")



@app.route("/staff/case_preview/<int:report_id>")
def case_preview(report_id):
    """
    GET /staff/case_preview/<id>
    Fetches the report and linked patient from the DB.
    Passes {results, report_id, notes, patient} to case_preview.html
    so staff can see the X-ray images and AI grade before sending to doctor.
    """
    if "user_id" not in session:
        return redirect(url_for("login"))

    db      = get_db_connection()
    report  = db.execute("SELECT * FROM reports  WHERE id = ?", (report_id,)).fetchone()
    patient = db.execute("SELECT * FROM patients WHERE id = ?", (report["patient_id"],)).fetchone()
    db.close()

    results = {
        "xray_path":    report["xray_path"],
        "overlay_path": report["overlay_path"],
        "prediction":   f"Grade {report['ai_grade']}",
        "confidence":   report["ai_confidence"],
    }

    bmi, bmi_cat, healthy_weight = calculate_bmi(patient["height"], patient["weight"])

    return render_template(
        "case_preview.html",
        results=results, report_id=report_id,
        notes=report["staff_notes"], patient=patient,
        bmi=bmi, bmi_cat=bmi_cat, healthy_weight=healthy_weight
    )



@app.route("/staff/send_to_doctor/<int:report_id>", methods=["POST"])
def send_to_doctor(report_id):
    """
    POST /staff/send_to_doctor/<id>
    Updates the patient details (if changed from the Case Preview screen),
    and changes the report status from 'draft' -> 'pending' so the
    doctor can see it in their queue.
    Redirects back to /dashboard after update.
    """
    patient_name = request.form.get("patient_name")
    patient_phone = request.form.get("patient_phone")
    patient_age = request.form.get("patient_age")
    patient_gender = request.form.get("patient_gender")
    patient_height = request.form.get("patient_height")
    patient_weight = request.form.get("patient_weight")
    notes = request.form.get("notes")

    db = get_db_connection()
    
   
    report = db.execute("SELECT patient_id FROM reports WHERE id=?", (report_id,)).fetchone()
    if report and report["patient_id"]:
        db.execute("""
            UPDATE patients 
            SET name=?, phone=?, age=?, gender=?, height=?, weight=?
            WHERE id=?
        """, (patient_name, patient_phone, patient_age, patient_gender, patient_height, patient_weight, report['patient_id']))

    if notes is not None:
        db.execute("UPDATE reports SET status='pending', staff_notes=? WHERE id=?", (notes, report_id))
    else:
        db.execute("UPDATE reports SET status='pending' WHERE id=?", (report_id,))
    
    db.commit()
    db.close()
    flash("Case sent to doctor successfully.", "success")
    return redirect(url_for("dashboard"))


@app.route("/report/<int:report_id>")
def view_report(report_id):
    """
    GET /report/<id>
    Displays a detailed, A4-style report view intended for printing.
    Can be viewed by the patient or staff/doctors.
    """
    if "user_id" not in session and "patient_phone" not in session:
        return redirect(url_for("login"))

    db = get_db_connection()
    report  = db.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    if not report:
        db.close()
        flash("Report not found.", "error")
        return redirect(url_for('dashboard'))

    patient = db.execute("SELECT * FROM patients WHERE id = ?", (report["patient_id"],)).fetchone()
    

    if session.get("role") == "patient" and patient["phone"] != session.get("patient_phone"):
        db.close()
        flash("Not authorized to view this report.", "error")
        return redirect(url_for('patient_dashboard'))

    doctor = None

    doctor = db.execute("SELECT * FROM users WHERE role = 'doctor' LIMIT 1").fetchone()

    db.close()
    
  
    bmi, bmi_cat, healthy_weight = calculate_bmi(patient["height"], patient["weight"])
    tips = []
    
    grade = report["final_grade"] if report["final_grade"] is not None else report["ai_grade"]
    
    if bmi:
       
        if bmi_cat == "Underweight":
            tips.append(f"**Nutrition & BMI:** Your BMI ({bmi}) indicates you are underweight. A lack of muscle mass can decrease joint support. Consider a clinical nutrition plan to reach your target range ({healthy_weight}).")
            tips.append("**Joint Support:** Focus on resistance training to build the muscles around your knees, which absorb shock and stabilize the joint.")
        elif bmi_cat == "Overweight":
            tips.append(f"**Weight Management:** Your BMI ({bmi}) is in the overweight category. Every extra kilogram of body weight equates to roughly 4 kilograms of extra pressure on your knees. Reaching your healthy target ({healthy_weight}) can significantly reduce daily pain.")
        elif bmi_cat == "Obese":
            tips.append(f"**Weight Management:** An Obese BMI ({bmi}) is a primary risk factor for rapid progression of Osteoarthritis. Reaching a healthy weight ({healthy_weight}) should be your primary non-surgical goal. Consult a specialist for a safe reduction plan.")
        elif bmi_cat == "Normal":
            tips.append(f"**Weight Management:** Excellent job maintaining a healthy BMI ({bmi}). Remaining in your target weight range ({healthy_weight}) minimizes unnecessary mechanical stress on your knee cartilage.")

       
        if grade >= 2 and (bmi_cat == "Overweight" or bmi_cat == "Obese"):
            tips.append("**Activity Modification:** With active cartilage wear, avoid high-impact activities (like running on hard surfaces). Substitute with low-impact alternatives like swimming, cycling, or using an elliptical machine while focusing on weight reduction.")
        elif grade >= 2 and bmi_cat == "Normal":
            tips.append("**Activity Modification:** Since your weight is optimal, focus purely on joint preservation. Avoid repetitive deep knee bending or prolonged squatting. Prioritize activities like brisk walking or water aerobics.")
            
   
    if grade == 0:
        tips.append("**Prevention:** No signs of Osteoarthritis detected. Maintain your joint health through regular, varied exercise and stretching to ensure flexibility.")
    elif grade == 1:
        tips.append("**Early Intervention:** Doubtful/Early joint narrowing is present. Begin a targeted physical therapy routine focusing on strengthening your quadriceps and hamstrings to protect the knee.")
    elif grade == 2:
        tips.append("**Moderate Care:** Mild Osteoarthritis is present. Consider discussing supportive footwear, knee braces, or mild anti-inflammatory strategies with your doctor.")
    elif grade == 3:
        tips.append("**Advanced Management:** Moderate Osteoarthritis with visible cartilage loss. You may benefit from advanced pain management strategies, localized injections, or walking aids during flare-ups.")
    elif grade == 4:
        tips.append("**Severe OA Consultation:** Severe joint space reduction (bone-on-bone). If your daily mobility is significantly impaired, an orthopedic surgical consultation for joint replacement or correction may be the most viable next step.")
            
    insights = {
        "bmi": bmi,
        "bmi_cat": bmi_cat,
        "healthy_weight": healthy_weight,
        "tips": tips
    }

    return render_template("report_detail.html", report=report, patient=patient, doctor=doctor, insights=insights)



@app.route("/staff/delete_patient/<int:patient_id>", methods=["POST"])
def delete_patient(patient_id):
    """
    POST /staff/delete_patient/<id>
    Permanently deletes a patient and all associated data:
      - Removes xray_path, heatmap_path, overlay_path from disk.
      - Deletes the reports row(s) from DB.
      - Deletes the patient row from DB.
    Requires staff session. Redirects to /dashboard on completion.
    """
    if "user_id" not in session or session["role"] != "staff":
        return jsonify({"error": "Unauthorised"}), 403

    db = get_db_connection()
    try:
        reports = db.execute(
            "SELECT xray_path, heatmap_path, overlay_path FROM reports WHERE patient_id=?",
            (patient_id,)
        ).fetchall()


        for report in reports:
            for col in ("xray_path", "heatmap_path", "overlay_path"):
                rel_path = report[col]
                if rel_path:
                    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
                    if os.path.exists(abs_path):
                        try:
                            os.remove(abs_path)
                        except Exception as fe:
                            print(f"File delete warning: {fe}")

 
        db.execute("DELETE FROM reports  WHERE patient_id=?", (patient_id,))
        db.execute("DELETE FROM patients WHERE id=?",         (patient_id,))
        db.commit()
        flash("Patient and all associated records deleted.", "success")

    except Exception as e:
        db.rollback()
        flash(f"Deletion failed: {e}", "error")
    finally:
        db.close()

    return redirect(url_for("dashboard"))



@app.route("/staff/quick_check")
def quick_check():
    """
    GET /staff/quick_check — renders the Quick AI Check page.
    Staff-only.  No patient data needed — just upload and get a result.
    """
    if "user_id" not in session or session["role"] != "staff":
        return redirect(url_for("login"))
    return render_template("quick_check.html")


@app.route("/api/quick_predict", methods=["POST"])
def quick_predict():
    """
    POST /api/quick_predict
    Called by JavaScript (fetch) from the Quick Check page.
    Accepts a multipart file + optional 'skip_validation' flag.

    Steps:
      1. Save uploaded file to TEMP_FOLDER.
      2. Call handler.predict() to run binary validation + OA grading.
      3. If valid X-ray: move file to UPLOAD_FOLDER and return JSON results.
      4. On error: return {error: ...} JSON with status 500.

    Returns JSON: {is_xray, prediction, confidence, xray_path,
                   overlay_path, heatmap_path, ...}
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file            = request.files["file"]
    cnn_enabled     = request.form.get("cnn_enabled") == "on"

    filename  = f"{int(time.time())}_{secure_filename(file.filename)}"
    temp_path = os.path.join(TEMP_FOLDER, filename)
    file.save(temp_path)

    try:
        results = handler.predict(temp_path, skip_validation=not cnn_enabled)

        if results.get("is_xray"):
            final_path = os.path.join(UPLOAD_FOLDER, filename)
            os.rename(temp_path, final_path)
        else:
        
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return jsonify(results)

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


@app.route("/api/quick_save", methods=["POST"])
def quick_save():
    """
    POST /api/quick_save
    Saves a Quick Check result to the DB as an anonymous (draft) record.
    Called by JavaScript when the user clicks 'Save to Database'.

    Expects JSON body with keys:
      prediction, confidence, xray_path, heatmap_path, overlay_path

    Creates a placeholder patient named 'Quick Test - <grade>'
    and a corresponding draft report row. Returns {success: true} on OK.
    """
    if "user_id" not in session:
        return jsonify({"error": "Unauthorised"}), 401

    data   = request.json
    db     = get_db_connection()
    cursor = db.cursor()

    try:
        cursor.execute(
            "INSERT INTO patients (name, age, gender, phone, created_by) VALUES (?,?,?,?,?)",
            (f"Quick Test — {data['prediction']}", 0, "N/A", "N/A", session["user_id"])
        )
        patient_id = cursor.lastrowid

        cursor.execute("""
            INSERT INTO reports
                (patient_id, xray_path, heatmap_path, overlay_path,
                 ai_grade, ai_confidence, status)
            VALUES (?,?,?,?,?,?,?)
        """, (
            patient_id,
            data.get("xray_path"),
            data.get("heatmap_path"),
            data.get("overlay_path"),
            data.get("grade", 0),
            data.get("confidence"),
            "draft"
        ))
        db.commit()
        return jsonify({"success": True})

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()



@app.route("/static/uploads/<path:filename>")
def serve_uploads(filename):
    """
    GET /static/uploads/<filename>
    Serves uploaded X-ray images and heatmap overlays from UPLOAD_FOLDER.
    Flask's default static handler does not cover dynamic upload paths,
    so this explicit route is needed for templates that build image URLs.
    """
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), "static", "uploads"),
        filename
    )




@app.route("/patient/dashboard")
def patient_dashboard():
    """
    GET /patient/dashboard
    Shows the patient a list of their reports.
    Requires patient session.
    """
    if "patient_phone" not in session or session.get("role") != "patient":
        return redirect(url_for("login"))

    db = get_db_connection()
    

    reports = db.execute("""
        SELECT r.*, p.name AS patient_name, p.age, p.gender, p.height, p.weight
        FROM   reports r
        JOIN   patients p ON r.patient_id = p.id
        WHERE  p.phone = ?
        ORDER  BY r.created_at DESC
    """, (session["patient_phone"],)).fetchall()
    
  
    reports_list = []
    for r in reports:
        r_dict = dict(r)
        bmi, bmi_cat, healthy_weight = calculate_bmi(r["height"], r["weight"])
        r_dict["bmi"] = bmi
        r_dict["bmi_cat"] = bmi_cat
        r_dict["healthy_weight"] = healthy_weight
        reports_list.append(r_dict)
    
    db.close()

    return render_template("patient_dashboard.html", reports=reports_list)




@app.route("/doctor/pending")
def pending_cases():
    """
    GET /doctor/pending
    Shows the doctor a list of reports with status='pending'
    (cases that staff has sent for review).
    Reuses staff_dashboard.html with a filtered reports list.
    """
    if "user_id" not in session or session["role"] != "doctor":
        return redirect(url_for("login"))

    db = get_db_connection()
    reports = db.execute("""
        SELECT r.*, p.name AS patient_name, p.age, p.height, p.weight
        FROM   reports r
        JOIN   patients p ON r.patient_id = p.id
        WHERE  r.status IN ('pending', 'reviewed')
        ORDER  BY r.created_at ASC
    """).fetchall()
    
    reports_list = []
    for r in reports:
        r_dict = dict(r)
        bmi, bmi_cat, healthy_weight = calculate_bmi(r["height"], r["weight"])
        r_dict["bmi"] = bmi
        r_dict["bmi_cat"] = bmi_cat
        reports_list.append(r_dict)
    
    db.close()

    return render_template(
        "staff_dashboard.html",
        reports=reports_list,
        stats={"total_patients": 0, "pending_review": sum(1 for r in reports if r["status"] == "pending"),
               "approved_cases": sum(1 for r in reports if r["status"] == "reviewed")}
    )

@app.route("/doctor/review/<int:report_id>", methods=["GET", "POST"])
def review_case(report_id):
    """
    GET  /doctor/review/<id> — render the case review form.
    POST /doctor/review/<id> — process the doctor's decision:
      - 'approve': set final_grade, doctor_notes, status='reviewed'.
      - 'reupload': revert status to 'draft' with a flag for staff.
    Renders case_review.html with {report, patient}.
    """
    if "user_id" not in session or session["role"] != "doctor":
        return redirect(url_for("login"))

    db = get_db_connection()

    if request.method == "POST":
        final_grade  = request.form.get("final_grade")
        doctor_notes = request.form.get("doctor_notes")
        action       = request.form.get("action")

        if action == "approve":
            db.execute("""
                UPDATE reports
                SET    final_grade = ?, doctor_notes = ?, status = 'reviewed'
                WHERE  id = ?
            """, (final_grade, doctor_notes, report_id))
            db.commit()
            flash("Report finalised and visible to patient.", "success")

        elif action == "reupload":
            db.execute("UPDATE reports SET status='draft' WHERE id=?", (report_id,))
            db.commit()
            flash("Re-upload request sent back to staff.", "success")

        db.close()
        return redirect(url_for("pending_cases"))

    report  = db.execute("SELECT * FROM reports  WHERE id=?", (report_id,)).fetchone()
    patient = db.execute("SELECT * FROM patients WHERE id=?", (report["patient_id"],)).fetchone()
    db.close()

    bmi, bmi_cat, healthy_weight = calculate_bmi(patient["height"], patient["weight"])

    return render_template("case_review.html", report=report, patient=patient, bmi=bmi, bmi_cat=bmi_cat, healthy_weight=healthy_weight)




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
