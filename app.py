from flask import *
import database as db

from detect_flow import detect_flow

app = Flask(__name__)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


  
@app.route('/uploaded', methods = ['POST'])  
def process():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f"static/files/og/{f.filename}")
        detect_flow(f"./static/files/og/{f.filename}")
        return redirect("/")


@app.route('/show/<id>')
def show(id):
    if db.get_byid(id)["status"] == 'pending':
        return redirect(f"/edit/{id}")
    return render_template("home.html", table = list(db.get_list_data()), show = db.get_byid(id))

@app.route("/edit/<id>", methods = ['POST'])
def edit_back(id):
    if request.method == 'POST':
        body = request.form

        print(body)
        db.update_human_edit(id, body["make"], body["model"], body["color"], body["plate"])
        return redirect(f"/edit/{db.get_lasted({'status': 'pending'})['_id']}")

@app.route("/markbad/<id>", methods = ['POST'])
def mark_bad(id):
    db.set_bad_cond(id)
    return redirect(f"/edit/{db.get_lasted({'status': 'pending'})['_id']}")

@app.route("/edit/<id>")
def edit_ui(id):
    return render_template("edit.html", table = list(db.get_list_data({'status': 'pending'})), show = db.get_byid(id))

@app.route('/')
def home():
    return render_template("home.html", table = list(db.get_list_data()), show = db.get_lasted())
def table_event(id):
    print(id)
if __name__ == '__main__':
    app.run(debug=True, port=8080)