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
        detect_flow(f"static/files/og/{f.filename}")
        return redirect("/")


@app.route('/show/<id>')
def show(id):
    print(db.get_byid(id))
    return render_template("home.html", table = list(db.get_list_data()), show = db.get_byid(id))


@app.route('/')
def home():
    print(db.get_lasted())
    return render_template("home.html", table = list(db.get_list_data()), show = db.get_lasted())
def table_event(id):
    print(id)
if __name__ == '__main__':
    app.run(debug=True, port=3000)