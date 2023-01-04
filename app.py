from flask import *
import database as db

app = Flask(__name__)

import frameSpliter as frameSpliter
import main_model as model
import os

@app.route('/upload')
def main():  
    return render_template("upload.html")  
  
@app.route('/uploaded', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']
        is_video = str(f.filename.split(".")[1]).lower() == "mp4"
        print(is_video)
        f.save("static/files/og/" + f.filename)
        #db.update_data("make", "model a", "กข 0000", "white", 'files/' + f.filename)
        return render_template("acknowledgement.html", name = f.filename, is_video=is_video)

@app.route('/processing', methods = ['POST'])
def processing():
    if request.method == 'POST':
        filename = request.form.get("filename")
        print(filename)
        if str(filename.split(".")[1]).lower() == "mp4":
            frameSpliter.framecapture("static/files/og/" + filename, "static/files/frames/" + filename.split(".")[0] + "/" )
        else:
            frameSpliter.imgcap("static/files/og/" + filename, "static/files/frames/" + filename.split(".")[0] + "/" )
        #for img in os.listdir("static/files/frames/" + filename.split(".")[0] + "/"):
        model.run("static/files/frames/" + filename.split(".")[0] + "/", "static/files/car/" + filename.split(".")[0] + "/", "static/files/license/" + filename.split(".")[0] + "/")
        return redirect("/", code=302)

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
    app.run(debug=True, port=8788)