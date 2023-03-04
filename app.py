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
    return render_template("detail.html", table = list(db.get_list_data({"status": "success"})), show = db.get_byid(id))

@app.route("/edit/<id>", methods = ['POST'])
def edit_back(id):
    if request.method == 'POST':
        body = request.form

        print(body)
        db.update_human_edit(id, body["make"], body["model"], body["color"], body["plate"])
        if db.get_lasted({'status': 'pending'}):
            return redirect(f"/edit/{db.get_lasted({'status': 'pending'})['_id']}")
        return redirect("/")

@app.route("/markbad/<id>", methods = ['POST'])
def mark_bad(id):
    db.set_bad_cond(id)
    if db.get_lasted({'status': 'pending'}):
        return redirect(f"/edit/{db.get_lasted({'status': 'pending'})['_id']}")
    return redirect("/")

@app.route("/edit/<id>")
def edit_ui(id):
    return render_template("edit.html", table = list(db.get_list_data({'status': 'pending'})), show = db.get_byid(id))

@app.route("/pending")
def pending():
    return redirect("/pending/page/1/limit/10")

@app.route("/pending/page/<page>/limit/<limit>")
def pending_list(page, limit):
    if (int(page) < 1):
        return redirect("/pending/page/1/limit/10")
    print(page, limit)
    pages, data = db.get_list_data_pagination(query={'status': 'pending'}, page=int(page)-1, limit=int(limit))
    if (int(page) > pages) and (pages != 0):
        return redirect(f"/pending/page/{pages}/limit/10")
    print(pages, data)
    return render_template("listing.html", table = list(data), pages = int(pages), current = int(page), limit = int(limit), root = "pending", search_action = "pending/search", keyword="", search = True)

@app.route("/pending/search", methods = ['POST'])
def pending_on_search():
    body = request.form
    search_type = str(body["type"])
    keyword = str(body["keyword"])
    return redirect(f"/pending/type/{search_type}/keyword/{keyword}/page/1/limit/10")

@app.route("/pending/type/<type>/keyword/<keyword>/page/<page>/limit/<limit>")
def pending_list_search(page, limit, type, keyword):
    if (int(page) < 1):
        return redirect(f"/pending/type/{type}/keyword/{keyword}/page/1/limit/10")
    if not keyword or not type:
        return redirect("/pending/page/1/limit/10")
    print(page, limit)
    pages, data = db.get_list_data_pagination(query={'status': 'pending', f"{type}": keyword}, page=int(page)-1, limit=int(limit))
    if (int(page) > pages) and (pages != 0) :
        return redirect(f"/pending/page/{pages}/limit/10")
    print(pages, data)
    return render_template("listing.html", table = list(data), pages = int(pages), current = int(page), limit = int(limit), root = f"pending/type/{type}/keyword/{keyword}", search_action = "pending/search", keyword=keyword, search = True)

@app.route("/success")
def success():
    return redirect("/success/page/1/limit/10")

@app.route("/success/page/<page>/limit/<limit>")
def success_list(page, limit):
    if (int(page) < 1):
        return redirect("/success/page/1/limit/10")
    print(page, limit)
    pages, data = db.get_list_data_pagination(query={'status': 'success'}, page=int(page)-1, limit=int(limit))
    if (int(page) > pages) and (pages != 0) :
        return redirect(f"/success/page/{pages}/limit/10")
    print(pages, data)
    return render_template("listing.html", table = list(data), pages = int(pages), current = int(page), limit = int(limit), root = "success", search_action = "success/search", keyword="", search = True)

@app.route("/success/search", methods = ['POST'])
def success_on_search():
    body = request.form
    search_type = str(body["type"])
    keyword = str(body["keyword"])
    return redirect(f"/success/type/{search_type}/keyword/{keyword}/page/1/limit/10")

@app.route("/success/type/<type>/keyword/<keyword>/page/<page>/limit/<limit>")
def success_list_search(page, limit, type, keyword):
    if (int(page) < 1):
        return redirect(f"/success/type/{type}/keyword/{keyword}/page/1/limit/10")
    if not keyword or not type :
        return redirect("/success/page/1/limit/10")
    print(page, limit)
    pages, data = db.get_list_data_pagination(query={'status': 'success', f"real_{type}": keyword}, page=int(page)-1, limit=int(limit))
    if (int(page) > pages) and (pages != 0) :
        return redirect(f"/success/page/{pages}/limit/10")
    print(pages, data)
    return render_template("listing.html", table = list(data), pages = int(pages), current = int(page), limit = int(limit), root = f"success/type/{type}/keyword/{keyword}", search_action = "success/search", keyword=keyword, search = True)

@app.route("/bad")
def bad():
    return redirect("/bad/page/1/limit/10")

@app.route("/bad/page/<page>/limit/<limit>")
def bad_list(page, limit):
    if (int(page) < 1):
        return redirect("/bad/page/1/limit/10")
    print(page, limit)
    pages, data = db.get_list_data_pagination(query={'status': 'bad_cond'}, page=int(page)-1, limit=int(limit))
    if (int(page) > pages) and (pages != 0) :
        return redirect(f"/bad/page/{pages}/limit/10")
    print(pages, data)
    return render_template("listing.html", table = list(data), pages = int(pages), current = int(page), limit = int(limit), root = "bad", search = False)

@app.route('/')
def home():
    return render_template("home.html", table = list(db.get_list_data({'status': {"$ne": "bad_cond"}}).limit(3)), show = db.get_lasted({'status': 'success'}))

if __name__ == '__main__':
    app.run(debug=True, port=8080)