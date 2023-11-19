from flask import Flask,request
from werkzeug.utils import redirect

app=Flask(__name__)#实例化一个flask对象app

@app.route("/")#路由
def hello_world():
    return "hello world"

@app.route("/hey/<float:username>")
def hey_yingong(username):
    return "hey %s" %(username+username)

@app.route("/baidu")
def baidu():
    return redirect("https://www.bilibili.com")

@app.route("/test/my/first",methods=["POST"])
def first_post():
    my_json=request.get_json()
    print(my_json)

app.run(host="0.0.0.0")#0.0.0.0任何主机都可以访问网页。 若不加，只有本机可以访问
