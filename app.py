from flask import Flask, request, jsonify, session

app = Flask(__name__)
app.secret_key="asass"


@app.route('/index', methods=["GET"])
def index():
    return 'Hello Flask!'


@app.route("/hey/<username>")
def hey_lu(username):
    return "hey %s" % username


@app.route("/my/<int:number>")
def my_number(number):
    return "my %s" % (number + number)


@app.route("/test/my/first", methods=["POST"])
def first_post():
    try:
        my_json = request.get_json()
        print(my_json)
        get_name = my_json.get("name")
        get_age = my_json.get("age")
        if not all([get_name, get_age]):
            return jsonify(msg="缺少参数")

        get_age += 10
        return jsonify(name=get_name, age=get_age)
    except Exception as e:
        print(e)
        return jsonify(msg="出错了，请查看是否正确访问")


@app.route("/try/login", methods=["POST"])
def login():
    """
    账号 username asd123
    密码 password asd
    :return:
    """
    get_data=request.get_json()
    username=get_data.get("username")
    password=get_data.get("password")

    if not all([username, password]):
        return jsonify(msg="参数不完整")

    if username == "asd123" and password == "asd":
        # 如果验证通过
        session["username"] = username
        return jsonify(msg="登录成功")
    else:
        return jsonify(msg="账号密码错误")


@app.route("/session", methods=["GET"])
def check_session():
    pass


@app.route("/try/logout", methods=["POST"])
def logout():
    pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
