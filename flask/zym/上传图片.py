import os
import random
from flask import Flask, request, jsonify
from datetime import datetime
from werkzeug.utils import secure_filename
from gevent import pywsgi

# 获取当前位置的绝对路径
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)


# 上传图片的接口
@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("files")
    # 获取安全的文件名
    filename = secure_filename(f.filename)
    print(filename, "------------")
    # 生成随机数
    random_num = random.randint(0, 100)
    # 获取文件的后缀
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(random_num) + "." + filename.rsplit('.', 1)[1]
    if not os.path.exists(filename):
        os.makedirs(filename, 755)
    file_path = basedir + "/static/file/" + filename
    f.save(file_path)
    # 返回前端可调用的一个链接
    # 可以配置成对应的外网访问的链接
    my_host = "http://127.0.0.1:8080"
    new_path_file = my_host + "/static/file/" + filename
    data = {"msg": "success", "imageURL": new_path_file}

    payload = jsonify(data)
    return payload, 200


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('127.0.0.1', 8080), app)
    server.serve_forever()
