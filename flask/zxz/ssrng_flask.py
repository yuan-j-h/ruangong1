# # from tkinter import Image
# # from gevent import pywsgi
# # from flask import Flask, request
# #
# # app = Flask(__name__)
# #
# # ALLOWED_EXTENSIONS = {'jpg'}
# #
# # def allowed_file(filename):
# #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# #
# # @app.route('/upload', methods=['POST'])
# # def upload():
# #     # 从请求中获取图片数据
# #     image_data = request.files['image']
# #
# #     # 将二进制数据转换为图片对象
# #     image = Image.open(image_data.stream)
# #
# #     # 进行图片处理或保存等操作
# #     # 这里我们仅仅将图片保存到本地
# #     image.save('uploaded_image.jpg')
# #
# #     return 'Image uploaded successfully!'
# #
# #
# # # if __name__ == '__main__':
# # #     app.run()
# #
# # server = pywsgi.WSGIServer(('0.0.0.0', 12345), app)
# # server.serve_forever()
# #
# from datetime import datetime
# from flask import Flask, request, jsonify
# import os
# import random
#
# # 获取当前位置的绝对路径
# basedir = os.path.abspath(os.path.dirname(__file__))
# print('basedir:' + basedir)
# app = Flask(__name__)
#
#
# # 上传图片接口
# @app.route("/upload", methods=["POST"])
# def upload():
#     print('12378973907390')
#     f = request.files.get('file')
#     print('dsjkdsfjkhfsidhfisd')
#     random_num = random.randint(0, 100)
#     print('sdfsdsfdf55551212')
#     # 获取文件的后缀名
#     filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(random_num) + "." + f.filename.rsplit('.', 1)[1].lower()
#     print('12222222222222:', filename)
#     file_path = basedir + "/static/file/" + filename
#     #print('12222222222222:', file_path)
#     f.save(file_path)
#
#     # 配置成对应的外网访问的链接
#     my_host = "http://127.0.0.1:5000"
#     new_path_file = my_host + "/static/file/" + filename
#     data = {"msg": "success", "url": new_path_file}
#
#     payload = jsonify(data)
#     return payload, 200
#
#
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000)
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
    f = request.files.get("file")
    # 获取安全的文件名
    filename = secure_filename(f.filename)
    print(filename, "------------")
    # 生成随机数
    random_num = random.randint(0, 100)
    # 获取文件的后缀
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(random_num) + "." + filename.rsplit('.', 1)[1]
    # if not os.path.exists(filename):
    #     os.makedirs(filename, 755)
    file_path = basedir + "/static/file/" + filename
    f.save(file_path)
    # 返回前端可调用的一个链接
    # 可以配置成对应的外网访问的链接
    my_host = "http://127.0.0.1:5000"
    new_path_file = my_host + "/static/file/" + filename
    data = {"msg": "success", "imageURL": new_path_file}

    payload = jsonify(data)
    return payload, 200


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('127.0.0.1', 5000), app)
    server.serve_forever()

