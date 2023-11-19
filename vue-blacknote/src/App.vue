<template>
  <section class="index_content">
    <div class="content_header" style="clear: both; overflow: hidden;">
      <div class="function_title">
        <img src="./images/function_title_word.svg" alt="CleverPDF Word" style="margin-left:0;" />
        <h1 style="margin-left:10px;" id="wtp_title">高光谱图像分析</h1>
        <h2 class="function_title_promit" id="wtp_promit">2023软件工程作业</h2>
      </div>
    </div>
    <div class="function_box" id="choosefile">
      <div class="function_choosefile" style="border-bottom: 0;">
        <input type="hidden" id="id" name="id" value="20170902001">
        <div class="function_choosefile_input">
          <div class="canvas_container" id="container">
            <div class="canvas_box">
              <canvas id="the-canvas" class="canvas_img" style="width:112px;height:145px;"></canvas>
              <div class="canvas_title"></div>
              <a class="canvas_delete"><img src="./images/canvas_delete.png" alt="canvas delete" /></a>
            </div>
          </div>
          <div class="input_box" id="chooseInput" v-show="!showConversionButton">
            <span><label for="openImage">选择图片</label></span>
            <input type="file" id="openImage" accept=".hdr, .bil,.tiff" name="files" @change="handleFileChange" />
          </div>
          <div class="input_box" id="startConversion" v-show="showConversionButton">
            <button type="button" class="input_box_conversion" @click="changeImage">开始分类</button>
          </div>
        </div>
        <div class="function_choosefile_font" id="wtp_choosefile">或者把文件拖拽到此区域</div>
      </div>
    </div>
  </section>
  <!-- 显示分类之后的图片 -->
  <img v-if="imageURL" :src="imageURL" alt="分类的结果" style="max-width: 100%;" />
  <p>imageURL = {{ imageURL }}</p>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      showConversionButton: false,
      uploadedFile: null,
      imageURL: null,
    };
  },
  methods: {
    handleFileChange(event) {
      const fileInput = event.target;
      const file = fileInput.files[0];
      if (file) {
        const fileName = file.name;
        const fileSize = file.size;
        console.log(`选择的文件: ${fileName}, 大小: ${fileSize} 字节`);
        this.uploadFile(file);
        this.showConversionButton = true;
      }
    },
    //上传文件
    uploadFile(file) {
      const formData = new FormData();
      formData.append('files', file);
      // 使用 Axios 将文件上传到服务器
      axios.post('http://127.0.0.1:8080/upload', formData)
        .then(response => {
          console.log('文件上传成功:', response.data);
          this.showConversionButton = true;
        })
        .catch(error => {
          console.error('上传文件时发生错误:', error);
          // 处理错误
        });
    },
    //函数有问题这里
    changeImage() {
      const url = 'http://127.0.0.1:8080/change';
      axios.get(url, { responseType: 'blob' })
        .then(response => {
          console.log('Response:', response);
          const imageURL = URL.createObjectURL(response.data);
          console.log('Image URL:', imageURL);
          this.imageURL = imageURL;
          console.log('this.imageURL:', this.imageURL);
          URL.revokeObjectURL(imageURL);
        })
        .catch(error => {
          console.error('加载图片时发生错误:', error);
        });
    },
  },
};
</script>
