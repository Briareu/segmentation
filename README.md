# segmentation
# 环境
* tensorflow-gpu == 1.13.1
* keras == 2.1.5
# 训练步骤
* 准备训练集，放入dataset2中（jpg下存入原图像，png为mask），以及train.txt文件；
* 将预训练模型放入model下；
* 运行train.py
# 预测步骤
* 将待预测图片放入img中；
* 运行predict.py
# 代码结构
* dataset2
    * jpg
    * png
* img
* img_out
* logs
* nets
    * Xception.py
    * deeplab.py
* predict.py
* test.py
* train.py
# 模块说明
* train.py --------- 训练文件
* predict.py --------- 预测文件
* test.py --------- 用于测试网络
