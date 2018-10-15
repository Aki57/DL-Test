### 5 Caffe入门

* 5-1-1.py：读取lmdb数据库的方法
* 5-1-2.py：读取leveldb数据库的方法
* 5-1-3.py：直接读取图像的方法
* 5-1-4.py：caffe的MNist案例的solver.prototxt内容
* 5-1-5.py：通过编码生成caffe中LeNet的Net配置文件
* 5-1-6.py：模型中间过结果的可视化
* 5-9文件夹中包含了5-9章所用到的文件：
  * caffe.proto：加入caffe.proto的内容
  * center_loss_layer.hpp：模型层头文件
  * center_loss_layer.cpp：模型层的实现文件
  * deploy.prototxt：加入center loss前的net部署文件
  * deploy_center_loss.prototxt：加入center loss后的net部署文件
  * predict.py：保存center_loss层约束的模型层输出的代码
  * solver.prototxt：加入center loss前的solver配置文件
  * solver_center_loss.prototxt：加入center loss后的solver配置文件
  * train.prototxt：加入center loss前的模型训练文件
  * train_center_loss.prototxt：加入center loss后的模型训练文件
  * visualize.py：可视化模型中间层的代码