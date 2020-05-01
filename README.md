# EdgeDetection
在Coherent Line Drawing的代码上做了一点小改动
## 改动
- ETF.cpp line 248 and 249: 添加了*abs(angle)，按照论文应该如此。
- 添加了opencv cv::Mat与其自定义类的转换。
- fdog.cpp line 300：添加了非极大值抑制，写了两种。
- 使之能运行的杂项改动。
## 使用
- 函数myFDoG的最后一个参数nms用来设置邻域大小。nms<=0时没有非极大值抑制。
## TODO
- 看代码貌似除了*abs(angle)还是有点小问题。
- 自适应的非极大值抑制。
- 复现的opencv版本太慢了，或许用Eigen好一些，或许像现在这样二维数组也挺好。
- 懒，烦请大佬代劳以上。