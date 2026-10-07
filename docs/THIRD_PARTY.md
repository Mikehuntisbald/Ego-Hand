# 来源和使用说明

主要上游：

- [WiLoR](https://github.com/rolpotamias/WiLoR)：RGB 手部重建、视觉主干及手侧 helper。
- [RF-DETR](https://github.com/roboflow/rf-detr)：检测与实例分割。
- [Ultralytics](https://github.com/ultralytics/ultralytics)：YOLO 检测。
- [SAM2](https://github.com/facebookresearch/sam2)：历史实例分割和辅助 YOLO 带 mask 对照。
- [HOT3D](https://github.com/facebookresearch/hot3d) 与 [hand_tracking_toolkit](https://github.com/facebookresearch/hand_tracking_toolkit)：数据、标定与手几何。
- [MANO](https://mano.is.tue.mpg.de/)：WiLoR 前端手模型资产，按其许可获取。

多域训练涉及 HOT3D、EgoHands、SurgicalHands、CPPE-5 和 100DOH。100DOH 作者下载说明限定非商业研究，本次训练权重按科研实验发布；实际使用还需遵守其他上游与数据来源条款。

仓库没有附加一个统一宽松许可覆盖所有代码、数据、上游模型和派生权重。保留上游版权和许可要求；源码快照、派生 checkpoint 和各数据的适用条款分别判断。原始图像/视频、样本级 GT、MANO/WiLoR/SAM2 基础资产没有随本次导出上传。
