# Fine-tuned [DeepLab V3+](https://github.com/tensorflow/models/tree/master/research/deeplab) on [CDnet2014](http://changedetection.net/)

This folder contains a demo about how to use our [fine-tuned version](./Models/frozen_inference_graph.pd) of the [DeepLab V3+](https://github.com/tensorflow/models/tree/master/research/deeplab) model on [CDnet2014](http://changedetection.net/). Refer to the jupyter notebook [DeepLab_Demo_cdnet_model](./DeepLab_Demo_cdnet_model.ipynb) for details.

If you successfully run the demo, you get the following result on a demo image.
![illustration](./illustration.png)

Logits of the semantic segmentation are saved at [this folder](./Logits).
![Logits](./Logits.png)

*Since the fine-tuning process consists of countless trivil modifications of the original code and would cause bugs on other machines, we do not publish our training code in the current stage.*