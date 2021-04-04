### Object-Detection-Yolo
For object classification, I used YOLOv3. The neural network divides the image into regions and predicts bounding boxes and probabilities for each region. The network only needs to view the image one time, and then the bounding boxes are weighted by the predictive probabilities and we set thresholds for number of object detection. As I will need some file that can describe object names(file-coco.names) by seeing it in video. I also need the config and weight files of pretrained-model into the Darknet architecture.(yolov3.cfg, yolov3.weights)

For video classification, I needed some files contaning all names that can tell all possible names for any object seen in video.
(bvlc_googlenet.caffemodel, bvlc_googlenet.prototxt,synset_word)
