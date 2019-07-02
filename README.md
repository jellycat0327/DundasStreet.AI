# DundasStreet.AI
For the detection of bicycles, vehicles, pedestrians on Dundas street

Inspired by the City of London's cycling master plan and the cycling routes selection problem, we are aiming to build a deep learning based object detection system to analyze traffic volume on Dundas Street in London. 

**Datasets**
Our source data was obtained from Live London webcams [here](http://www.londonwebcams.ca/) that provide real-time snapshots updated every 30 seconds, and contain images from the east and west on the block between Wellington and Clarence Streets. 
The default image size is 360 x 245. After crawling the images from the website, we applied an online tool called VGG Image Annotator[here] {http://goo.gl/Kb39RK} to quickly annotate the key objects with boxes and categories for 
the actual downloaded images. Then, we made the CSV datasets.

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

A full example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
```

This defines a dataset with 3 images.
`img_001.jpg` contains a cow.
`img_002.jpg` contains a cat and a bird.
`img_003.jpg` contains no interesting objects/animals.


### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```
**Code**
This repository contains all the necessary resources to train our neural networks on the  datasets provided. For the keras-retinanet, you can
see more details on [RetinaNet](https://github.com/fizyr/keras-retinanet).

# Training
python keras_retinanet/bin/train.py csv /path/to/CSV/annotations.csv /path/to/CSV/classes.csv

# Evaluating
python keras_retinanet/bin/evaluate.py csv {data-set path} {model_path}

# Testing
An example of testing the network can be seen in [this Notebook](https://github.com/jellycat0327/DundasStreet.AI/blob/master/examples/ResNet50RetinaNet.ipynb).
In general, inference of the network works as follows:

```python
boxes, scores, labels = model.predict_on_batch(inputs)
```




