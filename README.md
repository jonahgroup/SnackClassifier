# Snack Classifier
**Snack Classifier** is a RESTful API app to perform image classifying operations.
Although Snack Classifier has been designed to work with Jonah's open source project
[Snack Watcher](https://github.com/jonahgroup/SnackWatcher) closely,
this service could be used independently as an image classification *Microservice* if needed.
By giving it an image URL, it will classify the image and response with it's class name.
The more that Snack Classifier trained, the better that it could answer correctly.

## How to Run
Snack Classifier requirements can be installed by,

```
pip install -r requirements.txt
```

Once the requirements are installed, Snack Classifier app is started with:

```
python manage.py runserver
```

**Hint** From this point onwards, we shall assume the Snack Classifier restful service will be serve from <http://classifier:5000/api>.
If you are serving the API differently, please modify the URL to your system specific location.

### Configuration
The default configuration of the system located in `configuration/environment.ini`

The configuration keys are,

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| system | PORT | 5000 | classifier RESTful api is running from this port
| classifier | IMAGE_FOLDER_PATH | static/images | serving static images location
| classifier | CLASSIFIER_TYPE  | bayes | there are 4 predefined classifying algorithm (knn, svm, bayes, tree)
| classifier | CLASSIFIER_TRAIN_PATH | static/trainer/train | trained images storage location


## RESTful API
The RESTful API will have the context of, <http://classifier:5000/api>

| API | HTTP | Description |
|-----|------|-------------|
| /classify/names | GET | return the list of classified category names
| /classify?image=<image_url> | GET | return the classified category of the given image_url
| /classify | POST | given a list of image URLs and return their classified category
| /train | POST | given a list of trained image URLs and save them into training folder
| /retrain | GET | trigger retraining from the existed training set

The following example section assumed that you have install `httpie` package for easy HTTP communication. You could easily find `curl` equivalent commands to `http` (the actual running program of `httpie` package).

### Classified Image Name

All training images are layout in `CLASSIFIER_TRAIN_PATH`.
For each folder in `CLASSIFIER_TRAIN_PATH`, it defines a category (aka class) of images, for example, the following layout will define 3 classes of objects that the training classifier will be recognized, namely, ['brownie', 'candy', 'cookie', 'package'] classes in this case.

```
train
|____brownie
| |____brownie_001.png
| |____brownie_002.png
|____candy
| |____candy_001.png
| |____candy_002.png
| |____candy_003.png
|____cookie
| |____cookie_001.png
| |____cookie_002.png
|____package
| |____package_001.png
```

Usually, the more training images available for each class, the better classifier training that will be result.
You can imagine the classifier can understand more variations for that
class of objects if you give it more sample images. Later, when the classifier see a similar image again (not exactly the same but similar by features extraction), it would recognize that the unknown image belongs to which class of objects.

After training, the classifier can report the list of classes that it has been trained on. For example, we can get the list of class names by

```bash
http GET http://classifier:5000/api/classify/names
```

The API response will be,

```json
HTTP/1.0 200 OK
Content-Length: 86
Content-Type: application/json
Date: Sat, 21 Nov 2015 02:52:55 GMT
Server: Werkzeug/0.9.4 Python/2.7.6

{
    "classes": [
        "brownie",
        "candy",
        "cookie",
        "package"
    ]
}
```

### Classify A Single Image

After the classifier has been trained, with the training images to a list of possible classes. We could use the API to classify a single image.

For example, getting classify to a single image URL, `http://snack-web:8000/static/images/snack-2015_11_19-08_19_38/blob_01.png`, captured by the Snack Watcher.

```bash
http GET http://classifier:5000/api/classify?image=http://snack-web:8000/static/images/snack-2015_11_19-08_19_38/blob_01.png
```

The response will be,

```json
HTTP/1.0 200 OK
Content-Length: 105
Content-Type: application/json
Date: Sat, 21 Nov 2015 02:42:04 GMT
Server: Werkzeug/0.9.4 Python/2.7.6

{
    "class": "cookie",
    "image": "http://snack-web:8000/static/images/snack-2015_11_19-08_19_38/blob_01.png"
}
```

### Classify A List of Images

For efficiency, we don't want to call the classifier repeatedly for many images. We could collect all images into a list.

For example, posting the a list of image URLs via JSON.

```bash
% http POST http://classifier:5000/api/classify images:='["http://snack-web:8 000/static/images/snack-2015_11_19-08_19_38/blob_01.png"]'
```

or

```bash
http POST http://classifier:5000/api/classify < images.json
```

For example, the `images.json` JSON file could look like,

```json
{
  "images": [
    "http://classifier:5000/static/images/snack-2015_11_19-08_19_38/blob_01.png"
  ]
}
```

The response JSON will be a list of classified images, noticed that for each given `image` URL in the list, it would create a new `class` field associated with the `image` field as the result.

```json
HTTP/1.0 200 OK
Content-Length: 119
Content-Type: application/json
Date: Sat, 21 Nov 2015 21:25:16 GMT
Server: Werkzeug/0.9.4 Python/2.7.6

{
    "images": [
        {
            "class": "cookie",
            "image": "http://classifier:5000/static/images/snack-2015_11_19-08_19_38/blob_01.png"
        }
    ]
}
```

### Training with a List of Classified Images

For example, posting the trained images and saved into training folder for future training

The following `http` command will post the following JSON images list,

```bash
http POST http://classifier:5000/api/train \
  images:='[{"class": "test", "image": "http://snack-web:8000/static/images/test1_001.png"}]'
```

The response will confirm the successful posting,

```json
HTTP/1.0 200 OK
Content-Length: 2
Content-Type: application/json
Date: Sun, 29 Nov 2015 02:43:00 GMT
Server: Werkzeug/0.9.4 Python/2.7.6

{
  "images": [
    {
      "class": "test",
      "image": "http://snack-web:8000/static/images/test1_001.png",
      "saved": "static/trainer/train/test/test1_001.png"
    }
  ]
}
```

On the response, you could confirm to find `static/trainer/train/test/test1_001.png` has been saved. `test1_001.png` will be classified as `test` class.

To further clarify this trained image posting API, a bigger example illustrated using a JSON file `train.json`

```json
{
    "images": [
        {
            "class": "candy",
            "image": "http://snack-web:8000/static/images/snack-2015_12_01-17_45_00/blob_02.png"
        },
        {
            "class": "candy",
            "image": "http://snack-web:8000/static/images/snack-2015_12_01-17_45_00/blob_03.png"
        },
        {
            "class": "candy",
            "image": "http://snack-web:8000/static/images/snack-2015_12_01-17_45_00/blob_04.png"
        }
    ]
}
```

We could send the `train.json` JSON file by,

```bash
http POST http://classifier:5000/api/train < train.json
```

The response would be like,

```json
HTTP/1.0 200 OK
Content-Length: 827
Content-Type: application/json
Date: Tue, 01 Dec 2015 18:12:47 GMT
Server: Werkzeug/0.10.4 Python/2.7.3

{
    "images": [
        {
            "class": "candy",
            "image": "http://snack-web:8000/static/images/snack-2015_12_01-17_45_00/blob_02.png",
            "saved": "static/trainer/train/candy/candy_022.png"
        },
        {
            "class": "candy",
            "image": "http://snack-web:8000/static/images/snack-2015_12_01-17_45_00/blob_03.png",
            "saved": "static/trainer/train/candy/candy_023.png"
        },
        {
            "class": "candy",
            "image": "http://snack-web:8000/static/images/snack-2015_12_01-17_45_00/blob_04.png",
            "saved": "static/trainer/train/candy/candy_024.png"
        }
    ]
}
```

The response will confirm each image successfully saved location according to the given class.

### Trigger Re-training of Existed Images

After we have accumulated more classified images, we would definitely want to trigger the trainer to learn from the expanded training set. We could trigger the retraining by,

```bash
http GET http://classifier:5000/api/retrain
```

The training may take a few minutes depending on how many images in the
training set. After it completed the training, it would response
with the list of trained classes to confirm.

```
HTTP/1.0 200 OK
Content-Length: 67
Content-Type: application/json
Date: Wed, 02 Dec 2015 12:55:56 GMT
Server: Werkzeug/0.9.4 Python/2.7.6

{
    "classes": [
        "brownie",
        "candy",
        "cookie",
        "package"
    ]
}
```

## Snack Classification using Python and SimpleCV
The [Snack Watcher](https://github.com/jonahgroup/SnackWatcher)
detects if any snack has been put on the snack table. After we obtained
the difference as an image "Blob", it is desirable to proceed recognizing
if that "Blob" looks like some snack. Even better, the program
can tell if that "Blob" is a particular type of snack, e.g. cookies or candies.

We have used the amazing easy to use Python computer vision framework `SimpleCV`.
SimpleCV is an OpenCV wrapper that make image processing, feature detection
and display very simple; at least, we don't need to invest great effort to
achieve some interesting computer vision results. For image classification, SimpleCV
is built on top of Orange. Orange implements a rich set of image extraction operators and
machine learning algorithms to support our classification tasks.
For setting up OpenCV, SimpleCV and Orange for the project, please refer this blog post
[Raspberry Pi 3 for Computer Vision](http://jonahgroup.github.io/SnackWatcher/Raspberry-Pi-3-for-Computer-Vision)
for instructions.

The best way to learn about Snack Classification, is to try it out.
The [Snack Classifier](https://github.com/jonahgroup/SnackClassifier) is an open source project
for snack classification. It comes with Python commandline scripts to help experimenting with
the system. Once you felt comfortable, Snack Classifier could be used as a *Microservice* to
support your image classification needs; extending the functionality of snack watching!

### Snack Trainer (Python Script)
This script has been inspired by this article,
[A fruit image classifier with Python and SimpleCV](http://jmgomez.me/a-fruit-image-classifier-with-python-and-simplecv/).
We have enhanced the article's code snippet to make an easy to use
Snack Training and Classifier programs.

`snack-trainer.py`

```bash
Usage: snack-trainer.py [options] -c <classes> -m <method>

Options:
  -h, --help            show this help message and exit
  -g, --debug           debugging mode
  -c CLASSES, --class=CLASSES
                        detect classes, comma seperated
  -a TRAIN_PATH, --train=TRAIN_PATH
                        training samples path
  -t TEST_PATH, --test=TEST_PATH
                        testing samples path
  -r RESULT_PATH, --result=RESULT_PATH
                        testing results path
  -s CLASSIFIER, --classifier=CLASSIFIER
                        using classifier (svm|tree|bayes|knn)
  -f FEATURE_PATH, --feature=FEATURE_PATH
                        save training features into file
  -e CLASSIFIER_FILE, --save=CLASSIFIER_FILE
                        save classifier into file
```

The default path containing all the training images is `train`,
the default path containing all the testing images is `test` and
the default path where the results are written is `result`. Of course, you
can change the directories with the corresponding options.

All image classes must be in separate directories.
The class type must also align to the directories.
For example, if we have 2 classes `cookie` and `other`, the directory layout should be like,

```
  train
  |____cookie
  | |____cookie_001.png
  | |____cookie_002.png
  | |____ ...
  |____other
  | |____other_001.png
  | |____other_002.png
  | |____ ...

  test
  |____cookie
  | |____test_001.png
  | |____test_002.png
  | |____ ...
  |____other
  | |____test_003.png
  | |____test_004.png
  | |____ ...
```

After we setup all the image directories, we can execute the `snack-trainer.py` by,

```bash
python snack-trainer.py -g -c "cookie,other" -s tree
```

In this example, we are classifying `cookie` and `other`. We specify to use `tree` classifier to classify the `test` images. The result annotated images will be saved into `result`.

In addition, the trained classifier parameters are saved into, default by the classifier name, e.g. `tree.dat`.
We can use this classifier file for further image classification by `snack-classify.py` after training.

### Snack Classify (Python Script)
If we are using `tree` classifier for training, the default trained classifier will be saved as `tree.dat` (Note: the `snack-trainer.py` has an option to control the classifier saved filename.)
Subsequently, we can use `tree.dat` to reconstruct the classifier for classifying other images. `snack-classfy.py` is the utility program to perform further image classifications.

`snack-classify.py`

```bash
Usage: snack-classify.py [options] -c <classifier_file> -i <image>

Options:
  -h, --help            show this help message and exit
  -g, --debug           debugging mode
  -c CLASSIFIER_FILE, --classifier=CLASSIFIER_FILE
                        load classifier from file
  -i IMAGE_FILE, --image=IMAGE_FILE
                        classify this image file
```

If we want to classify `something.png` image, we can execute,

```bash
python snack-classify.py -c tree.dat -i something.png
```

It would reply the classified name for the given image.

## SimpleCV Packages
To further our insight about image training and classification, following the Python code snippets to
understand how to use SimpleCV machine learning packages (built on Orange), achieving a simple and flexible
solution.

### SimpleCV Feature Extraction
In this section, we are playing with [SimpleCV Feature Package](http://simplecv.sourceforge.net/doc/SimpleCV.Features.html).

#### Edge Feature Extractor
Create a 1D edge length histogram and 1D edge angle histogram.

This method takes in an image, applies an edge detector, and calculates the length and direction of lines in the image.
(bins = the number of bins)

For example, we would like to do the edge histogram feature extraction from `cookie_001.png` image. The features are slotted into e.g. 5 bins. Then, we print out the feature field names and their values.

```python
edge = EdgeHistogramFeatureExtractor(5)
img = Image('cookie_001.png')

edge_vecs = edge.extract(img)
edge_fields = edge.getFieldNames()
for i in range(0, edge.getNumFields()):
    print edge_fields[i], '=', edge_vecs[i]
```

The result would be like,

```
Length0 = 1.80579216354
Length1 = 1.66950596252
Length2 = 0.834752981261
Length3 = 0.45996592845
Length4 = 0.229982964225
Angle0 = 0.0
Angle1 = 0.00384859186815
Angle2 = 0.00635017658245
Angle3 = 0.00369012043829
Angle4 = 0.0
```

#### Hue Feature Extractor
Create a Hue Histogram feature extractor. This feature extractor takes in an image, gets the hue channel, bins the number of pixels with a particular Hue, and returns the results.
(bins = the number of Hue bins)

For example, we would like to do the hue histogram feature extraction from `cookie_001.png` image. The features are slotted into e.g. 5 bins. Then, we print out the feature field names and their values.

```python
hue = HueHistogramFeatureExtractor(5)
img = Image('cookie_001.png')

hue_vecs = hue.extract(img)
hue_fields = hue.getFieldNames()
for i in range(0, hue.getNumFields()):
    print hue_fields[i], '=', hue_vecs[i]
```

The result would be like,

```
Hue0 = 0.0183470315515
Hue1 = 0.000665208703398
Hue2 = 0.000427097170245
Hue3 = 0.000168505712084
Hue4 = 0.0
```

#### Haar Feature Extractor
This is used generate Haar like features from an image. These Haar like features are used by a the classifiers of machine learning to help identify objects or things in the picture by their features, or in this case haar features.

For a more in-depth review of Haar Like features see: <http://en.wikipedia.org/wiki/Haar-like_features>

For example, we will use SimpleCV's `haar.txt` file for Haar like features extraction parameters. Then, we print out the feature field names and their values. Since it has a lot of features, the result display has been clipped.

```python
haar = HaarLikeFeatureExtractor(fname='haar.txt')
img = Image('cookie_001.png')

haar_vecs = haar.extract(img)
haar_fields = haar.getFieldNames()
for i in range(0, len(haar_fields)):
    print haar_fields[i], '=', haar_vecs[i]
```

The result would be like,

```
feature2x1_1 = 2242356.0
feature2x1_2 = 498204.0
feature2x2_1 = -1306748.0
feature3x1_1 = 22915652.0
...
Angle_feature2x1_1 = 2242356.0
Angle_feature2x1_2 = 498204.0
Angle_feature2x2_1 = -1306748.0
Angle_feature3x1_1 = 22915652.0
...
```

### SimpleCV Classifier (Machine Learning)
In this section, we will explore the SimpleCV Machine Learning package.

The detail documentation is located at:
http://simplecv.sourceforge.net/doc/SimpleCV.MachineLearning.html

#### Classifier - SVM (Support Vector Machine()
This class encapsulates a SVM (Support Vector Machine) Classifier.
See: http://en.wikipedia.org/wiki/Support_vector_machine

#### Classifier - Tree
This method encapsulates a number of tree-based machine learning approaches and associated meta algorithms.

Decision trees: http://en.wikipedia.org/wiki/Decision_trees

boosted adpative decision trees http://en.wikipedia.org/wiki/Adaboost

random forrests http://en.wikipedia.org/wiki/Random_forest

bagging (bootstrap aggregating) http://en.wikipedia.org/wiki/Bootstrap_aggregating

#### Classifier - Naive Bayes
This class encapsulates a Naive Bayes Classifier.
See: http://en.wikipedia.org/wiki/Naive_bayes

#### Classifier - KNN (K Nearest Neighbor)
This class encapsulates a K- Nearest Neighbor Classifier.
See http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

### Image Feature Extraction & Classification
Equipped with the Extractor and Classifier knowledge, we are ready to apply them in the actual image extraction and classification.

```python
def getExtractors():
    hue = HueHistogramFeatureExtractor(10)
    edge = EdgeHistogramFeatureExtractor(10)
    haar = HaarLikeFeatureExtractor(fname='haar.txt')
    return [hue, edge, haar]

def getClassifiers(extractors):
    svm = SVMClassifier(extractors)
    tree = TreeClassifier(extractors)
    bayes = NaiveBayesClassifier(extractors)
    knn = KNNClassifier(extractors)
    return {
        'svm': svm,
        'tree': tree,
        'bayes': bayes,
        'knn': knn
    }

extractors = getExtractors()
classifiers = getClassifiers(extractors)
```

We obtained the classifiers by constructing them with a set of extractors. To break the operation down a little more,
we have created a set of extractors, namely `hue`, `edge` and `haar` extractors. These extractors are capable of
taking an image and extracting the essential features that is in an image. The `hue` extractor will notice the pixels hue and slot the values into bins. Similar, `edge` and `haar` extractors will perform what they designed to calculate from an image.

Mathematically, these extractors have **vectorized** an image characteristics into a set of feature values.
A feature, such as hue of red, could be characterized by value of 0.018 in field name of `Hue0` for example.
We can think **feature vector** is a kind of formal description for an image. Every images from our training set, will be characterized
as a **feature vector**.

```python
img = Image('cookie_001.png')
hue_feature_vec = extractors[0].extract(img)
print hue_feature_vec
```

The print `hue_feature_vect` will be like,

```
  [0.03549028421988077,
   0.001203778883175249,
   0.00035178807892081684,
   0.0009786293278747852,
   0.00033078967479953006,
   0.000523404665689852,
   0.00028308959630179216,
   5.3921827867008034e-05,
   0.0,
   0.0]
```

We have read the image `cookie_001.png`, and use hue extractor (in the example code, `extractors[0]`,
to extract the hue feature vector with their binned values.

```python
  img = Image('cookie_001.png')

  # iterating through all the extractors, and combine their extracted vectors
  feature_name = []
  feature_vec = []
  for i in range(0, len(extractors)):
      feature_name += extractors[i].getFieldNames()
      feature_vec += extractors[i].extract(img)

  # we can see how many features are extracted
  print "Total number of features:", len(feature_name)
  print "Total number of values:", len(feature_vec)
  print

  # display some random sample features
  for x in range(0, 8):
      i = int(random.random()*len(feature_name))
      print feature_name[i], '=', feature_vec[i]
```

We shall see it listed some random features fields and values.

```
Total number of features: 260
Total number of values: 260

Angle_feature4x2_8 = 1662682.0
Angle_feature4x4_10 = 5007468.0
Angle0 = 0.0
Angle_feature5x1_8 = 16554016.0
Angle2 = 0.00310151227022
feature3x1_4 = 23657996.0
Angle_feature5x1_12 = -181856.0
Angle_feature4x1_1 = 43748.0
```

The classifier will use the extractors, obtaining the feature vector,
to train itself to produce it's internal parameters for classification.

### Classify with Trained Classifier
If we have done the classifier training, we can save the classifier parameters into a file.
For example, we have saved the `tree` classifier into `tree.dat`. Our example `tree` classifier
has been trained with the categories of `[brownie, candy, cookie, other]`. After loading,
we can use the loaded classifier to classify our test images.

```python
classifierFile = 'tree.dat'
classifier = TreeClassifier.load(classifierFile)

test_img1 = Image('test_001.png')
test_img2 = Image('test_002.png')
test_img3 = Image('test_003.png')

print "test_001.png is", classifier.classify(test_img1)
print "test_002.png is", classifier.classify(test_img2)
print "test_003.png is", classifier.classify(test_img3)
```

The result will be like,

```
test_001.png is brownie
test_002.png is candy
test_003.png is cookie
```
