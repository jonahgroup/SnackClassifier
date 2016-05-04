import os
from app.settings import (
    CLASSIFIER_TYPE,
    CLASSIFIER_TRAIN_PATH
)

from snack_trainer import SnackTrainer
from SimpleCV import (
    Image
)

from log import log


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(trainer=None)
def get_trainer(retrain=False):
    if retrain or get_trainer.trainer is None:
        log("training in progress ...")
        get_trainer.trainer = SnackTrainer()
        classes = get_trainer.trainer.getClassNameFromPath(CLASSIFIER_TRAIN_PATH)
        # trainPaths = [CLASSIFIER_TRAIN_PATH + '/' + c for c in classes]
        extractors = [
            get_trainer.trainer.createExtractor('hue'),
            get_trainer.trainer.createExtractor('edge'),
            get_trainer.trainer.createExtractor('haar')
        ]

        classifier = get_trainer.trainer.createClassifier(CLASSIFIER_TYPE, extractors)
        get_trainer.trainer.setClassifier(classifier)
        get_trainer.trainer.trainClassifier(classes, CLASSIFIER_TRAIN_PATH)
        log("trained with classes: " + str(get_trainer.trainer.getClassNames()))
    return get_trainer.trainer


def get_image_class_names(trainer):
    log("get_image_class_names()")
    class_names = trainer.getClassNames()
    return {
        "classes": class_names
    }


def get_image_classify(trainer, img):
    log("get_image_classify(img=%s)" % img)
    image_c1ass = trainer.classifyImageFile(img)
    return {
        "image": img,
        "class": image_c1ass
    }


def save_trained_image(train_as, img):
    image = Image(img)
    trainPath = CLASSIFIER_TRAIN_PATH + '/' + train_as
    # make sure the train/{class} directory exist
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
    imageName = os.path.basename(img)
    (filename, fileext) = os.path.splitext(imageName)
    # make sure imageName is not duplicated by seq number
    uniq = 1
    imagePath = '%s/%s_%03d%s' % (trainPath, train_as, uniq, fileext)
    while os.path.exists(imagePath):
        imagePath = '%s/%s_%03d%s' % (trainPath, train_as, uniq, fileext)
        uniq += 1
    image.save(imagePath)
    log("save_trained_image(train_as=%s, img=%s, saved=%s)" % (train_as, img, imagePath))
    return {
        "image": img,
        "class": train_as,
        "saved": imagePath
    }
