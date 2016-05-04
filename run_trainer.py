#!/usr/bin/env python
import sys
from SimpleCV import (
    random,
    ImageSet
)
from app.snack_trainer import SnackTrainer


def run_trainer():
    snack_trainer = SnackTrainer()
    snack_trainer.parse_options(sys.argv)

    debug = snack_trainer.options.debug
    if snack_trainer.options.classes == "":
        classes = snack_trainer.getClassNameFromPath(snack_trainer.options.train_path)
    else:
        classes = snack_trainer.options.classes.split(',')
    trainPaths = [snack_trainer.options.train_path + '/' + c for c in classes]
    testPaths = [snack_trainer.options.test_path + '/' + c for c in classes]
    extractors = [
        snack_trainer.createExtractor('hue'),
        snack_trainer.createExtractor('edge'),
        snack_trainer.createExtractor('haar')
    ]
    resultPath = snack_trainer.options.result_path
    featurePath = snack_trainer.options.feature_path
    classifierFile = snack_trainer.options.classifier_file

    classifier_name = snack_trainer.options.classifier
    classifier = snack_trainer.createClassifier(classifier_name, extractors)
    snack_trainer.setClassifier(classifier)

    print "Using Classifier:", classifier_name
    print "Training Set:", trainPaths
    print "Training Features Save As:", featurePath
    snack_trainer.trainClassifier(classes, snack_trainer.options.train_path, featurePath)

    if classifierFile == "":
        classifierFile = "static/trainer/" + "%s.dat" % (classifier_name)

    print "Classifier Data Save As:", classifierFile
    classifier.save(classifierFile)

    if (debug):
        print "Testing Set:", testPaths
        imgs = ImageSet()
        for p in testPaths:
            imgs += ImageSet(p)
            random.shuffle(imgs)

        snack_trainer.testClassifier(classifier, classes, testPaths)

        print "Test Result:", resultPath
        snack_trainer.saveResults(classifier, imgs, resultPath)
    print "Done"

"""
main program
"""
if __name__ == "__main__":
    run_trainer()
