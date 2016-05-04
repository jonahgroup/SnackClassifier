#!/usr/bin/env python
import os
import sys
from optparse import OptionParser
from SimpleCV import (
    random,
    Color,
    KNNClassifier,
    NaiveBayesClassifier,
    TreeClassifier,
    SVMClassifier,
    HueHistogramFeatureExtractor,
    EdgeHistogramFeatureExtractor,
    HaarLikeFeatureExtractor,
    BOFFeatureExtractor,
    Image,
    ImageSet
)


class SnackTrainer():

    def __init__(self):
        self.classifier = None

    def createExtractor(self, extractorName, trainPaths=[]):
        if (extractorName == 'hue'):
            extractor = HueHistogramFeatureExtractor(10)
        elif (extractorName == 'edge'):
            extractor = EdgeHistogramFeatureExtractor(10)
        elif (extractorName == 'haar'):
            extractor = HaarLikeFeatureExtractor(fname='haar.txt')
        elif (extractorName == 'bof'):
            extractor = BOFFeatureExtractor()
            extractor.generate(trainPaths, imgs_per_dir=40)
            # need to build the vocabuary (feature words) for bag of feature
            # extractor.generate(trainPaths, imgs_per_dir=40)
        return extractor

    def createClassifier(self, classifierName, extractors):
        # extractors should be a list of extractor, e.g. [hue, edge]
        if (classifierName == 'svm'):
            classifier = SVMClassifier(extractors)
        elif (classifierName == 'tree'):
            classifier = TreeClassifier(extractors)
        elif (classifierName == 'bayes'):
            classifier = NaiveBayesClassifier(extractors)
        elif (classifierName == 'knn'):
            classifier = KNNClassifier(extractors)
        return classifier

    def getClassNameFromPath(self, trainPath):
        classes = []
        dirList = os.listdir(trainPath)
        for dirName in dirList:
            if os.path.isdir(trainPath + '/' + dirName):
                classes.append(dirName)
        return classes

    def setClassifier(self, classifier):
        self.classifier = classifier

    def getClassifier(self):
        return self.classifier

    def trainClassifier(self, classes, trainRootPath, featurePath=None):
        trainPaths = [trainRootPath + '/' + c for c in classes]
        if featurePath is None:
            self.classifier.train(trainPaths, classes, verbose=False)
        else:
            self.classifier.train(trainPaths, classes, savedata=featurePath, verbose=False)

    def testClassifier(self, classes, testRootPath):
        testPaths = [testRootPath + '/' + c for c in classes]
        print self.classifier.test(testPaths, classes, verbose=False)

    def saveClassifierFile(self, classifierFile):
        self.classifier.save(classifierFile)

    def getClassNames(self):
        return self.classifier.mClassNames

    def classifyImageFile(self, imageFile):
        image = Image(imageFile)
        return self.classifier.classify(image)

    def saveResults(self, classifier, imgs, resultPath):
        num = 1
        for img in imgs:
            className = classifier.classify(img)
            img.drawText(className, 10, 10, fontsize=60, color=Color.BLUE)
            img.save(resultPath + '/' + 'result_%02d.jpg' % num)
            num += 1

    def parse_options(self, args):
        """
        Parse command-line options
        """
        usage = "%prog [options] -c <classes> -a <train_path> -t <test_path> -r <result_path> -m <method>"
        parser = OptionParser(usage=usage)
        parser.add_option("-g", "--debug", action="store_true", dest="debug", default=False,
                          help="debugging mode"),
        parser.add_option("-c", "--class", action="store", dest="classes", default="",
                          help="detect classes, comma seperated"),
        parser.add_option("-a", "--train", action="store", dest="train_path", default="train",
                          help="training samples path"),
        parser.add_option("-t", "--test", action="store", dest="test_path", default="test",
                          help="testing samples path"),
        parser.add_option("-r", "--result", action="store", dest="result_path", default="result",
                          help="testing results path"),
        parser.add_option("-s", "--classifier", action="store", dest="classifier", default="tree",
                          help="using classifier (svm|tree|bayes|knn)"),
        parser.add_option("-f", "--feature", action="store", dest="feature_path", default="features.tab",
                          help="save training features into file"),
        parser.add_option("-e", "--save", action="store", dest="classifier_file", default="",
                          help="save classifier into file"),

        (self.options, args) = parser.parse_args(args)

        if not self.options.classifier:
            parser.print_help()
            exit(0)


def process():
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
        classifierFile = "%s.dat" % (classifier_name)

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
    process()
