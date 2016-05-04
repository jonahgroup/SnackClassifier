import os
from optparse import OptionParser
from SimpleCV import (
    Color,
    KNNClassifier,
    NaiveBayesClassifier,
    TreeClassifier,
    SVMClassifier,
    HueHistogramFeatureExtractor,
    EdgeHistogramFeatureExtractor,
    HaarLikeFeatureExtractor,
    BOFFeatureExtractor,
    Image
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
            extractor = HaarLikeFeatureExtractor(fname='static/trainer/haar.txt')
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

    def createClassifierFromFile(self, classifierType, classifierFile):
        if classifierType == 'svm':
            classifier = SVMClassifier.load(classifierFile)
        elif classifierType == 'tree':
            classifier = TreeClassifier.load(classifierFile)
        elif classifierType == 'bayes':
            classifier = NaiveBayesClassifier.load(classifierFile)
        elif classifierType == 'knn':
            classifier = KNNClassifier.load(classifierFile)
        else:
            classifier = None
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
        parser.add_option("-a", "--train", action="store", dest="train_path", default="static/trainer/train",
                          help="training samples path"),
        parser.add_option("-t", "--test", action="store", dest="test_path", default="static/trainer/test",
                          help="testing samples path"),
        parser.add_option("-r", "--result", action="store", dest="result_path", default="static/trainer/result",
                          help="testing results path"),
        parser.add_option("-s", "--classifier", action="store", dest="classifier", default="bayes",
                          help="using classifier (svm|tree|bayes|knn)"),
        parser.add_option("-f", "--feature", action="store", dest="feature_path", default="static/trainer/features.tab",
                          help="save training features into file"),
        parser.add_option("-e", "--save", action="store", dest="classifier_file", default="",
                          help="save classifier into file"),

        (self.options, args) = parser.parse_args(args)

        if not self.options.classifier:
            parser.print_help()
            exit(0)
