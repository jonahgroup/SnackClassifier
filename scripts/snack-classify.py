#!/usr/bin/env python
import sys
from optparse import OptionParser

from SimpleCV import *


class SnackClassify():

    def __init__(self):
        pass

    def load(self, classifierFile):
        self.classifier = TreeClassifier.load(classifierFile)

    def classify(self, imageFile):
        image = Image(imageFile)
        return self.classifier.classify(image)

    def classNames(self):
        return self.classifier.mClassNames

    def parse_options(self, args):
        """
        Parse command-line options
        """
        usage = "%prog [options] -c <classifier_file> -i <image>"
        parser = OptionParser(usage=usage)
        parser.add_option("-g", "--debug", action="store_true", dest="debug", default=False,
                          help="debugging mode"),
        parser.add_option("-c", "--classifier", action="store", dest="classifier_file", default="",
                          help="load classifier from file"),
        parser.add_option("-i", "--image", action="store", dest="image_file", default="",
                          help="classify this image file"),

        (self.options, args) = parser.parse_args(args)

        if not self.options.classifier_file or not self.options.image_file:
            parser.print_help()
            exit(0)


def process():
    snack_bot = SnackClassify()
    snack_bot.parse_options(sys.argv)

    classifierFile = snack_bot.options.classifier_file
    imageFile = snack_bot.options.image_file

    snack_bot.load(classifierFile)
    class_name = snack_bot.classify(imageFile)
    print class_name

"""
main program
"""
if __name__ == "__main__":
    process()
