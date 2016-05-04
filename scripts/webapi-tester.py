#!/usr/bin/env python
import os
import sys
from optparse import OptionParser

USE_HTTPIE = True


class WebAPITester():

    def __init__(self):
        pass

    def sendHttp(self, jsonFile):
        jsonName = os.path.basename(jsonFile)
        (filename, fileext) = os.path.splitext(jsonName)

        if self.options.output_path:
            if not os.path.exists(self.options.output_path):
                os.makedirs(self.options.output_path)
            output = self.options.output_path + '/' + filename + '.res'
        else:
            if filename:
                output = filename + '.res'
            else:
                output = ''

        url = self.options.url

        method = self.options.http_op
        if USE_HTTPIE:
            command = 'http --json --pretty=format'
            if (self.options.verbose):
                command += ' -v'
            command += ' %s %s' % (method, url)
        else:
            command = 'curl -i -H "Accept: application/json" -H "Content-Type: application/json" -X %s %s' % (
                method, url)

        if (method == 'POST') or (method == 'PUT'):
            command += ' < %s' % (jsonFile)
        if (output):
            command += ' > %s' % (output)

        if (self.options.debug):
            print command
        if (not self.options.debug):
            os.system(command)

    def getJsonFilesFromPath(self, jsonPath, jsonExt):
        jsonFiles = []
        dirList = os.listdir(jsonPath)
        for file in dirList:
            (filename, fileext) = os.path.splitext(file)
            filePath = jsonPath + '/' + file
            if os.path.isfile(filePath) and fileext == jsonExt:
                jsonFiles.append(filePath)
        return jsonFiles

    def getContextUrl(self):
        return self.contextUrl

    def parse_options(self, args):
        """
        Parse command-line options
        """
        usage = "%prog [options]"
        parser = OptionParser(usage=usage)
        parser.add_option("-g", "--debug", action="store_true", dest="debug", default=False,
                          help="debugging mode"),
        parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
                          help="verbose output"),
        parser.add_option("-p", "--http", action="store", dest="http_op", default="",
                          help="web api http operation (GET|POST|PUT|DELETE)"),
        parser.add_option("-u", "--url", action="store", dest="url", default="http://localhost:5000/api",
                          help="web api url"),
        parser.add_option("-i", "--input", action="store", dest="input_path", default="",
                          help="input JSON files path"),
        parser.add_option("-e", "--ext", action="store", dest="input_ext", default=".json",
                          help="test Json file extension"),
        parser.add_option("-o", "--output", action="store", dest="output_path", default="",
                          help="testing results path"),
        (self.options, args) = parser.parse_args(args)

        # must specify a http operation
        if not self.options.http_op:
            parser.print_help()
            exit(0)

        # if POST or PUT operation, it requires to have test input json file
        if self.options.http_op == 'POST' or self.options.http_op == 'PUT':
            if not self.options.input_path:
                parser.print_help()
                exit(0)


def process():
    tester = WebAPITester()
    tester.parse_options(sys.argv)

    debug = tester.options.debug
    if (os.path.isdir(tester.options.input_path)):
        json_files = tester.getJsonFilesFromPath(
            tester.options.input_path, tester.options.input_ext)
    else:
        json_files = [tester.options.input_path]

    if (debug):
        print "Json File List: ", json_files
    for file in json_files:
        tester.sendHttp(file)

"""
main program
"""
if __name__ == "__main__":
    process()
