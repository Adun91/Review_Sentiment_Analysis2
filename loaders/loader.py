__author__ = 'vreal'

from os.path import join
import numpy
import csv
import theano


class Loader():
    def load(self, file_name = "train.tsv"):
        with open(join("..", "data", file_name)) as tsv:
            lines = [line for line in csv.reader(tsv, dialect="excel-tab")]
            return theano.shared(numpy.asarray(lines[1:]))

