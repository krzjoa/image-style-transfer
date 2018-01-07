# -*- coding: utf-8 -*-
# Base styler class

from abc import abstractmethod, ABCMeta


class BaseStyler(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def process(self, original, style):
        pass

    @abstractmethod
    def load_process(self, original_path, style_path):
        pass