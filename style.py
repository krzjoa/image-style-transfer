# -*- coding: utf-8 -*-
'''

Image transfer style with multiple algorithms

Usage:
    style.py <img> <style> <output>

Options:
   -h   Show help.


'''


from docopt import docopt


if __name__ == '__main__':

    arguments = docopt(__doc__, version="Image styler 1.0")
