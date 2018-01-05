import cPickle as c

def load(path):
    with open(path, "r") as f:
        return c.load(f)