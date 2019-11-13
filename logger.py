import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
	    pass

# sys.stdout = Logger('train.log', sys.stdout)
#sys.stderr = Logger(a.log_file, sys.stderr)		# redirect std err, if necessary

# now it works

print('print something')