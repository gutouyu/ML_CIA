
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('echo', None, 'Text to echo.')


def func(argv):
  del argv  # Unused.

  print('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info),
        file=sys.stderr)
  logging.info('echo is %s.', FLAGS.echo)


if __name__ == '__main__':
  app.run(func)


# bash exe: `python absl_example.py -echo "test echo"`