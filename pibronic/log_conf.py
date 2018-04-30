import logging


# -----------------------------------------------------------
# LOGGING PREPERATIONS
# -----------------------------------------------------------
# predefined levels for logging
# CRITICAL 50
# ERROR    40
# WARNING  30
logging.FLOW = 25
# INFO     20
# DEBUG    10
logging.LOCK = 5
# NOTSET   0
# -----------------------------------------------------------

# add names
logging.addLevelName(logging.FLOW, "FLOW")
logging.addLevelName(logging.LOCK, "LOCK")


class MyLogger(logging.Logger):
    def flow(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.FLOW):
            self._log(logging.FLOW, message, args, **kwargs)

    def lock(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.LOCK):
            self._log(logging.LOCK, message, args, **kwargs)


logging.setLoggerClass(MyLogger)
log = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)-13s [%(levelname)s] %(funcName)s: %(message)s",
    # datefmt='%m/%d/%Y %I:%M:%S %p',
    datefmt='%d %I:%M:%S ',
    # level=logging.FLOW,
    level=logging.INFO,
    # level=logging.DEBUG,
    # level=logging.LOCK,
)
# -----------------------------------------------------------

# create logger
module_logger = logging.getLogger()


class Auxiliary:
    def __init__(self):
        self.logger = logging.getLogger('spam_application.auxiliary.Auxiliary')
        self.logger.info('creating an instance of Auxiliary')

    def do_something(self):
        self.logger.info('doing something')
        a = 1 + 1
        self.logger.info('done doing something')


def some_function():
    module_logger.info('received a call to "some_function"')