import io
import logging
import time
import tqdm

__all__ = [
    'logger',
    'progress_bar',
]

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
COLORS = {
    'WARNING': RED,
    'INFO': BLUE,
    'DEBUG': CYAN,
    'CRITICAL': YELLOW,
    'ERROR': RED
}
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
TQDM_OUTPUT = '_OUTPUT_TQDM_'
TQDM_END = '_END_TQDM_'


def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace(
            "$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        msg = record.msg
        if self.use_color and levelname in COLORS:
            # Background is 40, foreground 30.
            levelname_color = COLOR_SEQ % (
                30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color

        return logging.Formatter.format(self, record)


class TqdmLoggingHandler(logging.Handler):

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        FORMAT = ("[$BOLD%(levelname)s$RESET] %(message)s ")
        TQDM_FORMAT = "    %(message)s"
        COLOR_FORMAT = formatter_message(FORMAT, True)
        COLOR_FORMAT_TQDM = formatter_message(TQDM_FORMAT, True)
        self.formatter = ColoredFormatter(COLOR_FORMAT)
        self.tqdm = ColoredFormatter(COLOR_FORMAT_TQDM)
        self.last_tqdm_output = None
        self.max_tqdm_len = -1

    def emit(self, record):
        try:
            if TQDM_OUTPUT in record.msg:
                record.msg = record.msg.replace(TQDM_OUTPUT, "")
                msg = self.tqdm.format(record)
                end = "\r"
                self.last_tqdm_output = msg
                self.max_tqdm_len = max(self.max_tqdm_len, len(msg))
            elif TQDM_END in record.msg:
                record.msg = record.msg.replace(TQDM_END, "")
                msg = self.tqdm.format(record)
                end = "\n"
            else:
                msg = self.format(record)
                end = "\n"
            tqdm.tqdm.write(msg, end=end)
            self.flush()
        except Exception:
            self.handleError(record)


class TqdmToLogger(io.StringIO):
    """Sends tqdm output to logger"""
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(TqdmLoggingHandler())


def progress_bar(iterable, **kwargs):
    """Wraps tqdm.tqdm and passes everything nicely through logger."""
    out = TqdmToLogger(logger, level=logging.INFO)
    desc = kwargs.get('desc', 'progress')
    kwargs['desc'] = TQDM_OUTPUT + desc
    for x in tqdm.tqdm(iterable, file=out, bar_format=BAR_FORMAT, **kwargs):
        yield x
    logger.info(TQDM_END)
    return
