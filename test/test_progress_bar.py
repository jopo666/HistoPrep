import io
import sys

from histoprep.helpers._verbose import progress_bar


class CaptureStdout(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def test_progress_bar():
    with CaptureStdout() as output:
        for logs, __ in progress_bar([0, 1], desc="test", log_values=True):
            logs["dog"] = "good_boi"
    assert output == [
        "test |          | 0/2 [00:00<???]",
        "                                 ",
        "test |#####     | 1/2 [00:00<00:00]",
        "                                   ",
        "test |##########| 2/2 [00:00<00:00, dog=good_boi]",
    ]


def test_progress_bar_logs_suppress():
    # Progress bar should still return logs!
    with CaptureStdout() as output:
        for logs, i in progress_bar(range(10), suppress=True, log_values=True):
            pass
    assert output == []


def test_progress_bar_empty_iterable():
    for x in progress_bar([]):
        pass
