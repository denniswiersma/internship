#!/usr/bin/env python3

"""
Implemnets a singleton class that provides a centralised logging functionality.

Since a runID is generated for each model fit, a file handler can only be added
once the runID is known. Therefore, log messages are buffered until the runID is
known and then, depending on the stage of the analysis, flushed to file.

Multiple fits may rely on the same data loading and preparation steps, so the
these buffers are only cleared if the data is reloaded or refetched respectively.
(Flushing these two buffers is yet to be implemented since this is not yet
 critical.)
"""

# METADATA

# IMPORTS
import logging
from pathlib import Path


# CLASSES
class LogManager:
    # current instance of the class
    _instance = None
    # whether the class has been initialised
    _init_done = False

    def __new__(cls):
        """
        Defines the singleton behaviour of this class.
        """
        if not cls._instance:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls.__init__(cls._instance)
        return cls._instance

    def __init__(self):
        """
        Initialises the formatter, logger, and buffers on the first initialisation
        of this class.
        """
        if not self._init_done:
            self.formatter = logging.Formatter(
                "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
            )
            self.logger = self._setup_logger()
            # set flag to prevent re-initialisation
            self._init_done = True

            # initialise log message buffers
            self.load_buffer = []
            self.prep_buffer = []
            self.fit_buffer = []

    def _setup_logger(self):
        """
        Sets up the logger with a stream handler in order to print log messages
        to the console.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(self.formatter)
        logger.addHandler(console)

        return logger

    def add_file_handler(self, path: Path):
        """
        Adds a file handler to the logger to write log messages to a given file.

        :param path: path to the file to write log messsages to. Suffix will be
        changed to ".log" if not already.
        """
        # remove any existing file handlers (note, stream handler is not removed)
        # in order to prevent duplicate log entries
        self.remove_file_handler()

        log_file = path.with_suffix(".log")
        # make sure the directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def remove_file_handler(self):
        """
        Remove any existing file handlers from the logger.
        """
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)

    def add_load_buffer(self, message: str):
        """
        Appends a message to the load buffer while also printing it to the console.

        :param message: the log message.
        """
        self.logger.info(message)
        self.load_buffer.append(message)

    def add_prep_buffer(self, message: str):
        """
        Appends a message to the prep buffer while also printing it to the console.

        :param message: the log message.
        """
        self.logger.info(message)
        self.load_buffer.append(message)

    def add_fit_buffer(self, message: str):
        """
        Appends a message to the fit buffer while also printing it to the console.

        :param message: the log message.
        """
        self.logger.info(message)
        self.fit_buffer.append(message)

    def flush_buffers_to_file(self, path: Path):
        log_file = path.with_suffix(".log")
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # clear the fit buffer of messages from a possible earlier fit
        self.fit_buffer.clear()

        # write existing messages to file
        with open(log_file, "a") as file:
            for message in (
                self.load_buffer + self.prep_buffer + self.fit_buffer
            ):
                file.write(
                    self.formatter.format(
                        logging.makeLogRecord(
                            {
                                "msg": message,
                                "levelname": "INFO",
                                "name": self.logger.name,
                                "asctime": "",
                            }
                        )
                    )
                    + "\n"
                )


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
