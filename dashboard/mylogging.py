# -*- coding: utf-8 -*-

'''
  Ma version de logging. Je la préfère au fichier de configuration car elle permet
  de savoir dans quel fichier je suis (cf __name__ dans l'appel).
  Elle a aussi une mémoire qui permet de récupérer le dernier message.

  Pour ce qui est du seuil de déclanchement des avertissements il est préférable
  de ne rien mettre dans les autres fichier ainsi il suffit de modifier la valeur
  par défaut ici pour que toute la bibliothèque change le seuil.

  cf https://docs.python.org/2/howto/logging.html pour la doc

  >>> from testfixtures import LogCapture
  >>> l = LogCapture()
  >>> getLogger(__name__, level=logging.DEBUG).error('doctest'); print(l)
  mylogging ERROR
    doctest
'''

import logging
import logging.handlers

INFO = logging.INFO
DEBUG = logging.DEBUG

log_level = logging.DEBUG  # change this if you want a another default variable

class LastMessageLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.record = None
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def emit(self, record):
        super().handle(record)
        self.last_record = record

    def handle(self, record):
        # Call emit to store the last message, then let logging handle it
        self.emit(record)

    def get_last_message(self):
        return self.last_record.msg

def getLogger(name, level=log_level,
              filename=None, file_level=None):
    logging.setLoggerClass(LastMessageLogger)
    logger = logging.getLogger(name)
    #stream_handler = logging.StreamHandler()
    #logger.addHandler(stream_handler)
    logger.setLevel(level)
    # create formatter
    # create file handle if needed
    if filename is not None:
        print("Logs of %s go to %s" % (name, filename))
        fh = logging.handlers.RotatingFileHandler(filename, maxBytes=10*1024*1024, backupCount=3)
        if file_level is None:
            fh.setLevel(level)
        else:
            fh.setLevel(file_level)
        fh.setFormatter(logger.formatter)
        logger.addHandler(fh)
    else:
        # create console handler and set level to debug
        sh = logging.StreamHandler()
        sh.set_name("handler of %s" % name)
        sh.setLevel(level)
        sh.setFormatter(logger.formatter)
        logger.addHandler(sh)
    return logger


