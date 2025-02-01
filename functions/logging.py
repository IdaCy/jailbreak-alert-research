import logging
import os

def init_logger(
    log_file="a_debug.log",
    file_level=logging.DEBUG,   
    console_level=logging.CRITICAL
):
    """
    Logger that writes all debug logs to a file but only CRITICAL logs
    to the console (i.e., no debug/info/warning/error on-screen).
    """
    # 1) Remove any existing handlers for the named logger and the root logger
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.setLevel(logging.CRITICAL)

    logger = logging.getLogger("ReNeLLMLogger")
    logger.setLevel(logging.DEBUG)  # Internally handle all messages
    logger.propagate = False        # Don't pass logs to root

    # File handler (capture DEBUG and up to file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(file_level) 
    fmt_file = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)

    # Console handler (only CRITICAL)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)  # CRITICAL means almost nothing prints
    fmt_console = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)

    logger.debug("Logger initialized (console=CRITICAL, file=DEBUG).")
    return logger
