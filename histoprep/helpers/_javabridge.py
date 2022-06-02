# DEPRACATED (for now at least)

# import logging
# import warnings

# import bioformats
# import javabridge
# import warnings

# JAVABRIDGE_RUNNING = False
# JAVABRIDGE_KILLED = False


# def _init_logger():
#     """Set logging level to WARN."""
#     rootLoggerName = javabridge.get_static_field(
#         "org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;"
#     )
#     rootLogger = javabridge.static_call(
#         "org/slf4j/LoggerFactory",
#         "getLogger",
#         "(Ljava/lang/String;)Lorg/slf4j/Logger;",
#         rootLoggerName,
#     )
#     logLevel = javabridge.get_static_field(
#         "ch/qos/logback/classic/Level", "WARN", "Lch/qos/logback/classic/Level;"
#     )
#     javabridge.call(
#         rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel
#     )


# def kill_javabrige():
#     """Kills javabridge session."""
#     global JAVABRIDGE_KILLED
#     if JAVABRIDGE_RUNNING:
#         logging.debug("Killing javabridge.")
#         javabridge.kill_vm()
#         JAVABRIDGE_KILLED = True


# def start_javabridge(silent: bool = False):
#     """Starts javabridge session."""
#     global JAVABRIDGE_RUNNING
#     # Start javabridge.
#     if JAVABRIDGE_KILLED:
#         raise RuntimeError(
#             "Javabridge session has been killed and cannot be restarted. "
#             "Restart session."
#         )
#     if not JAVABRIDGE_RUNNING:
#         if not silent:
#             warnings.warn(
#                 "Initializing javabridge. Call histoprep.kill_javabridge() at "
#                 "the end of your program."
#             )
#         javabridge.start_vm(class_path=bioformats.JARS)
#         _init_logger()
#         JAVABRIDGE_RUNNING = True
