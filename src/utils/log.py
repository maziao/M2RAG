import os
import time
import logging


def configure_log(name: str = None):
    if name is not None:
        log_path = f"./log/run/{name}/{time.strftime('%Y%m%d-%H%M%S')}.log"
    else:
        log_path = f"./log/run/others/{time.strftime('%Y%m%d-%H%M%S')}.log"

    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_path,
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
