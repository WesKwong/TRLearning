# ----------------- init global variables ---------------- #
import utils.glob as glob
glob._init()
# ---------------------- get config ---------------------- #
from configs.config import config
glob.set('config', config)
# -------------------- set random seed ------------------- #
from tools.seed import set_seed
set_seed(config.random_seed)
# ---------------------- init logger --------------------- #
import sys
from loguru import logger
logger.remove()
logger.add(sys.stdout, level=config.log_level)
from utils.time import get_time_str
from utils.misc import get_results_dir

expt_time = get_time_str()
results_path = get_results_dir(config.results_path, expt_time+'/')
logger.add(results_path + f"{expt_time}.log", level=config.log_level)
glob.set('results_path', results_path)
glob.set('logger', logger)
# --------------------------- - -------------------------- #
from main import main

@logger.catch
def run():
    main()

if __name__ == "__main__":
    run()
