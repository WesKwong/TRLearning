# ----------------- get global variables ----------------- #
import utils.glob as glob
logger = glob.get('logger')
results_path = glob.get('results_path')
config = glob.get('config')
# --------------------------- - -------------------------- #
def run_experiment(experiment):
    pass

from utils.experiment_manager import create_experiments
def main():
    logger.info("Start Experiments")
    expt_settings = config.get_expt_settings()
    expts = create_experiments(expt_settings)
    logger.info(f"Expt Num: {len(expts)}")
    set_num = 0
    for name, expt in expts.items():
        logger.info(f"Expt Name: {name}, Setting Num: {len(expt)}")
        set_num += len(expt)
    logger.info(f"Total Setting Num: {set_num}")
    for i, (name, expt) in enumerate(expts.items()):
        logger.info(f"Running ({i+1}/{len(expts)}) Experiment: {name}")
        run_experiment(expt)
    logger.info("End Experiments")