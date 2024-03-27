# ----------------- get global variables ----------------- #
import utils.glob as glob
logger = glob.get('logger')
results_path = glob.get('results_path')
config = glob.get('config')
# --------------------------- - -------------------------- #

from tools.data import NgramData
from models.Model import NGramModel

def run_experiment(experiment):
    data_loader = NgramData()
    data_loader.load_data(path=config.data_path, name='train.txt', target='train')
    train_data = data_loader.get_train_set()
    data_loader.load_data(path=config.data_path, name='test.1.txt', target='test')
    test1_data = data_loader.get_test_set()
    data_loader.load_data(path=config.data_path, name='test.2.txt', target='test')
    test2_data = data_loader.get_test_set()
    for setting_cnt, setting in enumerate(experiment):
        logger.success(f"Running ({setting_cnt+1}/{len(experiment)}) Setting")
        hp = setting.hyperparameters
        logger.info(setting)

        model = NGramModel(train_data, hp, setting)

        logger.info(f"Perplexity Test1: {model.perplexity(test1_data)}")
        logger.info(f"Perplexity Test2: {model.perplexity(test2_data)}")
        setting.log({
            'perplexity_test1': model.perplexity(test1_data),
            'perplexity_test2': model.perplexity(test2_data),
        }, printout=False)
        setting.save_to_disc(results_path)

        del model
    del train_data, test1_data, test2_data


from utils.experiment_manager import create_experiments
def main():
    logger.success("Start Experiments")
    expt_settings = config.get_expt_settings()
    expts = create_experiments(expt_settings)
    logger.info(f"Experiment Number: {len(expts)}")
    set_num = 0
    for name, expt in expts.items():
        logger.info(f"Experiment Name: {name} | Setting Number: {len(expt)}")
        set_num += len(expt)
    logger.info(f"Total Setting Number: {set_num}")
    for i, (name, expt) in enumerate(expts.items()):
        logger.success(f"Running ({i+1}/{len(expts)}) Experiment: {name}")
        run_experiment(expt)
    logger.success("End Experiments")