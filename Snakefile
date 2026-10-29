from snakemake.utils import min_version
min_version("6.0")

configfile: f"configs/environments/default.yaml"

def expand_path(path):
    # Expands environment variables like $VAR
    path = os.path.expandvars(path)

    # Expands the '~' symbol to the user's home directory
    path = os.path.expanduser(path)

    return path

# get various paths from config file
config["TMP_DIR"] = expand_path(f"{config['tmp_dir']}")
config["RAW_DIR"] = expand_path(f"{config['raw_dir']}/raw")
config["PROCESSED_DIR"] = expand_path(f"{config['processed_dir']}/processed")
config["COMPRESSED_DIR"] = expand_path(f"{config['compressed_dir']}/compressed")
config["UNCOMPRESSED_DIR"] = expand_path(f"{config['uncompressed_dir']}/uncompressed")


# include all snakefiles for all individual datasets
# includes are relative to the directory of the Snakefile in which they occur

module perich_miller_population_2018_module:
    snakefile: "data/scripts/perich_miller_population_2018/Snakefile"
    config: config
use rule * from perich_miller_population_2018_module as perich_miller_population_2018_*
use rule all from perich_miller_population_2018_module as perich_miller_population_2018

module odoherty_sabes_nonhuman_2017_module:
    snakefile: "data/scripts/odoherty_sabes_nonhuman_2017/Snakefile"
    config: config
use rule * from odoherty_sabes_nonhuman_2017_module as odoherty_sabes_nonhuman_2017_*
use rule all from odoherty_sabes_nonhuman_2017_module as odoherty_sabes_nonhuman_2017

module churchland_shenoy_neural_2012_module:
    snakefile: "data/scripts/churchland_shenoy_neural_2012/Snakefile"
    config: config
use rule * from churchland_shenoy_neural_2012_module as churchland_shenoy_neural_2012_*
use rule all from churchland_shenoy_neural_2012_module as churchland_shenoy_neural_2012

module pei_churchland_nlbmaze_2021_module:
    snakefile: "data/scripts/pei_churchland_nlbmaze_2021/Snakefile"
    config: config
use rule * from pei_churchland_nlbmaze_2021_module as pei_churchland_nlbmaze_2021_*
use rule all from pei_churchland_nlbmaze_2021_module as pei_churchland_nlbmaze_2021  
