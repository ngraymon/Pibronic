#!python3

# system imports
import os
import sys

# third party imports

# local imports
import pibronic
from pibronic.log_conf import log
from pibronic.pimc import pimc as engine
from pibronic.data import file_structure

if (__name__ == '__main__'):
    log.debug("Starting execution of job")

    assert len(sys.argv) == 4, "incorrect number of arguments"
    input_parameters = sys.argv[1]
    path_scratch = sys.argv[2]
    id_job = sys.argv[3]

    """do a plus minus calculation"""
    data = engine.BoxDataPM.from_json_string(input_parameters)
    print(data.hash_vib)
    print(data.hash_rho)
    data.preprocess()

    path_fixed = os.path.normpath(os.path.join(path_scratch, "../.."))
    # TODO - this is redundant because a FS is already created in from_json_string
    FS = file_structure.FileStructure.from_boxdata(path_fixed, data)

    result = engine.BoxResultPM(data=data)
    result.path_root = FS.path_rho_results
    result.id_job = id_job

    engine.block_compute_pm(data, result)
