""" """

# system imports

# local imports
from ..context import pibronic
import pibronic.server.pimc_job_dispatcher as pjd

# third party imports
import pytest


def test_extract_job_id():
    test_job_id = 100
    out = "e"*20 + str(test_job_id)
    ret = pjd.extract_job_id(out.encode(), None)
    assert ret == test_job_id
    return


def test_submit_pimc_job():
    parameter_dictionary = {"hostname": "fake_host"}
    FS = "fake FileStructure object"

    with pytest.raises(Exception, message="This server is currently not supported"):
        pjd.submit_pimc_job(FS=FS, param_dict=parameter_dictionary)

    return
