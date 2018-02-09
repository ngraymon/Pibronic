#!python3

# system imports
import subprocess
import json
import sys
import os

# third party imports

# local imports
# from .. import constants

# cheating for now
sys.path.append('/home/ngraymon/pibronic/')
from pibronic import constants

cmd = ['/usr/bin/modulecmd', 'python', 'load', 'julia/0.6.0']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, error = p.communicate()
exec(out)

assert(len(sys.argv) in [3,4])
assert(sys.argv[2].isnumeric() and int(sys.argv[2]) >= 0)

temperature = 300.0 if (len(sys.argv) == 3) else float(sys.argv[3])
beta = 1. / (temperature * constants.boltzman)

julia = "julia /home/ngraymon/julia_confirm/VibronicToolkit.jl/bin/"
a = julia + "analytical.jl --conf {F:} --beta {T:}"
s = julia + "sos.jl --conf {F:} --beta {T:} --basis-size {B:}"
t = julia + "trotter.jl --conf {F:} --beta {T:} --basis-size {B:} --num-links {P:}"
x = julia + "sampling.jl --conf {F:} --beta {T:} --num-links {P:} --num-samples {X:}"

id_data = int(sys.argv[1])
id_rho = int(sys.argv[2])

path_root_data = "/work/ngraymon/pimc/data_set_{:d}/"
path_root_rho = path_root_data + "rho_{:d}/"
path_sampling = path_root_rho.format(id_data, id_rho) + "parameters/sampling_model.json"
path_sos_rho  = path_root_rho.format(id_data, id_rho) + "parameters/sos_B88.json"
path_sos_data = path_root_data.format(id_data) + "parameters/sos_B88.json"

if os.path.isfile(path_sos_data):
    with open(path_sos_rho, 'r') as sosFile:
        data = sosFile.read()
        if len(data) > 1:
            old_dict = json.loads(data)

# print(old_dict,"\n")

cmd = a.format(F=path_sampling, T=beta)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
(out, error) = p.communicate()
# print(out.decode())
# print(error.decode())
rawOutput = ''.join(list(out.decode()))
lineList = [string for string in rawOutput.split("\n") if string is not '']
output_dict = dict(line.split(":") for line in lineList)



temp_dict = {}
keyDict = {
    "Z_coupled": "ZH",
    "Z_harmonic": "Zrho",
    "Z_sampling": "Zrho",
    "Z_sampling+beta": "Zrho+(beta)",
    "Z_sampling-beta": "Zrho-(beta)",
    "E_coupled": "E",
    "E_harmonic": "Erho",
    "E_sampling": "Erho",
    "Cv_coupled": "Cv",
    "Cv_harmonic": "Cvrho",
    "Cv_sampling": "Cvrho",
    #
    "beta": "beta",
    "tau": "tau",
}

for newKey, oldKey in keyDict.items():
    if oldKey in output_dict:
        temp_dict[newKey] = float(output_dict[oldKey])

new_dict = { temperature : temp_dict}

old_dict.update(new_dict)

with open(path_sos_rho, 'w') as sosFile:
    json.dump(old_dict, sosFile)

with open(path_sos_data, 'w') as sosFile:
    json.dump(old_dict, sosFile)