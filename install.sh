while read -r -u5 line
do
    # If working from Isaac Sim's python binary:
    #$ISAACSIM_PYTHON_EXE -m pip install "$line"
    
    # If working from a conda environment:
    python -m pip install $line

    #NOTE: Installing packages as editable from command line seems to work if this script doesn't get the package installed.
done 5< requirements.txt


# NOTE: Hacks that make things work.

# (1)
conda install pydensecrf
# For now I am sticking to the conda version but this is potentially scarier since instead of just having
# a suspicious pydensecrf, it suspiciously modifies almost everything.
# This is absolutely a hazard, if you don't want to potentially ruin your drivers
# you can clone the git repository and pip install -e .
# Though you will get very scary compile errors, you can ignore them. :)
# This also touches the cuda version somehow--it might be what seems to upgrade the env to 12.2.

# (2)
python -m pip install --upgrade requests
# This somehow causes a JSONRequestError import (~approx. name) from Neptune to resolve. It also appears to install many unrelated things
# that are incredibly suspicious (i.e., new cuda version 11.6, cudnn)
# The true origin of the problem seems to be that IsaacSim is putting certain extensions in the sys.path that WVN needed to get from a dependency.
# I.e., in the import module error stack trace we get something like
# Stack Trace:
    # WVN                   failed to import   X   from    SOME_MODULE_IN_WVN
    # SOME_MODULE_IN_WVN    failed to import   Y   from   SOME_NEPTUNE_MODULE
    # SOME_NEPTUNE_MODULE   failed to import   Z   from   SOME_ISAAC_SIM_MODULE
# Base WVN should never import *anything* from IsaacSim--so the best solution is probably to prevent sys.path from being malformed but 
# the issue goes so deep this will do for now... 
