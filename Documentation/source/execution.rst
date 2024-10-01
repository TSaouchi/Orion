.. include:: references.rst

Execution
=========

Local machine
-------------

Cluster (HPC)
-------------

1. Run on graphical nodes or user home

.. code-block:: bash
    :emphasize-lines: 4, 7
    
    #!/bin/bash

    # Load modules or set up environment (if needed)
    source <source any needed env>

    # Execute your Python script
    python YOUR_PYTHON_CODE.py

2. Submit to be executed on calculation nodes

.. code-block:: bash
    :emphasize-lines: 3, 8, 9, 10, 11, 12, 16, 19

    #!/bin/bash

    #SBATCH --job-name=NAME
    #SBATCH -o output.out.tce
    #SBATCH -e output.err.tce
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --mem=100G
    #SBATCH -t 7-00:00:00
    #SBATCH -p <name of the calculation queue>
    #SBATCH --qos=<name of the calculation queue>-limits
    #SBATCH --mail-user=E-MAIL
    #SBATCH --mail-type=BEGIN,END,FAIL

    # Load modules or set up environment (if needed)
    source <source any needed env>

    # Execute your Python script
    python YOUR_PYTHON_CODE.py
