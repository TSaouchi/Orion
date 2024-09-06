.. include:: references.rst

Debug & Performence
===================

Some debugging processes are available in Orion :

- Output at different stages python objects memory usage (native to Orion)
- Debug using ``pdb`` tracer, this is useful to debug manually the code in a terminal (equivalent of debugging when using IDEs) (native to Orion)

To unable debugging one has to set the global ``debug`` argument to ``True`` by either doing: 

a. At execution:

.. code-block:: bash
    
    python orion.py --debug

b. Inside the code set in the main part :python:`debug = True`

For debugging purpose, one can use the ``Debug`` class that is available in Orion. This class is used to log out the output of the different debugging processes.

Python object memory usage
--------------------------

This is for interest if one want to monitor memory usage of recognized python object inside the code. 

.. code-block:: python

    Debug(loggername = 'memory_usage.log').memory_usage()

Breakpoints
-----------

This is easily doable with IDEs such as VSCode, Pycharm...etc. However, if for any reason one can't use an IDE, it's possible to manually set breakpoint using a built-in interactive debugger module named ``pdb`` that allows to inspect and debug Python code during execution. In Orion one has to do:

.. code-block:: python

    Debug(loggername = 'breakpoint.log').breakpoint()

Which will start an interactive debugging session at the line where the ``breakpoint`` method is called. The output of the session are redirected to a log file. Bellow is an example of a debugging session:

.. image:: _static\\pictures\\debug_session.png
