.. include:: references.rst

Welcome to Orion's documentation!
===================================

.. figure:: _static\\logo\\orion_logo.png
   :align: left  
   :scale: 20%   
   :alt: Description of the image
      
This tool serves ...

Orion usage principle
---------------------
There is a list of classes and, within each class, a list of available methods. The main use of the code is to pass arguments to a class and apply a method to it, in order to create or manipulate datasets.

.. code-block:: python

   # Read 
   Reader(<class arguments>).method(<method arguments>)
   # Manipulate data
   base = class(<class arguments>).method(<method arguments>)
   # Write
   Writer(<class arguments>).method(<method arguments>)

Additionally on can also plot results using |PlotltyGithub| with the class :python:`Plot`.

Documentation tree
------------------
.. dropdown:: Installation & Execution
         :color: primary
         :icon: rocket

         .. toctree::
            :maxdepth: 3

            installation
            execution

.. dropdown:: Debugger & Performance
         :color: primary
         :icon: cpu

         .. toctree::
            :maxdepth: 3

            debug_performance

.. dropdown:: Examples of application
         :color: primary
         :icon: book

         .. toctree::
            :maxdepth: 3

            examples

.. dropdown:: Classes and Methods
         :color: primary
         :icon: code

         .. toctree::
            :maxdepth: 4

            CodeSource
         
         .. toctree::
            :maxdepth: 2

            formulas
         
.. dropdown:: Tips
         :color: primary
         :icon: zap

         .. toctree::
            :maxdepth: 3

            ips

Work directory tree
-------------------

.. raw:: html

      <div class="tree">
      <li><i class="fa fa-folder-open"></i> Orion
         <ul>
            <li><i class="fa fa-folder"></i> bin
               <ul>
                  Python compilor, Oracle client...
                  <li><i class="fa fa-file-code"></i> Python</li>
               </ul>
            </li>
            Orion maturity level (Development/Testing/Production)
            <li><i class="fa fa-folder-open"></i> Testing
               <ul>
                  Master script
                  <li><i class="fa fa-file-code-o"></i> Orion.py</li> 
                  <li><i class="fa fa-file-text"></i> README.md</li>
                  <li><i class="fa fa-file-text"></i> Read_The_Doc.html</li>
                  Visual Studio Code: configuration files
                  <li><i class="fa fa-folder-open"></i> .vscode
                     <ul>
                        <li><i class="fa fa-file-code"></i> .env</li>
                        <li><i class="fa fa-file-code"></i> settings.json</li>
                     </ul>
                  </li>
                  Orion core features
                  <li><i class="fa fa-folder-open"></i> Core
                     <ul>
                        <li><i class="fa fa-file-code-o"></i> Base.py</li>
                        <li><i class="fa fa-file-code-o"></i> Debug.py</li>
                        <li><i class="fa fa-file-code-o"></i> Formulas.py</li>
                        <li><i class="fa fa-file-code-o"></i> Plotter.py</li>
                        <li><i class="fa fa-file-code-o"></i> Reader.py</li>
                        <li><i class="fa fa-file-code-o"></i> ScriptParser.py</li>
                        <li><i class="fa fa-file-code-o"></i> SharedMethods.py</li>
                        <li><i class="fa fa-file-code-o"></i> Writer.py</li>
                     </ul>
                  </li>
                  Orion data processing methods
                  <li><i class="fa fa-folder-open"></i> DataProcessing
                     <ul>
                        <li><i class="fa fa-file-code-o"></i> DataProcessor.py</li>
                     </ul>
                  </li>
                  Sphinx module
                  <li><i class="fa fa-folder"></i> Documentation</li>
                  Orion's test units
                  <li><i class="fa fa-folder-open"></i> Test
                     <ul>
                        <li><i class="fa fa-file-code-o"></i> test_base.py</li>
                        <li><i class="fa fa-file-code-o"></i> test_reader.py</li>
                     </ul>
                  </li>
               </ul>
            </li>
         </ul>
      </li>
   </div>


References
----------

- Author/Developer : Toufik Saouchi
- External Modules references:
   - |sphinx_web_site| website
