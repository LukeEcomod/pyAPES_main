# Instructions how to contribute to documenting pyAPES main repository

## Install sphinx

using your favorite package or python environment manager install sphinx

pip

```terminal
pip install sphinx
```

poetry
```terminal
poetry add -D sphinx
```

## Add the python module you want to be documented to docs/index.rst

If you would like to document the canopy.py in folder Canopy you would write
```rst
Canopy
---------
.. automodule:: canopy.canopy
   :members:
   :special-members:
   :exclude-members: __weakref__
```
- \-\-\-\-\-\-\-\- creates a subsection
    - see from this reStructuredText [cheat sheet](https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst)
- automodule builds docstrings from the python file directly
- :members: means include all the classes and methods in the module
- :special-members: means include functions that are specified like \_\_special\_\_
- :exclude-members: means exclude specific members that are given as a list separated with commas
- More info about keywords: [sphix.ext.autodoc documentation](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)

## run make html

In terminal navigate to the docs folder and run 'make html

```terminal
>> cd path/to/pyAPES_main/docs
>> make html
Running Sphinx v4.4.0
loading pickled environment... done
building [mo]: targets for 0 po files that are out of date
building [html]: targets for 0 source files that are out of date
updating environment: 0 added, 1 changed, 0 removed
reading sources... [100%] index
looking for now-outdated files... none found
pickling environment... done
checking consistency... done
preparing documents... done
writing output... [100% ] index                                                                     
generating indices... genindex py-modindex done
writing additional pages... search done
copying static files... done
copying extra files... done
dumping search index in English (code: en)... done
dumping object inventory... done
build succeeded.
```

## Inspect the html pages
The html file index.html will be generated in directory docs/_build/ you can open the file with your favorite web browser

If for some reason the build fails at this point try creating an empty docs/_build directory yourself and rerun ```make html```

## Correct errors

The ```make html``` command can result to errors or warnings they might look like this

```terminal
/path/to/pyAPES_main/canopy/canopy.py:docstring of canopy.canopy.CanopyModel.run:4: ERROR: Unexpected indentation.
/path/to/pyAPES_main/canopy/canopy.py:docstring of canopy.canopy.CanopyModel.run:20: WARNING: Block quote ends without a blank line; unexpected unindent.
```

Navigate to the lines where the errors come from and correct them

## Modify docstrings

This project has been set to use Google style docstrings with the ```sphinx.ext.napoleon``` extension. This means that some of the docstrings in the python files might have to be edited. See an example [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) about the correct syntax for Google docstrings

## Push to GitHub

Once you are happy with the generated module documentation push your contribution to GitHub

```terminal
git add .
git commit -m "Documented canopy module"
git push -u origin sphinx_documentation
````

**Note that at the moment we are not concerned about the visual layout of the generated html pages. The layout can be fixed once we have all the docstrings generated correctly.**
