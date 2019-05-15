You must have the `libreadnika` compiled and setup a environement variable `NIKA_LIB_PATH` to point the directory containing the `libreadnikadata.so` file.

You need to retrieve the `kiss_pointing_model.py` and copy it in the `src` directory

```
cd src
svn export https://lpsc-secure.in2p3.fr/svn/NIKA/Processing/Labtools/JM/KISS/kiss_pointing_model.py
2to3 -w kiss_pointing_model.py
```
