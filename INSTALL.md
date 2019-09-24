
You need to retrieve the `kiss_pointing_model.py` and copy it in the `src` directory

```
cd src
svn export https://lpsc-secure.in2p3.fr/svn/NIKA/Processing/Labtools/JM/KISS/kiss_pointing_model.py
2to3 -w kiss_pointing_model.py
```


You must have the `libreadnika` compiled and setup a environement variable `NIKA_LIB_PATH` to point the directory containing the `libreadnikadata.so` file.

```bash
export NIKA_LIB_PATH=/data/KISS/NIKA_lib_AB_OB_gui/Readdata/C/
```

And to use the database, give the location of KISS data
```bash
export KISS_DATA=/data/KISS/Raw/nika2c-data3/KISS
```