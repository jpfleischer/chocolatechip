# Running

To convert your darknet weights file to an onnx model,
you need Docker installed.

all that is needed to be changed is the convert.sh:
change the first two variables, MODEL_NAME and MODEL_DIR.

Then run `make` and that is it!

it is expected that the .weights are immediately available in the
specified MODEL_DIR and not nested in any subdirectories.

there should also only be one .cfg file.

NOTE: if your model dir is NOT in the /data dir then you must change
the docker volume mount in the Makefile! change it from -v /data:/data to
-v /my/lovely/model/folder:/data
or whatever your folder is.

