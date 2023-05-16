cd ../

MODELS=
SRC_FILE=
OUT_DIR=
NAME=
TEMP_OUTPUT_FILE=$OUT_DIR$NAME".out"
PLM_PATH=

cp ./ensemble/edit_level.py ./ensemble.py

python3 ./ensemble.py --models_path $MODELS --output_path $TEMP_OUTPUT_FILE --plm_path $PLM_PATH

rm ./ensemble.py
cd ./ensemble/
