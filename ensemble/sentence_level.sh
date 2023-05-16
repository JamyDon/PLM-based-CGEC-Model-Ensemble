cd ../

MODELS=
SRC_FILE=
OUT_DIR=
NAME=
TGT_FILE=$OUT_DIR$NAME".tgt"
PLM_PATH=

cp ./ensemble/sentence_level.py ./ensemble.py

python3 ./ensemble.py --models_path $MODELS --output_path $TGT_FILE --plm_path $PLM_PATH

rm ./ensemble.py