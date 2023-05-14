dataset=mosi
model=emt-dlfr
# run
mr=0.5
python run.py --datasetName $dataset --modelName $model --missing $mr \
--model_save_dir results/$dataset/$model/run/models \
--res_save_dir results/$dataset/$model/run/results \
--gpu_ids 0

