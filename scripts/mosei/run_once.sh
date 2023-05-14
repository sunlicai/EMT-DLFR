dataset=mosei
model=emt-dlfr
# run_once
python run_once.py --datasetName $dataset --modelName $model \
   --model_save_dir results/$dataset/$model/run_once/models \
   --res_save_dir results/$dataset/$model/run_once/results \
   --KeyEval 'Loss(pred_m)' \
   --gpu_ids $1
echo "done!"

