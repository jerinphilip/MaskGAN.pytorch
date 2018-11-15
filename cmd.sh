module load use.own
module load python/3.7.0
python3 -m mgan.sandbox --path datasets/aclImdb/train/concat-filtered.txt --spm_path datasets/aclImdb/train/imdb.model
