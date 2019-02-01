cmd = """nsml run -d piqa-nfs -g 1 -e run_piqa.py --memory 16G --nfs-output -a ' \
    --fs nfs \
    --do_index \
    --data_dir data/docs \
    --predict_file %d:%d \
    --output_dir index/wiki/large \
    --index_file demo_%d-%d.hdf5 \
    --load_dir KR18816/piqa-nfs/132 \
    --iteration 1 \
    --parallel'"""


