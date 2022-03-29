python train.py -m SparseMLP \
                --wd 5e-4 \
                --lp SparseMLP.log \
                --lm 100 150 180 \
                --en SparseMLP \
                --lr 0.1 \
                --nt CubeNorm \
                --gpu 1 \
                --seed 4 \
                # --blocks-per-stage 5 \
                # --kernel-size 1 \
                # --stages 3 \
                # --if-save True \
                # --save-dir ./save_temp/ResNet20_plug4_1/
