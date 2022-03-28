python train.py -m ResNet56_modified \
                --wd 5e-4 \
                --lp ResNet56_separatable_4plug_ex1.log \
                --lm 100 150 180 \
                --en ResNet56_separatable_4plug_ex1 \
                --lr 0.1 \
                --nt CubeNorm \
                --gpu 1 \
                --seed 4 \
                --blocks-per-stage 5 \
                --kernel-size 1 \
                --stages 3 \
                # --if-save True \
                # --save-dir ./save_temp/ResNet20_plug4_1/
