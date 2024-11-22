export ORI_DIR="./traindata_p0.0_none/train_person/" \
export COATED_DIR="./traindata_p1.0_wanet_unconditional_s2.0_k128_removeeval/train_person/" \
export GENERATED_INSPECTED_DIR="./DIA_coated_blur_gen/" \

python binary_classifier.py --ori_dir $ORI_DIR \
--coated_dir $COATED_DIR \
--generated_inspected_dir $GENERATED_INSPECTED_DIR 