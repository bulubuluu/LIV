#!/usr/bin/env bash

SEED=$((RANDOM % 101))

# mkdir -p sig_over_bkg && cd sig_over_bkg
# python -m nsbi.carl fit \
#     --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
#     --data.numerator_events '/global/cfs/cdirs/trn016/h4l_data/sm/ggzz4l_sig.csv' \
#     --data.denominator_events '/global/cfs/cdirs/trn016/h4l_data/sm/ggzz4l_bkg.csv' \
#     --data.batch_size 1024 \
#     --model.learning_rate 1e-4 \
#     --model.n_layers 16 \
#     --model.n_nodes 1024 \
#     --trainer.devices 1 \
#     --trainer.max_epochs 500 \
#     --seed_everything $SEED
# cd ..

# mkdir -p sbi_over_bkg && cd sbi_over_bkg
# python -m nsbi.carl fit \
#     --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
#     --data.numerator_events '/global/cfs/cdirs/trn016/h4l_data/sm/ggzz4l_sbi.csv' \
#     --data.denominator_events '/global/cfs/cdirs/trn016/h4l_data/sm/ggzz4l_sbi.csv' \
#     --data.denominator_reweight '["sbi","bkg"]' \
#     --data.batch_size 1024 \
#     --model.learning_rate 1e-5 \
#     --model.n_layers 16 \
#     --model.n_nodes 1024 \
#     --trainer.devices 1 \
#     --trainer.max_epochs 500 \
#     --seed_everything $SEED

cd /eos/user/y/yzhang4/LIV/LO/
source venv/bin/activate
# cd -
cd /eos/user/y/yzhang4/LIV/sbi-/sessions/day2/nsbi-tutorial

python3 -m nsbi.carl fit \
  --data.features '["Z1_PT","Z1_Eta","Z1_Phi","Z1_Mass","Z2_PT","Z2_Eta","Z2_Phi","Z2_Mass"]' \
  --data.numerator_events /eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_4e-6/result/total.root \
  --data.batch_size 1024 \
  --data.data_dir run/liv_over_bkg \
  --model.learning_rate 1e-5 \
  --model.n_layers 16 \
  --model.n_nodes 1024 \
  --trainer.devices 1 \
  --trainer.max_epochs 500 \
  --trainer.accelerator cpu \

python3 -m nsbi.carl fit \
  --data.numerator_events /eos/user/y/yzhang4/LIV/split-Z/files/Reconstructed_Z1Z2_LHEF_4e-6.root \
  --data.features '["Z1_PT","Z1_Eta","Z1_Phi","Z1_Mass","Z2_PT","Z2_Eta","Z2_Phi","Z2_Mass"]' \
  --data.batch_size 256 \
  --data.data_dir run/debug_root \
  --model.learning_rate 1e-4 \
  --model.n_layers 2 \
  --model.n_nodes 64 \
  --trainer.max_epochs 1 \
  --trainer.accelerator auto \
  --trainer.devices 1

