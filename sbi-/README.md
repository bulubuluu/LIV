# Train the model

```
#!/bin/bash

cd /eos/user/y/yzhang4/LIV/LO/
source venv/bin/activate
cd /eos/user/y/yzhang4/LIV/sbi-/sessions/day2/nsbi-tutorial
python3 -m nsbi.carl fit \
  --data.numerator_events /eos/user/y/yzhang4/LIV/split-Z/files/Reconstructed_Z1Z2_LHEF_4e-6.root \
  --data.features '["Z1_PT","Z1_Eta","Z1_Phi","Z1_Mass","Z2_PT","Z2_Eta","Z2_Phi","Z2_Mass"]' \
  --data.batch_size 1024 \
  --data.data_dir run/liv_over_bkg \
  --model.learning_rate 1e-5 \
  --model.n_layers 16 \
  --model.n_nodes 1024 \
  --trainer.devices 1 \
  --trainer.max_epochs 500 \
  --trainer.accelerator cpu \
```

# Plot
```
cd /eos/user/y/yzhang4/LIV/LO/
source venv/bin/activate
cd -
```

```
cd /eos/user/y/yzhang4/LIV/sbi-/sessions/day2/nsbi-tutorial
python3 1.py
```
