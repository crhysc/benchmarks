#!/bin/bash
wget https://github.com/hyllios/utils/raw/refs/heads/main/models/supercond_modnet_model/DS-A.pk.bz2
wget https://github.com/hyllios/utils/raw/refs/heads/main/models/supercond_modnet_model/DS-B.pk.bz2
python -c "import pandas as pd; dsb = pd.read_pickle('DS-A.pk.bz2', compression='bz2'); dsb.to_csv('dataset1.csv')" 
python -c "import pandas as pd; dsb = pd.read_pickle('DS-B.pk.bz2', compression='bz2'); dsb.to_csv('dataset2.csv')" 
