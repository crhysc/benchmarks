# benchmarks
setup:
1) git submodule update --init --recursive
2) conda install -n base -c conda-forge mamba
3) pip install uv dvc
4) cd tc_supercon; dvc repro -f
5) cd alexandria; dvc repro -f

   
