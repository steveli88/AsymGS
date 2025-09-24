export PYTHONPATH="/mnt/nvme2n1p1/neurips_2025/AsymmetricGS"
export NERFBASELINES_REGISTER="asym_gsgs_spec.py"

nerfbaselines train --method asym_gsgs --data dataset/phototourism/brandenburg-gate --output output/phototourism_runs/brandenburg-gate --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_gsgs --data dataset/phototourism/sacre-coeur --output output/phototourism_runs/mip_splatting/sacre-coeur --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_gsgs --data dataset/phototourism/trevi-fountain --output output/phototourism_runs/mip_splatting/trevi-fountain --backend python --eval-all-iters 1000::1000

