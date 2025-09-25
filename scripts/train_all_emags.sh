export PYTHONPATH="/mnt/nvme2n1p1/neurips_2025/AsymmetricGS"
export NERFBASELINES_REGISTER="asym_emags_spec.py"

nerfbaselines train --method asym_emags --data dataset/phototourism/brandenburg-gate --output output/phototourism_runs/brandenburg-gate --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/phototourism/sacre-coeur --output output/phototourism_runs/sacre-coeur --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/phototourism/trevi-fountain --output output/phototourism_runs/trevi-fountain --backend python --eval-all-iters 1000::1000

nerfbaselines train --method asym_emags --data dataset/on-the-go/corner-undistorted --output output/corner-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/fountain-undistorted --output output/fountain-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/mountain-undistorted --output output/mountain-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/patio-undistorted --output output/patio-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/spot-undistorted --output output/spot-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/patio_high-undistorted --output output/patio_high-undistorted --backend python --eval-all-iters 1000::1000

nerfbaselines train --method asym_emags --data dataset/robustnerf/android-undistorted --output output/android-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/robustnerf/crab2-undistorted --output output/crab2-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/robustnerf/statue-undistorted --output output/statue-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/robustnerf/yoda-undistorted --output output/yoda-undistorted --backend python --eval-all-iters 1000::1000

