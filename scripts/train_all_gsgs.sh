export PYTHONPATH="/mnt/nvme2n1p1/neurips_2025/AsymmetricGS"
export NERFBASELINES_REGISTER="asym_gsgs_spec.py"

nerfbaselines train --method asym_gsgs --data dataset/phototourism/brandenburg-gate --output output_gsgs/brandenburg-gate --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_gsgs --data dataset/phototourism/sacre-coeur --output output_gsgs/sacre-coeur --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_gsgs --data dataset/phototourism/trevi-fountain --output output_gsgs/trevi-fountain --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000

nerfbaselines train --method asym_gsgs --data dataset/on-the-go/corner-undistorted --output output_gsgs/corner-undistorted --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_gsgs --data dataset/on-the-go/fountain-undistorted --output output_gsgs/fountain-undistorted --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_gsgs --data dataset/on-the-go/mountain-undistorted --output output_gsgs/mountain-undistorted --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_gsgs --data dataset/on-the-go/patio-undistorted --output output_gsgs/patio-undistorted --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_gsgs --data dataset/on-the-go/spot-undistorted --output output_gsgs/spot-undistorted --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_gsgs --data dataset/on-the-go/patio_high-undistorted --output output_gsgs/patio_high-undistorted --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000

nerfbaselines train --method asym_gsgs --data dataset/robustnerf/android-undistorted --output output_gsgs/android-undistorted --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_gsgs --data dataset/robustnerf/crab2-undistorted --output output_gsgs/crab2-undistorted --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_gsgs --data dataset/robustnerf/statue-undistorted --output output_gsgs/statue-undistorted --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_gsgs --data dataset/robustnerf/yoda-undistorted --output output_gsgs/yoda-undistorted --set lambda_mul=1.0 --backend python --eval-all-iters 1000::1000

