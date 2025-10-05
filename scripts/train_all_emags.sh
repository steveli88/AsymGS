export PYTHONPATH="/mnt/nvme2n1p1/neurips_2025/AsymmetricGS"
export NERFBASELINES_REGISTER="asym_emags_spec.py"

nerfbaselines train --method asym_emags --data dataset/on-the-go/corner-undistorted --output output_emags/corner-undistorted --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/fountain-undistorted --output output_emags/fountain-undistorted --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/mountain-undistorted --output output_emags/mountain-undistorted --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/patio-undistorted --output output_emags/patio-undistorted --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/spot-undistorted --output output_emags/spot-undistorted --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/patio_high-undistorted --output output_emags/patio_high-undistorted --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000

nerfbaselines train --method asym_emags --data dataset/robustnerf/android-undistorted --output output_emags/android-undistorted --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/robustnerf/crab2-undistorted --output output_emags/crab2-undistorted --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/robustnerf/statue-undistorted --output output_emags/statue-undistorted --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/robustnerf/yoda-undistorted --output output_emags/yoda-undistorted --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000

nerfbaselines train --method asym_emags --data dataset/phototourism/brandenburg-gate --output output_emags/brandenburg-gate --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/phototourism/sacre-coeur --output output_emags/sacre-coeur --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/phototourism/trevi-fountain --output output_emags/trevi-fountain --set lambda_mul=0.1 --backend python --eval-all-iters 1000::1000
