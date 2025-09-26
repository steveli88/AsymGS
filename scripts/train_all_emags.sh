export PYTHONPATH="/mnt/nvme2n1p1/neurips_2025/AsymmetricGS"
export NERFBASELINES_REGISTER="asym_emags_spec.py"

nerfbaselines train --method asym_emags --data dataset/phototourism/brandenburg-gate --output output_emags/brandenburg-gate --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/phototourism/sacre-coeur --output output_emags/sacre-coeur --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/phototourism/trevi-fountain --output output_emags/trevi-fountain --backend python --eval-all-iters 1000::1000

nerfbaselines train --method asym_emags --data dataset/on-the-go/corner-undistorted --output output_emags/corner-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/fountain-undistorted --output output_emags/fountain-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/mountain-undistorted --output output_emags/mountain-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/patio-undistorted --output output_emags/patio-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/spot-undistorted --output output_emags/spot-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/on-the-go/patio_high-undistorted --output output_emags/patio_high-undistorted --backend python --eval-all-iters 1000::1000

nerfbaselines train --method asym_emags --data dataset/robustnerf/android-undistorted --output output_emags/android-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/robustnerf/crab2-undistorted --output output_emags/crab2-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/robustnerf/statue-undistorted --output output_emags/statue-undistorted --backend python --eval-all-iters 1000::1000
nerfbaselines train --method asym_emags --data dataset/robustnerf/yoda-undistorted --output output_emags/yoda-undistorted --backend python --eval-all-iters 1000::1000

