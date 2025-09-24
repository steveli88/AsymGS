# On-the-go dataset
python onthego_preprocess.py --input dataset/on-the-go/corner --output dataset/on-the-go/corner-undistorted --size 504
python onthego_preprocess.py --input dataset/on-the-go/fountain --output dataset/on-the-go/fountain-undistorted --size 504
python onthego_preprocess.py --input dataset/on-the-go/mountain --output dataset/on-the-go/mountain-undistorted --size 504
python onthego_preprocess.py --input dataset/on-the-go/patio --output dataset/on-the-go/patio-undistorted --size 480
python onthego_preprocess.py --input dataset/on-the-go/spot --output dataset/on-the-go/spot-undistorted --size 504
python onthego_preprocess.py --input dataset/on-the-go/patio_high --output dataset/on-the-go/patio_high-undistorted --size 504

# RobustNeRF dataset
colmap image_undistorter --image_path dataset/robustnerf/android/images --input_path dataset/robustnerf/android/sparse/0 --output_path dataset/robustnerf/android-undistorted --output_type COLMAP --max_image_size 504
cp dataset/robustnerf/android/train_list.txt dataset/robustnerf/android-undistorted/train_list.txt
cp dataset/robustnerf/android/test_list.txt dataset/robustnerf/android-undistorted/test_list.txt

colmap image_undistorter --image_path dataset/robustnerf/crab2/images --input_path dataset/robustnerf/crab2/sparse/0 --output_path dataset/robustnerf/crab2-undistorted --output_type COLMAP --max_image_size 432
cp dataset/robustnerf/crab2/train_list.txt dataset/robustnerf/crab2-undistorted/train_list.txt
cp dataset/robustnerf/crab2/test_list.txt dataset/robustnerf/crab2-undistorted/test_list.txt

colmap image_undistorter --image_path dataset/robustnerf/statue/images --input_path dataset/robustnerf/statue/sparse/0 --output_path dataset/robustnerf/statue-undistorted --output_type COLMAP --max_image_size 504
cp dataset/robustnerf/statue/train_list.txt dataset/robustnerf/statue-undistorted/train_list.txt
cp dataset/robustnerf/statue/test_list.txt dataset/robustnerf/statue-undistorted/test_list.txt

colmap image_undistorter --image_path dataset/robustnerf/yoda/images --input_path dataset/robustnerf/yoda/sparse/0 --output_path dataset/robustnerf/yoda-undistorted --output_type COLMAP --max_image_size 432
cp dataset/robustnerf/yoda/train_list.txt dataset/robustnerf/yoda-undistorted/train_list.txt
cp dataset/robustnerf/yoda/test_list.txt dataset/robustnerf/yoda-undistorted/test_list.txt

