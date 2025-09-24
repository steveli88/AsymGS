# On-the-go dataset
python onthego_preprocess.py --input dataset/on-the-go/corner --output dataset/on-the-go/corner-undistorted --size 504
python onthego_preprocess.py --input dataset/on-the-go/fountain --output dataset/on-the-go/fountain-undistorted --size 504
python onthego_preprocess.py --input dataset/on-the-go/mountain --output dataset/on-the-go/mountain-undistorted --size 504
python onthego_preprocess.py --input dataset/on-the-go/patio --output dataset/on-the-go/patio-undistorted --size 480
python onthego_preprocess.py --input dataset/on-the-go/spot --output dataset/on-the-go/spot-undistorted --size 504
python onthego_preprocess.py --input dataset/on-the-go/patio_high --output dataset/on-the-go/patio_high-undistorted --size 504

# RobustNeRF dataset
colmap image_undistorter --image_path dataset/android/images --input_path dataset/android/sparse/0 --output_path dataset/android-undistorted --output_type COLMAP --max_image_size 504
cp dataset/android/train_list.txt dataset/android-undistorted/train_list.txt
cp dataset/android/test_list.txt dataset/android-undistorted/test_list.txt

colmap image_undistorter --image_path dataset/crab2/images --input_path dataset/crab2/sparse/0 --output_path dataset/crab2-undistorted --output_type COLMAP --max_image_size 432
cp dataset/crab2/train_list.txt dataset/crab2-undistorted/train_list.txt
cp dataset/crab2/test_list.txt dataset/crab2-undistorted/test_list.txt

colmap image_undistorter --image_path dataset/statue/images --input_path dataset/statue/sparse/0 --output_path dataset/statue-undistorted --output_type COLMAP --max_image_size 504
cp dataset/statue/train_list.txt dataset/statue-undistorted/train_list.txt
cp dataset/statue/test_list.txt dataset/statue-undistorted/test_list.txt

colmap image_undistorter --image_path dataset/yoda/images --input_path dataset/yoda/sparse/0 --output_path dataset/yoda-undistorted --output_type COLMAP --max_image_size 432
cp dataset/yoda/train_list.txt dataset/yoda-undistorted/train_list.txt
cp dataset/yoda/test_list.txt dataset/yoda-undistorted/test_list.txt

