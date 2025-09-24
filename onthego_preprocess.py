import logging
import os
import json


def extract_train_test_split(original_path):
    with open(os.path.join(original_path, 'transforms.json'), 'r') as f:
        transforms_data = json.load(f)
        strip_prefix = lambda x: x.split('/')[-1]  # noqa
        image_names = [strip_prefix(x["file_path"]) for x in transforms_data["frames"]]
        image_names = sorted(image_names)
    with open(os.path.join(original_path, "split.json"), 'r') as f:
        split_data = json.load(f)
    train_list = [image_names[x] for x in split_data['clutter']]
    test_list = [image_names[x] for x in split_data['extra']]
    assert len(train_list) > 0, 'No training images found'
    assert len(test_list) > 0, 'No test images found'
    return train_list, test_list


def prepare_directorys(output):
    database_path = os.path.join(output, 'original', 'database.db')
    map_output_path = os.path.join(output, 'original', 'sparse')
    if not os.path.exists(map_output_path):
        os.makedirs(map_output_path)
    open(database_path, 'a').close()
    return database_path, map_output_path


def preprocess_nerfonthego_dataset(path, output_path, max_image_size):
    os.makedirs(os.path.join(output_path, 'original'), exist_ok=True)

    # First, we create the train/test splits
    train_list, test_list = extract_train_test_split(path)
    with open(os.path.join(output_path, 'train_list.txt'), 'w') as f:
        for item in train_list:
            f.write("%s\n" % item)
    with open(os.path.join(output_path, 'test_list.txt'), 'w') as f:
        for item in test_list:
            f.write("%s\n" % item)

    database_path, map_output_path = prepare_directorys(output)

    # Next, we run COLMAP to extract the features and matches
    logging.info("Running feature extractor")
    os.system(f"""colmap feature_extractor \
                    --database_path "{database_path}" \
                    --image_path "{os.path.join(path, 'images')}" \
                    --ImageReader.camera_model SIMPLE_RADIAL \
                    --ImageReader.single_camera 1 \
                    --SiftExtraction.use_gpu 1""")

    logging.info("Running exhaustive feature matcher")
    os.system(f"""colmap exhaustive_matcher \
                    --database_path "{database_path}" \
                    --SiftMatching.use_gpu 1""")

    logging.info("Running mapper")
    os.system(f"""colmap mapper \
                    --database_path "{database_path}" \
                    --image_path "{os.path.join(path, 'images')}" \
                    --output_path "{map_output_path}" """)

    # Undistort images
    logging.info("Undistorting images")
    os.system(f"""colmap image_undistorter \
                    --image_path "{os.path.join(path, 'images')}" \
                    --input_path "{os.path.join(map_output_path, '0')}" \
                    --output_path "{output_path}" \
                    --output_type COLMAP \
                    --max_image_size {max_image_size}""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="On-the-go processing")

    parser.add_argument("--input", default="tree", type=str, help="path to input")
    parser.add_argument("--output", default="tree-undistorted", type=str, help="path to output")
    parser.add_argument('--size', default=504, type=int) # 480 or 504
    args = parser.parse_args()

    # path = os.path.join("dataset", "on-the-go", args.input)
    # output = os.path.join("dataset", "on-the-go", args.output)
    # max_image_size = args.size

    path = args.input
    output = args.output
    max_image_size = args.size

    preprocess_nerfonthego_dataset(path, output, max_image_size)

