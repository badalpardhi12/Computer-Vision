from image_2d_box import get_velo_from_cam
import os

if __name__ == '__main__':
    valid_data_list_filename = "./valid_full_list.txt"
    basefilename = "./data/"
    new_dir_path = "./data/gt_boxes_velo/"

    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
        print("Created new dir")

    with open(valid_data_list_filename, "r") as f: 
        for line in f.readlines():
            line = line.split("\n")[0]
            gt_boxes_velo = get_velo_from_cam(line, basefilename)
            
            with open(os.path.join(new_dir_path, line + 'txt'), 'w') as fp:
                for i in range(gt_boxes_velo.shape[0]):
                    fp.write(str(gt_boxes_velo[i, 0]) + ", ")
                    fp.write(str(gt_boxes_velo[i, 1]) + ", ")
                    fp.write(str(gt_boxes_velo[i, 2]) + ", ")
                    fp.write(str(gt_boxes_velo[i, 3]) + ", ")
                    fp.write(str(gt_boxes_velo[i, 4]) + ", ")
                    fp.write(str(gt_boxes_velo[i, 5]) + "\n")
            print(line, gt_boxes_velo.shape)
            #filenames_list.append(line)