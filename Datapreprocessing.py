import numpy as np
import os
import sys
import json
import cv2 as cv2

SUCCESS = 1; FAILURE = -1; COMPLETE = 0;

class Datapreprocessing():
    def __init__(self):
        return

    "Load the training dataset"
    def load_train_dataset(self, filename):
        if len(os.listdir(filename)) == 0:
            print("Directory is empty"); return
        else:
            x = np.array(os.listdir(filename))
            test_str = x[500]
            #print(int(test_str[:-4]))   # Convert the string to int-id

        return x

    "Load the training dataset in JSON"
    def load_gnd_truth_dataset(self,path,filename):
        if len(os.listdir(path)) == 0:
            print("Directory is empty"); return
        else:
            with open(filename) as json_file:
                gnd_truth_dataset = json.load(json_file)
                #print(data.keys())
                #print(data['annotations'][5]['bbox'])
                #print(data['annotations'][5]['image_id'])
                #print(data['annotations'][5]['category_id'])
                #print(data['annotations'][5]['id'])
                #print(data['info'])
                #print(data['licenses'])
                #print(data['annotations'])
                #print(data['categories'])
            return gnd_truth_dataset

    "Load the image with bounding box for testing"
    def load_img(self, path, image_id, bbox):
        img = cv2.imread(os.path.join(path, image_id), 1)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0])+int(bbox[2]), int(bbox[1])+int(bbox[3])), (255,0,0), 2)
        cv2.imshow("Image",img); cv2.waitKey(0); cv2.destroyAllWindows()


    def train_img_index(self, filename, image_id):
        x = np.array(os.listdir(filename))
        #x_train_index = [ind for ind, elems in enumerate(x) if image_id == int(elems[:-4])]

        #x_train_index = [obj for elems in image_id if elems == int()]
        print(image_id.shape[0], x.shape)
        x_train_index = []
        for img_id in range(0,image_id.shape[0],1000):
            for obj in range(x.shape[0]):
                if image_id[img_id] == int(x[obj][:-4]):
                    x_train_index.append(obj)
                print(img_id)

        print(np.array(x_train_index).shape)
        print(x_train_index)


        index = 0
        for elems in x:
            image_train_id = int(elems[:-4])
            if image_train_id == image_id:
                return index
            index += 1
        return FAILURE

    "Returns a new list containing only person with bounding box"
    def train_img_category_person(self, path, gnd_json, category_id, filename):
        x = self.load_gnd_truth_dataset(path=path, filename=gnd_json)
        x_image_id = [elems['image_id'] for elems in x['annotations'] if elems['category_id'] == category_id]
        x_bbox = [elems['bbox'] for elems in x['annotations'] if elems['category_id'] == category_id]
        x_segmentation = [elems['segmentation'] for elems in x['annotations'] if elems['category_id'] == category_id]
        x_seg_area = [elems['area'] for elems in x['annotations'] if elems['category_id'] == category_id]
        x_category = [elems['category_id'] for elems in x['annotations'] if elems['category_id'] == category_id]

        x_image_id = np.array(x_image_id); x_bbox = np.array(x_bbox);
        x_segmentation = np.array(x_segmentation); x_seg_area = np.array(x_seg_area); x_category = np.array(x_category)

        x_train = self.load_train_dataset(filename=filename)

        x_train_test = np.array([int(elems[:-4]) for elems in x_train])

        img_id_list = []
        img_bbox_list = []
        img_segmentation = []
        img_seg_area = []
        img_category = []


        for i in range(x_train_test.shape[0]):
            img_id_list_test = np.where(x_image_id == x_train_test[i])
            img_id_list.append(img_id_list_test)

        for i in range(x_train_test.shape[0]):
            annotations_lst = [x_bbox[annotations] for annotations in img_id_list[i]]
            img_segmentation_list_test = [x_segmentation[annotations] for annotations in img_id_list[i]]
            img_seg_area_list_test = [x_seg_area[annotations] for annotations in img_id_list[i]]
            img_category_list = [x_category[annotations] for annotations in img_id_list[i]]

            img_bbox_list.append(annotations_lst)
            img_segmentation.append(img_segmentation_list_test)
            img_seg_area.append(img_seg_area_list_test)
            img_category.append(img_category_list)

        img_id_list = np.array(img_id_list)
        img_bbox_list = np.array(img_bbox_list)
        img_segmentation = np.array(img_segmentation)
        img_seg_area = np.array(img_seg_area)
        img_category = np.array(img_category)
        #print(img_id_list.shape, img_bbox_list.shape, img_segmentation.shape, img_seg_area.shape)

        return img_id_list, img_bbox_list, img_segmentation, img_seg_area, img_category

    def disp_img_bounding_boxes(self, img_path, img_train, img_bbox, img_segm, show_segm=True, show_bbox=True):
        target_height, target_width = 448, 448
        for i in range(0,len(img_train),1):
            img = cv2.imread(os.path.join(img_path, img_train[i]))
            scale_width = (img.shape[1]/target_width)
            scale_height = (img.shape[0]/target_height)
            img = cv2.resize(img, (int(img.shape[1]/scale_width), int(img.shape[0]/scale_height)), interpolation = cv2.INTER_CUBIC)

            if show_bbox:
                for j in range(len(img_bbox[i,0])):
                    x_top, y_top = int(img_bbox[i,0][j,0]/scale_width), int(img_bbox[i,0][j,1]/scale_height)

                    x_bottom, y_bottom = int((img_bbox[i,0][j,0]+img_bbox[i,0][j,2])/scale_width), int((img_bbox[i,0][j,1]+img_bbox[i,0][j,3])/scale_height)

                    cv2.rectangle(img, (x_top, y_top), (x_bottom, y_bottom), (255,0,0), 2)
                    cv2.putText(img, 'Person', (x_top, y_top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    #cv2.rectangle(img, (int(img_bbox[i,0][j,0]), int(img_bbox[i,0][j,1])), (int(img_bbox[i,0][j,0]+img_bbox[i,0][j,2]), int(img_bbox[i,0][j,1]+img_bbox[i,0][j,3])), (255,0,0), 2)

            if show_segm:
                for j in range(len(img_segm[i,0])):
                    tmp = np.array(img_segm[i,0][j])
                    if len(tmp.shape) == 0:
                        continue

                    if len(tmp.shape) < 2:
                        for segments in tmp:
                            for k in range(0,len(segments),2):
                                if k == len(segments)-1:
                                    _x, _y = int(segments[k]/scale_width), int(segments[k-1]/scale_height)
                                    cv2.circle(img, (_x, _y), 2, (0,255,255),2)
                                    break
                                else:
                                    _x, _y = int(segments[k]/scale_width), int(segments[k-1]/scale_height)
                                    cv2.circle(img, (_x, _y), 2, (0,255,255),2)
                    else:
                        for segments in tmp:
                            for k in range(0, len(segments),2):
                                if k == len(segments)-1:
                                    _x, _y = int(segments[k]/scale_width), int(segments[k-1]/scale_height)
                                    cv2.circle(img, (_x, _y), 2, (0,0,255),2)
                                    break
                                else:
                                    _x, _y = int(segments[k]/scale_width), int(segments[k-1]/scale_height)
                                    cv2.circle(img, (_x, _y), 2, (0,0,255),2)

            cv2.imshow('image_'+str(i), img)
            k = cv2.waitKey(0)

            if k == 27: # Escape to quit
                break
            else: # Spacebar to continue
                print("Image {}".format(i))
                cv2.destroyWindow(winname='image_'+str(i))


    def scroll_through_images(self, img_path, bbox, img_id):
        target_height, target_width = 227, 227

        img_cnt = 0
        while(img_cnt < 5000):
            _img_id = img_id[img_cnt]
            _bbox = bbox[img_cnt]

            for images in os.listdir(img_path):
                if int(images[:-4]) == _img_id:
                    img = cv2.imread(os.path.join(img_path, images), 1)

                    scale_width = (img.shape[1]/target_width)
                    scale_height = (img.shape[0]/target_height)

                    img = cv2.resize(img, (int(img.shape[1]/scale_width), int(img.shape[0]/scale_height)), interpolation = cv2.INTER_CUBIC)

                    x_top, y_top = int(_bbox[0]/scale_width), int(_bbox[1]/scale_height)

                    x_bottom, y_bottom = int(x_top+_bbox[2]/scale_width), int(y_top+_bbox[3]/scale_height)

                    cv2.rectangle(img, (x_top, y_top), (x_bottom, y_bottom), (255,0,0), 2)
                    #cv2.imshow("Image",img);
                    #k = cv2.waitKey(0)
            img_cnt += 1
            print(img_cnt)
        return COMPLETE


    "Preprocessing the specified datasets into appropriate format"
    def preprocessed_dataset(self, dataset):
        return COMPLETE

    "Preprocessing certain features of the dataset --> IMPLEMENT ADDITIONAL FEATURES"
    def preprocess_dataset_features(self):
        return COMPLETE

    "Split the preprocessed dataset into batches"
    def split_dataset(self, batch_nr):
        return COMPLETE

    "Normalize the dataset between 0 and 1"
    def normalize_0_1(self, dataset):
        return COMPLETE

    "Normalize the dataset between -1 and 1"
    def normalize_negative_1_to_positive_1(self, dataset):
        return COMPLETE

    "Normalize the dataset between 0 and infinite"
    def normalize_zero_to_infinite(self):
        return COMPLETE

    "Saving the newly formatted dataset"
    def save_preprocessed_datasets(self, preprocessed_dataset):
        return COMPLETE



def main(_):
    myClass = Datapreprocessing()
    train_dataset = myClass.load_train_dataset(filename="./Datasets/Val2017")
    gnd_truth_dataset = myClass.load_gnd_truth_dataset(filename="./Datasets/annotations/instances_val2017.json", path="./Datasets/annotations")

    x_img_id, x_bbox, x_segm, x_segm_area, x_category = myClass.train_img_category_person(path="./Datasets/annotations", gnd_json="./Datasets/annotations/instances_val2017.json", category_id=1, filename="./Datasets/Val2017")

    myClass.disp_img_bounding_boxes(img_path="./Datasets/Val2017",img_train=train_dataset ,img_bbox=x_bbox, img_segm=x_segm, show_segm=True, show_bbox=True)

    #test2 = myClass.train_img_index(filename="./Datasets/Val2017", image_id=x_img_id)

    #myClass.scroll_through_images(img_path="./Datasets/Val2017", bbox=x_bbox, img_id=x_img_id)

    #print(gnd_truth_dataset['images'][0])
    #test_img = myClass.load_img(path="./Datasets/Val2017", image_id=train_dataset[test2], bbox=x_bbox[9])

    return SUCCESS

if __name__ == '__main__':
    main(None)
