import os
from ultralytics import YOLO
import cv2
import string
import easyocr
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime
import random

customer=0
membership=0

dict_char_to_int = {'O': '0',
                    'U': '0',
                    'I': '1',
                    'J': '3',
                    'L': '4',
                    'G': '6',
                    'S': '5',
                    'Z': '2',
                    'E': '8'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'L',
                    '6': 'G',
                    '5': 'S',
                    '2': 'Z'}


def write_csv(results, output_path):

    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{}\n'.format('car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for car_id in results.keys():
            print(results[car_id])
            if 'car' in results[car_id].keys() and \
               'license_plate' in results[car_id].keys() and \
               'text' in results[car_id]['license_plate'].keys():
                f.write('{},{},{},{},{},{}\n'.format(car_id,
                                                        '[{} {} {} {}]'.format(
                                                            results[car_id]['car']['bbox'][0],
                                                            results[car_id]['car']['bbox'][1],
                                                            results[car_id]['car']['bbox'][2],
                                                            results[car_id]['car']['bbox'][3]),
                                                        '[{} {} {} {}]'.format(
                                                            results[car_id]['license_plate']['bbox'][0],
                                                            results[car_id]['license_plate']['bbox'][1],
                                                            results[car_id]['license_plate']['bbox'][2],
                                                            results[car_id]['license_plate']['bbox'][3]),
                                                        results[car_id]['license_plate']['bbox_score'],
                                                        results[car_id]['license_plate']['text'],
                                                        results[car_id]['license_plate']['text_score'])
                        )
    f.close()

def check_customer_membership(file_path, formatted_text):
    df = pd.read_csv(file_path)

    if formatted_text in df['Customer'].values:
        customer = 1
        membership = df.loc[df['Customer'] == formatted_text, 'Membership'].values[0]
    else:
        customer = None
        membership = None

    return customer, membership


def format_license(text):

    license_plate_ = ''

    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_char_to_int, 3: dict_char_to_int, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int, 9: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if j < len(text):
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

    return license_plate_


reader = easyocr.Reader(['en'], gpu=False)

store_results = {}

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO(r'C:\Users\Dell\Desktop\CV_Project\runs\detect\train13\weights\last.pt')

# load image
# img = cv2.imread(r'C:\Users\Dell\Desktop\innovation lab\Screenshot (150).png')
# below for demo
# img = cv2.imread(r'C:\Users\Dell\Downloads\IMG_20240414_182309.jpg')

# img = cv2.imread(r'C:\Users\Dell\Pictures\Camera Roll\WIN_20240414_18_26_50_Pro.jpg')
# ************
# img = cv2.imread(r'C:\Users\Dell\Downloads\PXL_20240414_130051501.jpg')

# img = cv2.imread(r'C:\Users\Dell\Pictures\Camera Roll\WIN_20240414_18_28_33_Pro.jpg')

# IMG_20240414_182256.jpg
img = cv2.imread(r'C:\Users\Dell\Downloads\IMG_20240414_182256.jpg')


# img=cv2.resize(img,(2048,2048))
vehicles = [2, 3, 5, 7]

# detect vehicles

detections = coco_model(img)[0]

detections_ = []

car_bbox = []
for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection
    if int(class_id) in vehicles:
        detections_.append([x1, y1, x2, y2, score])
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        car_bbox.append(((x1, y1), (x2, y2)))

folder_path_save = r'C:\Users\Dell\Desktop\CV_Project\Final_images_car_numplate'

# detect license plates
license_plates = license_plate_detector(img)[0]
for idx, license_plate in enumerate(license_plates.boxes.data.tolist()):
    curr_time = int(time.time())

    x1, y1, x2, y2, score, class_id = license_plate
    if int(class_id) == 0:
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        # cv2.imshow('frame',img)
        # cv2.imwrite(f'detect_img_{idx}.jpg',img)

        file_path_save = os.path.join(folder_path_save, f'detect_img_{curr_time}_{idx}.jpg')
        cv2.imwrite(file_path_save, img)

        # import matplotlib.pyplot as plt

        # Display the image using matplotlib
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        # cv2.waitKey(0)

    # assign license plate to car
        car_idx = -1
        for j in range(len(car_bbox)):
            top_corner, bottom_corner = car_bbox[j]

            if x1 > top_corner[0] and y1 > top_corner[1] and x2<bottom_corner[0] and y2 < bottom_corner[1]:
                car_idx = j
                break

        # crop license plate
        license_plate_crop = img[y1:y2, x1: x2, :]
        # license_plate_crop = cv2.fastNlMeansDenoisingColored(license_plate_crop,None,10,10,7,21)

        # process license plate
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255,
                                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                                          12)

        # cv2.imshow('frame', license_plate_crop_thresh)
        # res = cv2.resize(license_plate_crop,(192,96))
        # import matplotlib.pyplot as plt

        # Display the image using matplotlib
        plt.imshow(license_plate_crop_thresh, cmap='gray')
        plt.axis('off')
        plt.show()

        cv2.imwrite(f'numplate_{idx}.jpg',license_plate_crop)
        file_path_save = os.path.join(folder_path_save, f'numplate_{curr_time}_{idx}.jpg')
        cv2.imwrite(file_path_save, license_plate_crop_thresh)
        # cv2.waitKey(0)
        # nimg=cv2.imread(r'C:\Users\Dell\Downloads\numplate_blur.png')
        license_number_detections = reader.readtext(license_plate_crop_thresh)

        unformatted_text = ''

        for license_number_detection in license_number_detections:
            bbox, unformatted_text, text_score = license_number_detection
            # print(unformatted_text)
           

            unformatted_text = unformatted_text.upper().replace(' ', '')
            unformatted_text = unformatted_text.upper().replace('.', '')
            # unformatted_text = unformatted_text.upper().replace('\'', '')
            unformatted_text = unformatted_text.upper().replace('(', '')


            # print(unformatted_text)

            unformatted_text = unformatted_text[0:10]
            if len(unformatted_text) <= 10:
                formatted_text = format_license(unformatted_text)
            else:
                formatted_text = unformatted_text
            print(formatted_text)

            file_path = 'cust_memb.csv'
            customer, membership = check_customer_membership(file_path, formatted_text)






            if car_idx != -1:
                store_results[car_idx] = {'car': {'bbox': [top_corner[0], top_corner[1], bottom_corner[0], bottom_corner[1]]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': formatted_text,
                                                                'bbox_score': score,
                                                                'text_score': text_score}}

# print(store_results)
# write results
def append_csv(results, output_path):
    with open(output_path, 'a') as f:
        for car_id in results.keys():
            if 'car' in results[car_id].keys() and \
               'license_plate' in results[car_id].keys() and \
               'text' in results[car_id]['license_plate'].keys():
                f.write('{},{},{},{},{},{}\n'.format(car_id,
                                                     '[{} {} {} {}]'.format(
                                                         results[car_id]['car']['bbox'][0],
                                                         results[car_id]['car']['bbox'][1],
                                                         results[car_id]['car']['bbox'][2],
                                                         results[car_id]['car']['bbox'][3]),
                                                     '[{} {} {} {}]'.format(
                                                         results[car_id]['license_plate']['bbox'][0],
                                                         results[car_id]['license_plate']['bbox'][1],
                                                         results[car_id]['license_plate']['bbox'][2],
                                                         results[car_id]['license_plate']['bbox'][3]),
                                                     results[car_id]['license_plate']['bbox_score'],
                                                     results[car_id]['license_plate']['text'],
                                                     results[car_id]['license_plate']['text_score'])
                        )
    f.close()

append_csv(store_results, './test1.csv')



def predict_dynamic_price(file_path, live_file, slot_number):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Extract the current day of the week (1 for Monday, 2 for Tuesday, ..., 7 for Sunday)
    current_day_of_week = datetime.now().weekday() + 1

    # Extract the current hour
    current_hour = datetime.now().hour

    # Randomly assign IsSpecialDay as 0 or 1
    is_special_day = random.randint(0, 1)

    # Prompt the user to enter the slot number
    print("Enter the slot number:", slot_number)

    # Extract the row corresponding to the input slot number from the end of the live file
    last_5_rows = pd.read_csv(live_file).tail(5)
    input_row = last_5_rows[last_5_rows.iloc[:, 3] == slot_number].iloc[0]

    # Assign the required features
    input_row['DayOfWeek'] = current_day_of_week
    input_row['TimeOfDay'] = current_hour
    input_row['IsSpecialDay'] = is_special_day

    # Extract Reservations, TotalAvailableSpots, and PercentageAvailableSpots from the row
    reservations = input_row.iloc[4]
    total_available_spots = input_row.iloc[5]
    percentage_available_spots = input_row.iloc[6]

    # Set CustomerType and MembershipStatus to 1
    input_row['CustomerType'] = customer
    input_row['MembershipStatus'] = membership

    # Reshape the input row to match the model's input shape
    input_row = input_row[['DayOfWeek', 'TimeOfDay', 'IsSpecialDay', 'SlotNumber','Reservations', 'TotalAvailableSpots', 'PercentageAvailableSpots', 'CustomerType', 'MembershipStatus']].values.reshape(1, -1)
    print(input_row)
    # Load the trained linear regression model
    model = LinearRegression()

    # Load the trained model
    X = df[['DayOfWeek', 'TimeOfDay', 'IsSpecialDay', 'SlotNumber','Reservations', 'TotalAvailableSpots', 'PercentageAvailableSpots', 'CustomerType', 'MembershipStatus']]
    y = df['DynamicPrice']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X, y)
    # y_pred = model.predict(X_test)
    
    # Predict the price for the input row
    predicted_price = abs(model.predict(input_row))

    print(f"Predicted Price for Slot {slot_number}: Rs.{predicted_price[0]:.2f}")

# Test the function
file_path = 'slot_data_train.csv'
live_file = 'slot_data.csv'
slot_number = int(input("Enter the slot number: "))
predict_dynamic_price(file_path, live_file, slot_number)



# 
# uncomment below
# 







# # writing changes to image

# final_results = pd.read_csv('./test1.csv')

# for car_id in np.unique(final_results['car_id']):

#     car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(final_results[final_results['car_id'] == car_id]['car_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
#     x1, y1, x2, y2 = ast.literal_eval(final_results[final_results['car_id'] == car_id]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

#     license_plate_image = img[int(y1):int(y2), int(x1):int(x2), :]
#     license_plate_image = cv2.resize(license_plate_image, (int((x2 - x1) * 400 / (y2 - y1)), 400))

#     H, W, _ = license_plate_image.shape
#     print(license_plate_image.shape)
#     print(img.shape)

#     try:

#         (text_width, text_height), _ = cv2.getTextSize(
#             final_results[final_results['car_id'] == car_id]['license_plate_number'],
#             cv2.FONT_HERSHEY_SIMPLEX,
#             4.3,
#             17)

#         cv2.putText(img,
#                     final_results[final_results['car_id'] == car_id]['license_plate_number'],
#                     (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
#                     cv2.FONT_HERSHEY_COMPLEX,
#                     4.3,
#                     (0, 0, 0),
#                     17)

#     except:
#         print("error")
#         pass

#     cv2.imshow('img', img)
#     cv2.waitKey(0)








