import face_recognition
import glob
import cv2
import os


def read_employee_in_folder(folder_path):
    employees = {}
    for img in glob.glob(folder_path + '*.jpg'):  # All jpeg images
        employee_name = os.path.splitext(os.path.basename(img))[0]
        employee_image = face_recognition.load_image_file(img)
        employees[employee_name] = employee_image
    return employees


def get_face_encodings(images):
    face_encoding_stack = []
    for img in images:
        face_encoding_stack.append(face_recognition.face_encodings(img)[0])
    return face_encoding_stack


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
    min_distance = min(face_distances)
    # print(min_distance)
    match_point = min_distance
    if match_point > tolerance:
        match_point = tolerance
    return list(face_distances <= match_point)


# Load a sample picture and learn how to recognize it.
enclave_employees = read_employee_in_folder("/home/terry/Documents/ML/enclave-employee-images/")
enclave_employees_name = list(enclave_employees.keys())
enclave_employees_image = list(enclave_employees.values())
enclave_employees_face_encoding = get_face_encodings(enclave_employees_image)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
i = 0

while True:
    folder_path = input("Enter path: ")
    print("Processing " + folder_path)

    for img in glob.glob(folder_path + '*.jpg'):  # All jpeg images
        print("Processing image: " + img)
        unknown_image = cv2.imread(img)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = compare_faces(enclave_employees_face_encoding, face_encoding)
            name = "Unknown"

            for idx in range(0, len(matches)):
                if matches[idx]:
                    name = enclave_employees_name[idx]
                    break

            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(unknown_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(unknown_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        out_dir = folder_path + "out/"
        print("Write image to: " + out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_dir + str(i) + ".jpg", unknown_image)
        i = i + 1

    input_var = input("Do you want to continue (Y/N): ")
    if input_var.lower() == "y":
        continue
    else:
        break

