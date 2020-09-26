# opencv
import cv2
# პარამეტრების შესაბამისად ცვლის ფოტოს ფორმას (3D, 4D...)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# მოდელის შექმნა , მოდელი არის ფენების ჯგუფი  ობიექტებად ქცეული
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# აბრუნებს 4D ფორმის მქონე ობიექტს
from tensorflow.keras.layers import AveragePooling2D
# აბრუნებს ბულიანის ტიპი ობიექტს  თუ რომელ რეჟიმშია ობიექტი training mode/inference mode
from tensorflow.keras.layers import Dropout
# იწყებს დონეების დავლას და დამუშავებას თუ რომელია პირველი და ბოლო დონე ,  და ახდენს ობიექტის იმპლემენტაციას
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
# ქმნის მოდელს თუ რომელი უნდა იყოს შეტანილი და გამოტანილი მოდელები
from tensorflow.keras.layers import Input
# გამოაქვს მოდელების ჩამონათვალი
from tensorflow.keras.models import Model
# ახდენს მოდელების  ოპტიმიზაციას
from tensorflow.keras.optimizers import Adam
# Numpy array-ს შესაქმნელად
from tensorflow.keras.preprocessing.image import img_to_array
# numpy ტიპის ფოტოების ჩასატვირთად/დასამუშავებლად
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
# მასივს აბრუნებს ორობითში
from tensorflow.keras.utils import to_categorical
# ყოფს მასივებსა და მატრიცებს
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# ფოტოების მისამართების ასაღებად
from imutils import paths
import os
# ობიექტების დაპლოტვა
import matplotlib.pyplot as plt
# ჩადგმული  მასივებისა და მატრიცების დამუშავებისთვის
import numpy as np

# img = cv2.imread("C:/Users/Giorgi/Desktop/bruno.jpg")

# ფუნქცია cvtColor საშუალებას გვაძლევს ფოტოს შევუცვალოთ ფერი RGB-any
# changedImg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

# cv2.imshow("changed", changedImg)
# cv2.waitKey(0)
# imgB = cv2.GaussianBlur(img, (15, 15), 0)
# cv2.imshow("ttt", imgB)
# cv2.imshow("images", img)
# img3 = cv2.Canny(img, 100, 100)
# cv2.imshow("2", img3)

# cv2.waitKey(0)


# cap = cv2.VideoCapture(0)
# # ვიდეოს ჩაწერისას ,რეზოლუცია ერთმანეთს უნდა ემთხვეოდეს !!
# cap.set(3, 640)
# cap.set(4, 480)
# # cap.set(10, 100)

# org = (420, 135)
# color = (255, 255, 255)
# fourc = cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G')
# დასაზუზსტებელაი რეზოლუცია , ამ ეტაპზე მხოლოდ ამ რეზოლუციით ინახავს ჩანაწერს
# out = cv2.VideoWriter("E:/python/qwert.avi", fourc, 20.0, (640, 480))
# while cap.isOpened():
#     suc, img = cap.read()
#     if suc == True:
#         img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
#         img = cv2.putText(img, 'Work...', org,
#                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#         out.write(img)
#         cv2.imshow("imgaes", img)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         # write the flipped frame
#         out.write(frame)

#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()


# # load cascade
# face_cascade = cv2.CascadeClassifier(
#     'E:/python/haarcascade_frontalface_default.xml')

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     suc, img = cap.read()
#     if suc == True:
#         # img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
#         # img = cv2.putText(img, 'Work...', org,
#         #                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#         # out.write(img)
#         # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(img, 1.2, 5)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             t = cv2.putText(img, "Giorgi", (x+30, y+h+30),
#                             cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
#         # gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#         cv2.imshow("imgaes", img)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()


# img = cv2.imread('E:/python/tet.jpg')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# for (x, y, w, h) in faces:
#     print("x", x)
#     print("y", y)
#     print("w ", w)
#     print("h", h)
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# cv2.imshow("img", img)
# cv2.waitKey(0)

def createAndSaveModel():
    # ფოტოების მისამართი
    imagePaths = list(paths.list_images('E:/python/dataset'))
    # print(len(imagePath))
    data = []
    labels = []
    # ციკლი ფოტოების მისამართისთვის
    for imagePath in imagePaths:
        # ფოტოების გარე მისამართის აღება
        label = imagePath.split(os.path.sep)[-2]
        # ფოტოს ჩატვირთვა და ზომის შემცირება 224-ზე
        image = load_img(imagePath, target_size=(224, 224))
        #  3D Numpy მასივში ფოტოების შენახვა
        image = img_to_array(image)
        # ფუნქცია აკონვერტირებს ფოტოებს RGB to BGR-ში და ინახავს float32 ფორმატში
        image = preprocess_input(image)
        # მასივში ელემენტების ჩამატება
        data.append(image)
        labels.append(label)

    # მასივის კონვერტირება Numpy Array-ის ფორმატში
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # weight - პარამეტრი  'imagenet' (pre-training on ImageNet)
    # include_top - დაკავშირებული ქსელი არის თუ არა ზედა დონეზე/ფენაზე
    # input_shape = მიწოდებული ფოტოს რეზოლუციის მითითება , 3  აღნიშნავს ჩანელების რაოდენობას
    baseModel = MobileNetV2(
        weights="imagenet",  include_top=False, input_shape=(244, 244, 3))

    # headModel-ის შექმნა რომელიც იქნება baseModel-ის ზედა ფენა
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    # FC  მოდელის ზედა შრედ დაყენება , რომელიც დამუშავდება და მითითება თუ რომელი მოდელია შეტანის და რომელი გამოტანის
    print(type(baseModel.input))
    print(type(headModel))
    model = Model(inputs=baseModel.input, outputs=headModel)
    # ციკლის დატრიალება მოდელზე , რათა არ მოხდეს პროცესის დროს მათი შეცვლა
    for layer in baseModel.layers:
        layer.trainable = False

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.20, stratify=labels, random_state=42)
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    INIT_LR = 1e-4
    EPOCHS = 20
    BS = 32
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    model.save('mask_recog_ver2.h5')
# createAndSaveModel()


def checkMaskStatus():
    cascPath = "E:/python/haarcascade_frontalface_alt2.xml"
    freeCascade = cv2.CascadeClassifier(cascPath)
    model = load_model("E:/python/mask_recog_ver2.h5")

    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = freeCascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
        # მასივში შეინახება  ამოცნობილი სახეები კასკადის მიერ
        faces_list = []
        # მასივში შეინახება ამოცნობილი სახეები მოდელის მიერ
        preds = []
        for (x, y, w, h) in faces:
            face_frame = frame[y:y+h, x:x+w]
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame = preprocess_input(face_frame)
            faces_list.append(face_frame)
            if len(faces_list) > 0:
                preds = model.predict(faces_list)
            for pred in preds:
                (mask, withoutMask) = pred
            label = "pirbadit" if mask > withoutMask else "pirbadis gareshe "
            color = (0, 255, 0) if label == "pirbadit" else (0, 0, 255)
            label = "{} : {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


checkMaskStatus()
