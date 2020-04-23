# Packages used in Macro model
import torch
import h5py
import time
import scipy.io as io
import numpy as np
from model import CSRNet
from torchvision import transforms
# Packages used in Micro Model
import tensorflow as tf
import cv2
import os
# Packages used in the design
from PyQt5 import QtWidgets, QtGui, QtCore
# Packages used to manipulate images
import PIL.Image as Image
from matplotlib import pyplot as plt, cm
from multiprocessing import Process
# sumiltaneous running proc
from threading import Thread
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Ui_MainWindow(object):

    # Here we put the general Layout and design of the whole application

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 552)
        MainWindow.setStyleSheet(""" background-color:#808080""")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 240, 141, 111))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.radioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.radioButton.setFont(font)
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout.addWidget(self.radioButton)
        self.radioButton.toggled.connect(self.Radio_Button)

        self.radioButton_2 = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout.addWidget(self.radioButton_2)
        self.radioButton_2.toggled.connect(self.Radio_Button)
        self.radioButton.setStyleSheet(""" color:#F0FFF0""")
        self.radioButton_2.setStyleSheet(""" color:#F0FFF0""")

        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(
            QtCore.QRect(150, 70, 701, 281))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(
            self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.graphicsView = QtWidgets.QGraphicsView(
            self.horizontalLayoutWidget)
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalLayout.addWidget(self.graphicsView)

        self.graphicsView_2 = QtWidgets.QGraphicsView(
            self.horizontalLayoutWidget)
        self.graphicsView_2.setObjectName("graphicsView_2")

        self.horizontalLayout.addWidget(self.graphicsView_2)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(230, 370, 131, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(73)
        font.setStrikeOut(False)
        self.label.setFont(font)
        self.label.setAutoFillBackground(False)
        self.label.setScaledContents(False)
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        self.label.setStyleSheet(""" color:#F0FFF0""")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(630, 370, 140, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")

        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.label_2.setFont(font)
        self.label_2.setAutoFillBackground(False)
        self.label_2.setScaledContents(False)
        self.label_2.setWordWrap(False)
        self.label_2.setObjectName("label_2")
        self.label_2.setStyleSheet(""" color:#F0FFF0""")

        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(117, 446, 290, 23))
        self.progressBar.setProperty("value", 0)

        self.progressBar.setObjectName("progressBar")

        self.lcdNumber = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber.setGeometry(QtCore.QRect(690, 420, 161, 61))

        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.lcdNumber.setFont(font)
        self.lcdNumber.setSmallDecimalPoint(True)
        self.lcdNumber.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcdNumber.setObjectName("lcdNumber")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(420, 420, 261, 81))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.label_3.setFont(font)
        self.label_3.setAutoFillBackground(False)
        self.label_3.setScaledContents(False)
        self.label_3.setWordWrap(False)
        self.label_3.setObjectName("label_3")
        self.label_3.setStyleSheet(""" color:#F0FFF0""")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(70, 20, 801, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_4.setStyleSheet(""" color:#F0FFF0""")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 130, 101, 61))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        # To Link the button upload to our function Upload_Image
        self.pushButton.clicked.connect(self.Upload_Image)

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(0, 100, 121, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_5.setStyleSheet(""" color:#F0FFF0""")

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(0, 210, 141, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_6.setStyleSheet(""" color:#F0FFF0""")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 420, 101, 61))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.Go)

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 875, 21))
        self.menubar.setObjectName("menubar")

        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

# Naming of each component

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "Crowd Analysis Desktop Application 2020 "))
        self.radioButton.setText(_translate("MainWindow", "Macro Model"))
        self.radioButton_2.setText(_translate("MainWindow", "Micro  Model"))
        self.label.setText(_translate("MainWindow", "Original Image"))
        self.label_2.setText(_translate("MainWindow", "Modified Image"))
        self.label_3.setText(_translate(
            "MainWindow", "Predected Number of Crowd :"))
        self.label_4.setText(_translate(
            "MainWindow", "Crowd Analysis  Using CSRNET and Faster RCNN Methods "))
        self.pushButton.setText(_translate("MainWindow", "Upload"))
        self.label_5.setText(_translate("MainWindow", "1) Select the Image !"))
        self.label_6.setText(_translate(
            "MainWindow", "2)Choose an algorithm !"))
        self.pushButton_2.setText(_translate("MainWindow", "Go ! "))

# Macro Model definition

    def Macro_Model(self):
        current_time = time.strftime("%Y-%m-%d-%H:%M")

        # Reading the path of the image which have been stored by the  openFileNameDialog function
        Log_file = open("path.txt", "r")
        path = Log_file.readline()
        Log_file.close()
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]),
        ])

        if path != "":
            # Load our Pretrained Model
            model = CSRNet()

        # Load weights
            checkpoint = torch.load(
                '/root/Desktop/Whole_Project/0model_best.pth.tar', map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

        # Image modification
            img = transform(Image.open(path).convert('RGB'))
            output = model(img.unsqueeze(0))
            temp = np.asarray(output.detach().cpu().reshape(
                output.detach().cpu().shape[2], output.detach().cpu().shape[3]))
            plt.imshow(temp, cmap=cm.jet)
        # Saving the modified image whith the current time
            plt.savefig(
                '/root/Desktop/Whole_Project/Saved_Modified_Images/'+current_time+'.png')
        # Preparing the graphic_View2 gadget to display the modified image
            Window_View = QtWidgets.QGraphicsScene()
            pixmap = QtGui.QPixmap(
                '/root/Desktop/Whole_Project/Saved_Modified_Images/'+current_time+'.png')
            Window_View.addPixmap(pixmap.scaled(
                360, 370, QtCore.Qt.KeepAspectRatio))

            self.graphicsView_2.setScene(Window_View)
            Predicted_num = int(output.detach().cpu().sum().numpy())
            self.lcdNumber.display(Predicted_num)

        else:
            print("You should upload an image ! ")

# Micro Model definition
    def initiate_tensorflow(self, path):
        self.path = path
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')  # Defining tensors for the graph
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')  # Each box denotes part of image with a person detected
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')  # Score represents the confidence for the detected person
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

    def detect(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})  # Using the model for detection

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0, i, 0] * im_height),
                             int(boxes[0, i, 1]*im_width),
                             int(boxes[0, i, 2] * im_height),
                             int(boxes[0, i, 3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def micro(self):
        # Reading the path of the image which have been stored by the  openFileNameDialog function
        Log_file = open("path.txt", "r")
        path = Log_file.readline()
        Log_file.close()
        if path != "":
            model_path = '/root/Desktop/Whole_Project/my_model1.pb'
            self.initiate_tensorflow(model_path)
            threshold = 0.1

            count = 0
            img = cv2.imread(path)

            img = cv2.resize(img, (640, 480))
            boxes, scores, classes, num = self.detect(img)
            for i in range(len(boxes)):
                if classes[i] == 1 and scores[i] > threshold:

                    box = boxes[i]
                    cv2.rectangle(img, (box[1], box[0]),
                                  (box[3], box[2]), (255, 0, 0), 2)
                    count += 1

            current_time = time.strftime("%Y-%m-%d-%H:%M")

            cv2.imwrite('/root/Desktop/Whole_Project/Micro_Model_Saver/' +
                        current_time+'.png', img)

        # Preparing the graphic_View2 gadget to display the modified image
            Window_View = QtWidgets.QGraphicsScene()
            pixmap = QtGui.QPixmap(
                '/root/Desktop/Whole_Project/Micro_Model_Saver/' +
                current_time+'.png')
            Window_View.addPixmap(pixmap.scaled(
                360, 370, QtCore.Qt.KeepAspectRatio))

            self.graphicsView_2.setScene(Window_View)
            self.lcdNumber.display(count)
        
            self.sess.close()
            
            

        else:
            print("You should upload an image ! ")


# To specify which Button was selected

    def Radio_Button(self):
        choice = self.radioButton.sender()
        if choice.isChecked():
            Log_file = open("choice.txt", "w")
            Log_file.write(choice.text())
            Log_file.close()

# This function let us to choose an image

    def openFileNameDialog(self, QDialog_Element):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            QDialog_Element, "Choose image of the CROWD ", "", "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            # Save path of choosen image  to a local file in order to call it in Go funtion when we execute the program
            Log_file = open("path.txt", "w")
            Log_file.write(fileName)
            Log_file.close()

            # Preparing the graphic View gadget to display an image
            Window_View = QtWidgets.QGraphicsScene()
            pixmap = QtGui.QPixmap(fileName)
            # resizing the original image to fit the view box
            Window_View.addPixmap(pixmap.scaled(
                360, 360, QtCore.Qt.KeepAspectRatio))
            self.graphicsView.setScene(Window_View)
        else:
            print("Upload a Crowd image ! ")

# To upload an image in order to be analysed

    def Upload_Image(self):
        # intiate an Qdialog element
        window = QtWidgets.QDialog()
        window.setGeometry(10, 10, 640, 480)
        self.openFileNameDialog(window)

# To execute the full process

    def Go(self):
        Log_file = open("choice.txt", "r")
        choice = Log_file.readline()
        Log_file.close()
        self.progressBar.setValue(0)
        if choice == "Macro Model":
            print("Macro Model is Loading ... !")
            Thread(target=self.handleTimer).start()
            self.Macro_Model()
        if choice == "Micro  Model":
            print("Micro Model is Loading ... !")
            Thread(target=self.handleTimer).start()
            self.micro()

# Clear files log
    def file_clear(self, path):
        file_cl = open(path, "w")
        file_cl.write("")
        file_cl.close()

# ProgressBar : to adjust the time of running program to this progress bar
    def handleTimer(self):
        value = self.progressBar.value()
        while value < 100:
            time.sleep(0.15)
            value = value + 1
            self.progressBar.setValue(value)


# Main Programme

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.file_clear("path.txt")
    ui.file_clear("choice.txt")
    MainWindow.show()
    sys.exit(app.exec_())
