import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from vidsortml.video_sorter import videoAnalyser
import os
import time
import pickle
import pathlib
from vidsortml.utilities.database_utilities import load_database

CONFIGURATIONS_ROOT_DIR = pathlib.Path(__file__).parent / 'configurations'

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Face Recognizer for Videos'
        self.left = 100
        self.top = 100
        self.width = 590
        self.height = 150
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
        
        self.show()
    
class MyTableWidget(QWidget):

    completed_videos: list = []
    
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.resize(300,200)
        
        # Add tabs
        self.tabs.addTab(self.tab1,"Analyse")
        self.tabs.addTab(self.tab2,"Train")
        self.tabs.addTab(self.tab3,"Search")

        self.init_tab1()
        self.init_tab2()
        self.init_tab3()

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.first = True

        self.show()

    @pyqtSlot()
    def init_tab1(self):
        
        # Create first tab layout
        self.tab1.layout = QVBoxLayout(self)

        # Populate the first tab
        self.tab1.form_layout = QFormLayout()

        # Add the select video and classifier entries
        self.tab1.Button_choose_video_folder = QPushButton('Select Video Folder', self)
        self.tab1.Button_choose_video_folder.setToolTip('Select the video folder')
        self.tab1.Button_choose_video_folder.clicked.connect(self.choose_video_folder)
        self.tab1.Line_video_folder = QLineEdit(self)
        self.tab1.Line_video_folder.setEnabled(True)
        self.tab1.Line_video_folder.setPlaceholderText("...")
        self.tab1.form_layout.addRow(self.tab1.Button_choose_video_folder, self.tab1.Line_video_folder)

        self.tab1.Button_choose_classifier_model = QPushButton('Select Classifier', self)
        self.tab1.Button_choose_classifier_model.setToolTip('Select the classifier')
        self.tab1.Button_choose_classifier_model.clicked.connect(self.choose_classifier_model)
        self.tab1.Line_classifier_model = QLineEdit(self)
        self.tab1.Line_classifier_model.setEnabled(True)
        self.tab1.Line_classifier_model.setPlaceholderText("...")
        self.tab1.form_layout.addRow(self.tab1.Button_choose_classifier_model, self.tab1.Line_classifier_model)

        self.tab1.layout.addLayout(self.tab1.form_layout)

        # Add other operating parameters
        self.tab1.params_layout = QHBoxLayout()
        v_box = QVBoxLayout()
        v_box.addWidget(QLabel("Capture Interval"))
        self.tab1.Spin_capture_interval = QSpinBox()
        self.tab1.Spin_capture_interval.setValue(2)
        self.tab1.Spin_capture_interval.setMinimum(1)
        self.tab1.Spin_capture_interval.resize(30, 32)
        self.tab1.Spin_capture_interval.setToolTip(r'Analyse only every nth second of the video, where n is the provided integer. A higher n results in less frames being analysed, but faster analysis times.')
        v_box.addWidget(self.tab1.Spin_capture_interval)
        self.tab1.params_layout.addLayout(v_box)

        v_box = QVBoxLayout()
        v_box.addWidget(QLabel("Detection Threshold"))
        self.tab1.Spin_detection_threshold = QSpinBox()
        self.tab1.Spin_detection_threshold.setValue(80)
        self.tab1.Spin_detection_threshold.setRange(1, 100)
        self.tab1.Spin_detection_threshold.setToolTip(r"Classifier must have confidence >= n% to match a face with a person.")
        v_box.addWidget(self.tab1.Spin_detection_threshold)
        self.tab1.params_layout.addLayout(v_box)

        v_box = QVBoxLayout()
        v_box.addWidget(QLabel("Save Threshold"))
        self.tab1.Spin_save_threshold = QSpinBox()
        self.tab1.Spin_save_threshold.setValue(90)
        self.tab1.Spin_save_threshold.setRange(1, 100)
        self.tab1.Spin_save_threshold.setToolTip(r"Classifier must have confidence >= n% to save the face it found for future training of that person.")
        v_box.addWidget(self.tab1.Spin_save_threshold)
        self.tab1.params_layout.addLayout(v_box)

        self.tab1.layout.addLayout(self.tab1.params_layout)

        # Add all buttons
        self.tab1.buttons_layout = QVBoxLayout()
        h_box = QHBoxLayout()
        self.tab1.Button_Exit = QPushButton('Load', self)
        self.tab1.Button_Exit.setToolTip('Load parameters from a configuration file')
        self.tab1.Button_Exit.clicked.connect(self.load_configuration)
        h_box.addWidget(self.tab1.Button_Exit)

        self.tab1.Button_Save = QPushButton('Save', self)
        self.tab1.Button_Save.setToolTip('Save entered parameters to a configuration file')
        self.tab1.Button_Save.clicked.connect(self.save_configuration)
        h_box.addWidget(self.tab1.Button_Save)

        self.tab1.Button_Load = QPushButton('EXIT', self)
        self.tab1.Button_Load.setToolTip('Exit the program')
        self.tab1.Button_Load.clicked.connect(self.closeEvent)
        h_box.addWidget(self.tab1.Button_Load)

        self.tab1.buttons_layout.addLayout(h_box)

        self.tab1.Button_start = QPushButton('START', self)
        self.tab1.Button_start.setToolTip('Start the classification')
        self.tab1.Button_start.clicked.connect(self.start)
        self.tab1.buttons_layout.addWidget(self.tab1.Button_start)

        self.tab1.layout.addLayout(self.tab1.buttons_layout)

        self.tab1.setLayout(self.tab1.layout)

    @pyqtSlot()
    def init_tab2(self):
        self.tab2.layout = QHBoxLayout(self)
        self.update_table()
        self.tab2.layout.addWidget(self.tab2.tableWidget)

        self.tab2.right_layout = QVBoxLayout()
        self.tab2.form_layout = QFormLayout()

        self.tab2.Button_choose_training_folder = QPushButton('Select Training Folder', self)
        self.tab2.Button_choose_training_folder.setToolTip('Select the folder containing all training images.')
        self.tab2.Button_choose_training_folder.clicked.connect(self.choose_training_folder)
        self.tab2.Line_training_folder = QLineEdit(self)
        self.tab2.Line_training_folder.setEnabled(True)
        self.tab2.Line_training_folder.setPlaceholderText("...")
        self.tab2.form_layout.addRow(self.tab2.Button_choose_training_folder, self.tab2.Line_training_folder)

        self.tab2.Label_model_name = QLabel()
        self.tab2.Label_model_name.setText("Model name:")
        # self.tab2.Label_model_name.setAlignment(Qt.AlignCenter)
        self.tab2.Line_model_name = QLineEdit(self)
        self.tab2.Line_model_name.setEnabled(True)
        self.tab2.Line_model_name.setPlaceholderText("...")
        self.tab2.form_layout.addRow(self.tab2.Label_model_name, self.tab2.Line_model_name)

        self.tab2.right_layout.addLayout(self.tab2.form_layout)
        
        self.tab2.radio_buttons_layout = QHBoxLayout()
        self.tab2.Label_radio_buttons = QLabel()
        self.tab2.Label_radio_buttons.setText("Model type:")
        self.tab2.radio_buttons_layout.addWidget(self.tab2.Label_radio_buttons)
        self.tab2.rb_mlp = QRadioButton("MLP")
        self.tab2.rb_svc = QRadioButton("SVC")
        self.tab2.rb_knn = QRadioButton("KNN")
        self.tab2.rb_mlp.setChecked(True)
        self.tab2.rb_svc.setChecked(False)
        self.tab2.rb_knn.setChecked(False)
        self.tab2.rb_mlp.toggled.connect(lambda:self.rb_state_change(self.tab2.rb_mlp))
        self.tab2.rb_svc.toggled.connect(lambda:self.rb_state_change(self.tab2.rb_svc))
        self.tab2.rb_knn.toggled.connect(lambda:self.rb_state_change(self.tab2.rb_knn))

        self.tab2.radio_buttons_layout.addWidget(self.tab2.rb_mlp)
        self.tab2.radio_buttons_layout.addWidget(self.tab2.rb_svc)
        self.tab2.radio_buttons_layout.addWidget(self.tab2.rb_knn)

        self.tab2.right_layout.addLayout(self.tab2.radio_buttons_layout)

        self.tab2.buttons_layout = QHBoxLayout()
        self.tab2.Button_Train = QPushButton('TRAIN', self)
        self.tab2.Button_Train.setToolTip('Train the model with the selected people')
        self.tab2.Button_Train.clicked.connect(self.train)
        self.tab2.buttons_layout.addWidget(self.tab2.Button_Train)
        self.tab2.Button_Learn = QPushButton('LEARN', self)
        self.tab2.Button_Learn.setToolTip('Learn the faces of the people in the training folder')
        self.tab2.Button_Learn.clicked.connect(self.learn)
        self.tab2.buttons_layout.addWidget(self.tab2.Button_Learn)

        self.tab2.right_layout.addLayout(self.tab2.buttons_layout)

        self.tab2.layout.addLayout(self.tab2.right_layout)
        self.tab2.setLayout(self.tab2.layout)

    @pyqtSlot()
    def init_tab3(self):
        pass

    @pyqtSlot()
    def update_table(self):
        self.tab2.tableWidget = QTableWidget()
        self.tab2.tableWidget.setColumnCount(1)
        root_dir = pathlib.Path(__file__).parent
        encodings_dir = pathlib.Path(root_dir / 'encodings')
        if encodings_dir.is_dir():
            for name in encodings_dir.iterdir():
                rowPosition = self.tab2.tableWidget.rowCount()
                self.tab2.tableWidget.insertRow(rowPosition)
                self.tab2.tableWidget.setItem(rowPosition,0, QTableWidgetItem(name.stem))
        self.tab2.tableWidget.setHorizontalHeaderLabels(["Name"])
        self.tab2.tableWidget.cellClicked.connect(self.cell_click)

    @pyqtSlot()
    def learn(self):
        # TODO: Make this multithreaded so it doesn't block the GUI process and cause it to timeout...
        from vidsortml.utilities.model_utilities import extract_all_face_encodings
        print("LEARNING....")
        extract_all_face_encodings(self.tab2.Line_training_folder.text())
        self.update_table()

    @pyqtSlot()
    def train(self):
        # TODO: Make this multithreaded so it doesn't block the GUI process and cause it to timeout...
        from vidsortml.utilities.model_utilities import train_model
        names = []
        for row in range(self.tab2.tableWidget.rowCount()):
            if self.tab2.tableWidget.item(row, 0).background() == QBrush(QColor("green")):
                names.append(self.tab2.tableWidget.item(row, 0).text())
        print("TRAINING....")
        if self.tab2.rb_mlp.isChecked():
            model_type="mlp"
        elif self.tab2.rb_svc.isChecked():
            model_type="svc"
        elif self.tab2.rb_knn.isChecked():
            model_type="knn"
        else:
            model_type = None
        train_model(model_name=self.tab2.Line_model_name.text(), names=names, model_type=model_type)

    @pyqtSlot()
    def rb_state_change(self, rb):
        if rb.text() == "MLP":
            if rb.isChecked() == True:
                self.tab2.rb_svc.setChecked(False)
                self.tab2.rb_knn.setChecked(False)
        if rb.text() == "SVC":
            if rb.isChecked() == True:
                self.tab2.rb_mlp.setChecked(False)
                self.tab2.rb_knn.setChecked(False)
        if rb.text() == "KNN":
            if rb.isChecked() == True:
                self.tab2.rb_svc.setChecked(False)
                self.tab2.rb_mlp.setChecked(False)

    def cell_click(self, index):
        if self.tab2.tableWidget.item(index, 0).background() == QBrush(QColor("green")):
            self.tab2.tableWidget.item(index, 0).setBackground(QBrush(QColor("white")))
        else:
            self.tab2.tableWidget.item(index, 0).setBackground(QBrush(QColor("green")))
    
    @pyqtSlot()
    def choose_training_folder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory", "./images/training/"))
        if folder:
            self.tab2.Line_training_folder.setText(folder)

    @pyqtSlot()
    def choose_video_folder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory", "./videos/"))
        if folder:
            self.tab1.Line_video_folder.setText(folder)
    
    @pyqtSlot()
    def choose_classifier_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Choose file", "./models/","All Files (*)", options=options)
        if fileName:
            self.tab1.Line_classifier_model.setText(fileName)

    def start_init(self):
        self.model_name = os.path.split(self.tab1.Line_classifier_model.text())[1]
        self.db = load_database(os.path.split(self.tab1.Line_classifier_model.text())[1])
        self.threadpool = QThreadPool()
        self.first = False

    @pyqtSlot()
    def start(self):
        if self.first:
            self.start_init()
        self.tab1.Button_start.setEnabled(False)
        print("NOT IMPLEMENTED YET --- Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        if len(self.completed_videos) == len(os.listdir(self.tab1.Line_video_folder.text())):
            print("All videos in the provided folder have been analysed.")
            self.tab1.Button_start.setEnabled(True)
            self.first = True
            return 0
        for video in os.listdir(self.tab1.Line_video_folder.text()):
            if video not in self.completed_videos:
                if video not in self.db.data.keys():
                    worker = Worker(self.execute_this_fn, (self.tab1.Line_video_folder.text() + '/' + video, self.model_name, self.tab1.Spin_capture_interval.value(), self.tab1.Spin_detection_threshold.value(), self.tab1.Spin_save_threshold.value(),))
                    worker.signals.result.connect(self.print_output)
                    worker.signals.finished.connect(self.thread_complete)
                    worker.signals.progress.connect(self.progress_fn)

                    # Analyse the video
                    self.threadpool.start(worker)
                    time.sleep(0.5)
                    if self.busy:
                        break
                else:
                    print(video, "has already been analysed.")
                    self.completed_videos.append(video)
                    self.start()


    @pyqtSlot()
    def load_configuration(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filepath, _ = QFileDialog.getOpenFileName(self,"Choose file", str(CONFIGURATIONS_ROOT_DIR), "All Files (*);;Python Files (*.py)", options=options)
        if filepath:
            try:
                with open(filepath, 'rb+') as f:
                    configuration = pickle.load(f)
                    self.tab1.Line_video_folder.setText(configuration["video folder"])
                    self.tab1.Line_classifier_model.setText(configuration["classifier path"])
                    self.tab1.Spin_capture_interval.setValue(configuration["capture interval"])
                    self.tab1.Spin_detection_threshold.setValue(configuration["detection threshold"])
                    self.tab1.Spin_save_threshold.setValue(configuration["save threshold"])
                    print("Load successful.")
            except Exception as e:
                print(e)
                return 1
            return 0

    @pyqtSlot()
    def save_configuration(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Choose save file", str(CONFIGURATIONS_ROOT_DIR), "All Files (*)", options=options)
        if fileName:
            try:
                with open(fileName, 'wb+') as f:
                        configuration = {}
                        # Get configuration stuff
                        configuration["video folder"] = self.tab1.Line_video_folder.text()
                        configuration["classifier path"] = self.tab1.Line_classifier_model.text()
                        configuration["capture interval"] = self.tab1.Spin_capture_interval.value()
                        configuration["detection threshold"] = self.tab1.Spin_detection_threshold.value()
                        configuration["save threshold"] = self.tab1.Spin_save_threshold.value()
                        pickle.dump(configuration, f)
                        print("Save successful.")
            except Exception as e:
                print(e)
                return 1
            return 0

    @pyqtSlot()
    def closeEvent(self):
        """Generate 'question' dialog on clicking 'X' button in title bar.

        Reimplement the closeEvent() event handler to include a 'Question'
        dialog with options on how to proceed - Save, Close, Cancel buttons
        """
        reply = QMessageBox.question(
            self, "Message",
            "Are you sure you want to quit? Any unsaved work will be lost.",
            QMessageBox.Save | QMessageBox.Close | QMessageBox.Cancel,
            QMessageBox.Save)

        if reply == QMessageBox.Close:
            app.quit()
        elif reply == QMessageBox.Save:
            self.save_configuration()

    def execute_this_fn(self, *args, **kwargs):
        self.busy = True
        video, model, capture_interval, detection_threshold, save_threshold = args[0]
        va = videoAnalyser(video, model, capture_interval)
        va.detect_faces_in_video(detection_threshold=detection_threshold, save_threshold=save_threshold)
        return "Analysis finished."

    def print_output(self, s):
        print(s)
        
    def thread_complete(self, video):
        print("Thread completed.\n")
        self.completed_videos.append(video)
        self.busy = False
        self.start()

    def progress_fn(self, n):
        print("%d%% done" % n)

class WorkerSignals(QObject):
    '''!
    Defines the signals available from a running worker thread.
    Supported signals are...
    finished : No data
    error : `tuple` (exctype, value, traceback.format_exc() )
    result : `object` data returned from processing, anything
    progress : `int` indicating % progress 
    '''
    finished = pyqtSignal(str)
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class Worker(QRunnable):
    '''!
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    @param callback: The function callback to run on this worker thread. Supplied args and kwargs will be passed through to the runner.
    @type callback: function
    @param args: Arguments to pass to the callback function
    @param kwargs: Keywords to pass to the callback function
    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()    

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress        

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            video_name, _, _, _, _ = self.args[0]
            video_name = os.path.split(video_name)[1]
            result = self.fn(*self.args, **self.kwargs)
        except:
            import traceback
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit(video_name)  # Done


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())