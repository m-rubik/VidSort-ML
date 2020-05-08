import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from video_sorter import videoAnalyser
import os
import time
from utilities.database_utilities import load_database

class App(QWidget):

    completed_videos: list = []

    def __init__(self):
        super().__init__()
        self.title = 'Face Recognizer for Videos'
        self.left = 100
        self.top = 100
        self.width = 590
        self.height = 150
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Add elements
        self.Label_video_folder = QLabel(self)
        self.Label_video_folder.setText('Video Folder:')
        self.Line_video_folder = QLineEdit(self)
        self.Line_video_folder.setEnabled(True)
        self.Line_video_folder.setPlaceholderText("...")
        self.Line_video_folder.move(80, 5)
        self.Line_video_folder.resize(400, 32)
        self.Label_video_folder.move(5,13)
        self.Button_choose_video_folder = QPushButton('Select Video Folder', self)
        self.Button_choose_video_folder.setToolTip('Select the video folder')
        self.Button_choose_video_folder.move(485,10)
        self.Button_choose_video_folder.clicked.connect(self.choose_folder)

        self.Label_classifier_model = QLabel(self)
        self.Label_classifier_model.setText('Classifier:')
        self.Line_classifier_model = QLineEdit(self)
        self.Line_classifier_model.setEnabled(True)
        self.Line_classifier_model.setPlaceholderText("...")
        self.Line_classifier_model.move(80, 40)
        self.Line_classifier_model.resize(400, 32)
        self.Label_classifier_model.move(5,47)
        self.Button_choose_classifier_model = QPushButton('Select Classifier', self)
        self.Button_choose_classifier_model.setToolTip('Select the classifier')
        self.Button_choose_classifier_model.move(485,42)
        self.Button_choose_classifier_model.clicked.connect(self.choose_classifier_model)

        self.Button_start = QPushButton('START', self)
        self.Button_start.setToolTip('Start the classification')
        self.Button_start.move(5,80)
        self.Button_start.clicked.connect(self.start)

        self.Button_Exit = QPushButton('Exit', self)
        self.Button_Exit.setToolTip('Exit the program')
        self.Button_Exit.move(80,80)
        self.Button_Exit.clicked.connect(self.closeEvent)

        self.show()

    @pyqtSlot()
    def choose_folder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if folder:
            self.Line_video_folder.setText(folder)
    
    @pyqtSlot()
    def choose_classifier_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Choose file", "","All Files (*)", options=options)
        if fileName:
            self.Line_classifier_model.setText(fileName)
        # Load the associated database
        self.db = load_database(os.path.split(self.Line_classifier_model.text())[1])

    @pyqtSlot()
    def start(self):
        self.Button_start.setEnabled(False)
        self.model_name = os.path.split(self.Line_classifier_model.text())[1]
        if len(self.completed_videos) == len(os.listdir(self.Line_video_folder.text())):
            self.Button_start.setEnabled(True)
            return 0
        for video in os.listdir(self.Line_video_folder.text()):
            if video not in self.completed_videos:
                if video not in self.db.data.keys():
                    worker = Worker(self.execute_this_fn, (self.Line_video_folder.text() + '/' + video, self.model_name, 2, 85,))
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
    
    # @pyqtSlot()
    # def open_multiple_files(self):
    #     options = QFileDialog.Options()
    #     options |= QFileDialog.DontUseNativeDialog
    #     files, _ = QFileDialog.getOpenFileNames(self,"Choose files", "","All Files (*);;Python Files (*.py)", options=options)
    #     if files:
    #         print(files)
    
    # @pyqtSlot()
    # def save_file_dialog(self):
    #     options = QFileDialog.Options()
    #     options |= QFileDialog.DontUseNativeDialog
    #     fileName, _ = QFileDialog.getSaveFileName(self,"Choose save file","","All Files (*);;Text Files (*.txt)", options=options)
    #     if fileName:
    #         print(fileName)

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

    def execute_this_fn(self, *args, **kwargs):
        self.busy = True
        video, model, time_skip, save_threshold = args[0]
        va = videoAnalyser(video, model, time_skip)
        va.detect_faces_in_video(save_threshold=save_threshold)
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
            video_name, _, _, _ = self.args[0]
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