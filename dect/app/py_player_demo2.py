



import sys
import uuid

import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimediaWidgets import QVideoWidget
from HumanDetect.app.GUI import Ui_MainWindow
from HumanDetect.app.myVideoWidget import myVideoWidget


class VideoFrameGrabber(QAbstractVideoSurface):
    frameAvailable = pyqtSignal(QImage)

    def __init__(self, widget: QWidget, parent: QObject):
        super().__init__(parent)

        self.widget = widget

    def supportedPixelFormats(self, handleType):
        return [QVideoFrame.Format_ARGB32, QVideoFrame.Format_ARGB32_Premultiplied,
                QVideoFrame.Format_RGB32, QVideoFrame.Format_RGB24, QVideoFrame.Format_RGB565,
                QVideoFrame.Format_RGB555, QVideoFrame.Format_ARGB8565_Premultiplied,
                QVideoFrame.Format_BGRA32, QVideoFrame.Format_BGRA32_Premultiplied, QVideoFrame.Format_BGR32,
                QVideoFrame.Format_BGR24, QVideoFrame.Format_BGR565, QVideoFrame.Format_BGR555,
                QVideoFrame.Format_BGRA5658_Premultiplied, QVideoFrame.Format_AYUV444,
                QVideoFrame.Format_AYUV444_Premultiplied, QVideoFrame.Format_YUV444,
                QVideoFrame.Format_YUV420P, QVideoFrame.Format_YV12, QVideoFrame.Format_UYVY,
                QVideoFrame.Format_YUYV, QVideoFrame.Format_NV12, QVideoFrame.Format_NV21,
                QVideoFrame.Format_IMC1, QVideoFrame.Format_IMC2, QVideoFrame.Format_IMC3,
                QVideoFrame.Format_IMC4, QVideoFrame.Format_Y8, QVideoFrame.Format_Y16,
                QVideoFrame.Format_Jpeg, QVideoFrame.Format_CameraRaw, QVideoFrame.Format_AdobeDng]

    def isFormatSupported(self, format):
        imageFormat = QVideoFrame.imageFormatFromPixelFormat(format.pixelFormat())
        size = format.frameSize()

        return imageFormat != QImage.Format_Invalid and not size.isEmpty() and \
               format.handleType() == QAbstractVideoBuffer.NoHandle

    def start(self, format: QVideoSurfaceFormat):
        imageFormat = QVideoFrame.imageFormatFromPixelFormat(format.pixelFormat())
        size = format.frameSize()

        if imageFormat != QImage.Format_Invalid and not size.isEmpty():
            self.imageFormat = imageFormat
            self.imageSize = size
            self.sourceRect = format.viewport()

            super().start(format)

            self.widget.updateGeometry()
            self.updateVideoRect()

            return True
        else:
            return False

    def stop(self):
        self.currentFrame = QVideoFrame()
        self.targetRect = QRect()

        super().stop()

        self.widget.update()

    def present(self, frame):
        if frame.isValid():
            cloneFrame = QVideoFrame(frame)
            cloneFrame.map(QAbstractVideoBuffer.ReadOnly)
            image = QImage(cloneFrame.bits(), cloneFrame.width(), cloneFrame.height(),
                           QVideoFrame.imageFormatFromPixelFormat(cloneFrame.pixelFormat()))
            self.frameAvailable.emit(image)  # this is very important
            cloneFrame.unmap()

        if self.surfaceFormat().pixelFormat() != frame.pixelFormat() or \
                self.surfaceFormat().frameSize() != frame.size():
            self.setError(QAbstractVideoSurface.IncorrectFormatError)
            self.stop()

            return False
        else:
            self.currentFrame = frame

            self.widget.repaint(self.targetRect)

            return True

    def updateVideoRect(self):
        size = self.surfaceFormat().sizeHint()
        size.scale(self.widget.size().boundedTo(size), Qt.KeepAspectRatio)

        self.targetRect = QRect(QPoint(0, 0), size)
        self.targetRect.moveCenter(self.widget.rect().center())

    def paint(self, painter):
        if self.currentFrame.map(QAbstractVideoBuffer.ReadOnly):
            oldTransform = self.painter.transform()

        if self.surfaceFormat().scanLineDirection() == QVideoSurfaceFormat.BottomToTop:
            self.painter.scale(1, -1)
            self.painter.translate(0, -self.widget.height())

        image = QImage(self.currentFrame.bits(), self.currentFrame.width(), self.currentFrame.height(),
                       self.currentFrame.bytesPerLine(), self.imageFormat)

        self.painter.drawImage(self.targetRect, image, self.sourceRect)

        self.painter.setTransform(oldTransform)

        self.currentFrame.unmap()


class App(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)

        # Show main window
        self.view = QMainWindow()

        self.centralWidget = QWidget(self.view)

        self.gridLayout = QGridLayout(self.centralWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)

        self.video_item = QVideoWidget()

        self.gridLayout.addWidget(self.video_item)

        self.view.setCentralWidget(self.centralWidget)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        self.grabber = VideoFrameGrabber(self.video_item, self)
        self.mediaPlayer.setVideoOuetput(self.grabber)

        self.grabber.frameAvailable.connect(self.process_frame)

        self.mediaPlayer.durationChanged.connect(self.update_duration)
        self.mediaPlayer.positionChanged.connect(self.update_slider_position)

        local = QUrl.fromLocalFile('c:/temp/lorem.mp4')
        media = QMediaContent(local)
        self.mediaPlayer.setMedia(media)
        self.mediaPlayer.play()

        self.view.show()

    def process_frame(self, image):
        # Save image here
        image.save('c:/temp/{}.jpg'.format(str(uuid.uuid4())))

    def update_duration(self):
        pass

    def update_slider_position(self):
        pass

'''
if __name__ == '__main__':
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)


    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = App(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    sys.excepthook = except_hook
    sys.exit(app.exec_())
'''

class myMainWindow(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.videoFullScreen = False   # 判断当前widget是否全屏
        self.videoFullScreenWidget = myVideoWidget()   # 创建一个全屏的widget
        self.videoFullScreenWidget.setFullScreen(1)
        self.videoFullScreenWidget.hide()               # 不用的时候隐藏起来
        #self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.wgt_video)  # 视频播放输出的widget，就是上面定义的
        self.btn_open.clicked.connect(self.openVideoFile)   # 打开视频文件按钮
        self.btn_play.clicked.connect(self.playVideo)       # play
        self.btn_stop.clicked.connect(self.pauseVideo)       # pause
        self.player.positionChanged.connect(self.changeSlide)      # change Slide
        self.videoFullScreenWidget.doubleClickedItem.connect(self.videoDoubleClicked)  #双击响应
        self.wgt_video.doubleClickedItem.connect(self.videoDoubleClicked)   #双击响应

        #self.video_item = QVideoWidget()
        #self.grabber = VideoFrameGrabber(self.video_item, self)
        #self.grabber = VideoFrameGrabber()
        #elf.player.setVideoOutput(self.grabber)

        #self.grabber.frameAvailable.connect(self.process_frame)

    def process_frame(self, image):
        # Save image here
        print("*****")

    def openVideoFile(self):
        self.player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))  # 选取视频文件
        self.player.play()  # 播放视频
    def playVideo(self):
        self.player.play()
    def pauseVideo(self):
        self.player.pause()
    def changeSlide(self,position):
        self.vidoeLength = self.player.duration()+0.1
        self.sld_video.setValue(round((position/self.vidoeLength)*100))
        self.lab_video.setText(str(round((position/self.vidoeLength)*100,2))+'%')
    def videoDoubleClicked(self,text):
        if self.player.duration() > 0:  # 开始播放后才允许进行全屏操作
            if self.videoFullScreen:
                self.player.pause()
                self.videoFullScreenWidget.hide()
                self.player.setVideoOutput(self.wgt_video)
                self.player.play()
                self.videoFullScreen = False
            else:
                self.player.pause()
                self.videoFullScreenWidget.show()
                self.player.setVideoOutput(self.videoFullScreenWidget)
                self.player.play()
                self.videoFullScreen = True




if __name__ == '__main__':
    app = QApplication(sys.argv)
    vieo_gui = myMainWindow()
    vieo_gui.show()
    sys.exit(app.exec_())