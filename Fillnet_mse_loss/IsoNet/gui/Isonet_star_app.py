# -*- coding: utf-8 -*-
'''
    File name: Isonet_star_app.py
    Author: Hui Wang (EICN)
    Date created: 4/21/2021
    Date last modified: 06/01/2021
    Python Version: 3.6.5
'''
import sys,os
import logging
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem,QMessageBox
from PyQt5.QtCore import QProcess

#Isonet packages
from IsoNet.gui.isonet_gui import Ui_MainWindow ##need to change in the package
from IsoNet.gui.model_star import Model, setTableWidget #need to change in the package
from IsoNet.util.metadata import MetaData,Label,Item


class MainWindowUIClass( Ui_MainWindow ):
    def __init__( self ):
        '''
        Initialize the super class
        初始化超类
        '''
        super().__init__()
        self.model = Model()
        
        #reset process as None 将进程重置为无
        self.p = None
        self.previous_log_line = ""
        self.setting_file = ".isonet.setting"
        # check for pid in last running
        #if os.path.isfile(self.model.pid_file):
        #    os.remove(self.model.pid_file)
        
    #界面初始化设置
    def setupUi( self, MW ):
        '''
        Setup the UI of the super class, and add here code
        that relates to the way we want our UI to operate.
        '''
        """
        设置超类的用户界面，并在这里添加与我们希望用户界面操作方式相关的代码。
        """
        #进行初始化
        super().setupUi( MW )

        #load default content in tomograms.star 
        setTableWidget(self.tableWidget, self.model.md)
        
        #set up functions when cells be clicked 设置当单元格被点击时的功能
        #self.tableWidget.cellPressed[int, int].connect(self.browseSlotTable)
        self.tableWidget.cellDoubleClicked[int, int].connect(self.browseSlotTable)
        self.tableWidget.cellChanged[int,int].connect(self.updateMDItem) 
        #self.tableWidget.horizontalHeaderItem(1).setToolTip("Header 0");
        #for i,lab in enumerate(self.model.header):
        #    self.tableWidget.horizontalHeaderItem(i-1).setToolTip(self.get_toolTip(lab))
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        ########################
        # connect function to buttons 连接功能到按钮
        ########################
        '''
        self.pushButton_insert.setStyleSheet("background-color : lightblue")
        self.pushButton_delete.setStyleSheet("background-color : lightblue")
        self.pushButton_open_star.setStyleSheet("background-color : lightblue")
        self.pushButton_3dmod.setStyleSheet("background-color : lightblue")
        self.button_deconov_dir.setStyleSheet("background-color : lightblue")
        self.button_mask_dir.setStyleSheet("background-color : lightblue")
        self.button_subtomo_dir.setStyleSheet("background-color : lightblue")
        self.button_result_dir_refine.setStyleSheet("background-color : lightblue")
        self.button_result_dir_predict.setStyleSheet("background-color : lightblue")
        self.button_subtomo_star_refine.setStyleSheet("background-color : lightblue")
        self.button_pretrain_model_refine.setStyleSheet("background-color : lightblue")
        self.button_tomo_star_predict.setStyleSheet("background-color : lightblue")
        self.button_pretrain_model_predict.setStyleSheet("background-color : lightblue")
        self.button_continue_iter.setStyleSheet("background-color : lightblue")
        self.pushButton_deconv.setStyleSheet("background-color : lightblue")
        self.pushButton_generate_mask.setStyleSheet("background-color : lightblue")
        self.pushButton_extract.setStyleSheet("background-color : lightblue")
        self.pushButton_refine.setStyleSheet("background-color : lightblue")
        self.pushButton_predict.setStyleSheet("background-color : lightblue")
        self.pushButton_predict_3dmod.setStyleSheet("background-color : lightblue")
        '''

        self.pushButton_insert.clicked.connect(self.copyRow)
        self.pushButton_delete.clicked.connect(self.removeRow)
        self.pushButton_open_star.clicked.connect(self.open_star)
        self.pushButton_3dmod.clicked.connect(self.view_3dmod)

        self.button_deconov_dir.clicked.connect(lambda: self.browseFolderSlot("deconv_dir"))
        self.button_mask_dir.clicked.connect(lambda: self.browseFolderSlot("mask_dir"))
        self.button_subtomo_dir.clicked.connect(lambda: self.browseFolderSlot("subtomo_dir"))
        self.button_result_dir_refine.clicked.connect(lambda: self.browseFolderSlot("result_dir_refine"))
        self.button_result_dir_predict.clicked.connect(lambda: self.browseFolderSlot("result_dir_predict"))
        
        self.button_subtomo_star_refine.clicked.connect(lambda: self.browseSlot("subtomo_star_refine"))
        self.button_pretrain_model_refine.clicked.connect(lambda: self.browseSlot("pretrain_model_refine"))
        self.button_tomo_star_predict.clicked.connect(lambda: self.browseSlot("tomo_star_predict"))
        self.button_pretrain_model_predict.clicked.connect(lambda: self.browseSlot("pretrain_model_predict"))
        self.button_continue_iter.clicked.connect(lambda: self.browseSlot("continue_from"))
        
        self.pushButton_deconv.clicked.connect(self.deconvolve)
        self.pushButton_generate_mask.clicked.connect(self.make_mask)
        self.pushButton_extract.clicked.connect(self.extract_subtomo)
        self.pushButton_refine.clicked.connect(self.refine)
        self.pushButton_predict.clicked.connect(self.predict)
        self.pushButton_predict_3dmod.clicked.connect(self.view_predict_3dmod)

        self.actionGithub.triggered.connect(self.openGithub)

        #########################
        #set icon location
        #设置icon文件夹地址
        #########################

        #get the root path for isonet 获取地址bin文件夹
        isonet_path = os.popen("where isonet.py").read()
        tmp = isonet_path.split("bin\\isonet.py")
        root_path = tmp[0]
        #获取icon图标文件夹
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(root_path+"gui\\icons\\icon_folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_deconov_dir.setIcon(icon)
        self.button_mask_dir.setIcon(icon)
        self.button_subtomo_star_refine.setIcon(icon)
        self.button_subtomo_dir.setIcon(icon)
        self.button_pretrain_model_refine.setIcon(icon)
        self.button_result_dir_refine.setIcon(icon)
        self.button_tomo_star_predict.setIcon(icon)
        self.button_pretrain_model_predict.setIcon(icon)
        self.button_result_dir_predict.setIcon(icon)
        self.button_continue_iter.setIcon(icon)

        self.read_setting()
        
        ###Set up log file monitor###
        ###"设置日志文件监控"###
        import datetime
        now = datetime.datetime.now()
        #create a empty log file 创建一个空的日志文件
        if not self.model.isValid(self.model.log_file):
            os.system("echo {} > {}".format(now.strftime("%Y-%m-%d %H:%M:%S"), self.model.log_file))
        self.textBrowser_log.setText(self.model.getLogContent(self.model.log_file))
        self.textBrowser_log.moveCursor(QtGui.QTextCursor.End)

        #self.horizontalLayout_48.hide()
        #for widgets in self.horizontalLayout_44.children():
            #print(widgets.widget())
            #for widget in widgets.children():
                #print(widget)
            #    widget.hide()
        
        ####################
        #self.log_watcher = QtCore.QFileSystemWatcher([self.model.log_file])
        #self.log_watcher.fileChanged.connect(self.update_log)
    

    #connect to all the main function button to run the process in the background  将所有主要功能按钮连接到后台运行进程
    #cmd is the command need to be excuted, and btn pass the button object         cmd 是需要执行的命令，btn 传递按钮对象

    def start_process(self, cmd, btn):
        #在按了按钮后开始运行具体的某些命令"Deconvolve","Generate Mask","Extract","Refine","Predict"
        if self.mw.p is None:  # No process running.
            self.mw.p = QProcess()
            #change the status of the current botton 修改当前按钮的状态
            if btn.text() in ["Deconvolve","Generate Mask","Extract","Refine","Predict"]:
                self.model.btn_pressed_text =  btn.text()
                btn.setText("Stop")
                btn.setStyleSheet('QPushButton {color: red;}')
            else:
                btn.setEnabled(False)
            self.mw.p.readyReadStandardOutput.connect(self.dataReady)
            self.mw.p.finished.connect(lambda: self.process_finished(btn))
            self.mw.p.start(cmd)
        #停止某些命令"Deconvolve","Generate Mask","Extract","Refine","Predict"运行
        elif btn.text() =="Stop":
            if self.mw.p:
                self.mw.p.kill()
            else:
                if self.model.btn_pressed_text:
                    btn.setText(self.model.btn_pressed_text)
        else:
            self.warn_window("Already runing another job, please wait until it finished!")

    def process_finished(self, btn):
        if btn.text() == "Stop":
            if self.model.btn_pressed_text:
                btn.setText(self.model.btn_pressed_text)
                #btn.setText("Refine")
                self.model.btn_pressed_text = None
                btn.setStyleSheet("QPushButton {color: black;}")
        else:
            btn.setEnabled(True)
        self.model.read_star()
        setTableWidget(self.tableWidget, self.model.md)   
        self.mw.p = None
        
    #link to log window to display output of stdout 链接到日志窗口，显示标准输出的输出
    def dataReady(self):
        cursor = self.textBrowser_log.textCursor()
        #cursor.movePosition(cursor.End)
        # have transfer byte string to unicode string 将字节字符串转换为Unicode字符串
        import string
        printable = set(string.printable)
        printable.add(u'\u2588')

        txt = str(self.mw.p.readAll(),'ISO-8859-1')
        #txt += self.mw.p.errorString()

        printable_txt = "".join(list(filter(lambda x: x in printable, txt)))
        
        if '[' in self.previous_log_line and '[' in printable_txt:
            cursor.movePosition(cursor.StartOfLine, cursor.MoveAnchor)
            cursor.movePosition(cursor.End, cursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()
        cursor.insertText(printable_txt)
        f = open(self.model.log_file, 'a+')
        f.write(printable_txt)
        f.close()

        self.previous_log_line = printable_txt
        #self.textBrowser_log.ensureCursorVisible()
        verScrollBar = self.textBrowser_log.verticalScrollBar()
        scrollIsAtEnd = verScrollBar.maximum() - verScrollBar.value()
        if scrollIsAtEnd <=100:
            verScrollBar.setValue(verScrollBar.maximum()) # Scrolls to the bottom

        #self.textBrowser_log.moveCursor(QtGui.QTextCursor.End)

    def removeRow(self): #Delete按钮
        #print(self.tableWidget.selectionModel().selectedIndexes()[0].row())
        #print(self.tableWidget.selectionModel().selectedIndexes()[0].column())

        indices = self.tableWidget.selectionModel().selectedRows() 
        if indices:
            for index in sorted(indices,reverse=True):
                self.tableWidget.removeRow(index.row()) 
        self.updateMD()

    def copyRow(self): # insert按钮
        rowCount = self.tableWidget.rowCount()
        columnCount = self.tableWidget.columnCount()
        if rowCount <=0 :
            self.tableWidget.insertRow(self.tableWidget.rowCount())
            for j in range(columnCount):
                #self.model.md._setItemValue(it,Label(self.model.header[j+1]),self.tableWidget.item(i, j).text())
                #print(self.default_value(self.model.header[j+1]))
                self.tableWidget.setItem(0, j, QTableWidgetItem(self.default_value(self.model.header[j+1])))
                #print(self.tableWidget.item(0, j).text())
        else:
            indices = self.tableWidget.selectionModel().selectedRows() 

            if indices:
                for index in sorted(indices):
                    self.tableWidget.insertRow(self.tableWidget.rowCount())
                    rowCount = self.tableWidget.rowCount()
                    for j in range(columnCount):
                        if self.model.header[j+1] in ["rlnDeconvTomoName","rlnMaskName","rlnCorrectedTomoName","rlnMaskBoundary"]:
                            self.tableWidget.setItem(rowCount-1, j, QTableWidgetItem("None"))
                        #self.tableWidget.cellChanged[rowCount-1, j].connect(self.updateMD)  
                        else:
                            self.tableWidget.setItem(rowCount-1, j, QTableWidgetItem(self.tableWidget.item(index.row(), j).text()))
            else:
                self.tableWidget.insertRow(self.tableWidget.rowCount())
                rowCount = self.tableWidget.rowCount()
                for j in range(columnCount):
                    if self.model.header[j+1] in ["rlnDeconvTomoName","rlnMaskName","rlnCorrectedTomoName","rlnMaskBoundary"]:
                            self.tableWidget.setItem(rowCount-1, j, QTableWidgetItem("None"))
                    elif not self.tableWidget.item(rowCount-2, j) is None:
                        self.tableWidget.setItem(rowCount-1, j, QTableWidgetItem(self.tableWidget.item(rowCount-2, j).text()))
        self.updateMD()
    
    def default_value(self, label):#默认值在导入数据集文件时候设置默认值
        switcher = {
            "rlnMicrographName": "None",
            "rlnPixelSize": "1",
            "rlnDefocus": "0",
            "rlnNumberSubtomo":"100",
            "rlnSnrFalloff":"1",
            "rlnDeconvStrength": "1",
            "rlnDeconvTomoName":"None",
            "rlnMaskBoundary":"None",
            "rlnMaskDensityPercentage": "50",
            "rlnMaskStdPercentage": "50",
            "rlnMaskName": "None"

        }
        return switcher.get(label, "None")
        
    def switch_btn(self, btn):
        switcher = {
            "mask_dir": self.lineEdit_mask_dir,
            "deconv_dir": self.lineEdit_deconv_dir,
            "subtomo_dir": self.lineEdit_subtomo_dir,
            "result_dir_refine": self.lineEdit_result_dir_refine,
            "result_dir_predict": self.lineEdit_result_dir_predict,
            "subtomo_star_refine":self.lineEdit_subtomo_star_refine,
            "pretrain_model_refine":self.lineEdit_pretrain_model_refine,
            "tomo_star_predict": self.lineEdit_tomo_star_predict,
            "pretrain_model_predict":self.lineEdit_pretrain_model_predict,
            "continue_from": self.lineEdit_continue_iter
        }
        return switcher.get(btn, "Invaid btn name")
    
    def file_types(self, item):
        switcher = {#在按钮附近时提示词
            "rlnMicrographName":"mrc or rec file (*.mrc *.rec) ;; All Files (*)",
            "rlnDeconvTomoName":"mrc or rec file (*.mrc *.rec) ;; All Files (*)",
            "rlnMaskName":"mrc or rec file (*.mrc *.rec) ;; All Files (*)",
            "rlnMaskBoundary": "mod file (*.mod) ;; All Files (*)" 
        }
        return switcher.get(item, "Invaid file types")
    def get_toolTip(self,label):
        switcher = {#在按钮附近时提示词
            "rlnMicrographName": "Your tomogram filenames",
            "rlnPixelSize": "pixel size of your input tomograms",
            "rlnDefocus": "estimated defocus value around 0 degree",
            "rlnNumberSubtomo":"number of subtomograms to be extraced",
            "rlnSnrFalloff":"SNR fall rate with the frequency",
            "rlnDeconvStrength": "(1.0) Strength of the deconvolution",
            "rlnDeconvTomoName":"automaticly saved deconved tomogram filename",
            "rlnMaskBoundary":"model file that define your mask boundary(optional)",
            "rlnMaskDensityPercentage": "The approximate percentage of pixels to keep based on their local pixel density",
            "rlnMaskStdPercentage": "The approximate percentage of pixels to keep based on their local standard deviation",
            "rlnMaskName": "automaticly saved mask tomogram filename"
        }
        return switcher.get(label, "None")
    """
        "rlnMicrographName": "你的层析图文件名",
        "rlnPixelSize": "你的输入层析图的像素大小",
        "rlnDefocus": "大约在0度附近的估计离焦值",
        "rlnNumberSubtomo": "要提取的亚体积个数",
        "rlnSnrFalloff": "随着频率的信噪比下降速率",
        "rlnDeconvStrength": "(1.0) 去卷积的强度",
        "rlnDeconvTomoName": "自动保存的去卷积层析图文件名",
        "rlnMaskBoundary": "定义你的掩模边界的模型文件（可选）",
        "rlnMaskDensityPercentage": "基于局部像素密度，保留的像素近似百分比",
        "rlnMaskStdPercentage": "基于局部标准差，保留的像素近似百分比",
        "rlnMaskName": "自动保存的掩模层析图文件名"
    """
    def updateMD ( self ):        
        star_file = self.model.tomogram_star
        rowCount = self.tableWidget.rowCount()
        columnCount = self.tableWidget.columnCount()
        data = self.model.md._data
        self.model.md = MetaData()
        self.model.md.addLabels('rlnIndex')
        for j in range(columnCount):
            self.model.md.addLabels(self.model.header[j+1])
            #self.model.md.addLabels(self.tableWidget.horizontalHeaderItem(j).text())

        for i in range(rowCount):
            #TODO check the folder contains only tomograms.
            it = Item()
            self.model.md.addItem(it)
            self.model.md._setItemValue(it,Label('rlnIndex'),str(i+1))
            for j in range(columnCount):
                try:
                    #print("update:",Label(self.model.header[j+1]),self.tableWidget.item(i, j).text())
                    if len(self.tableWidget.item(i, j).text()) <1:
                        
                        if self.model.header[j+1] != "rlnMaskBoundary":
                            previous_value = getattr(data[i],self.model.header[j+1])
                        else:
                            previous_value = "None"

                        self.model.md._setItemValue(it,Label(self.model.header[j+1]),previous_value)
                        self.tableWidget.setItem(i, j, QTableWidgetItem(str(previous_value)))
                    else:
                        self.model.md._setItemValue(it,Label(self.model.header[j+1]),self.tableWidget.item(i, j).text())
                     
                    #self.model.md._setItemValue(it,Label(self.tableWidget.horizontalHeaderItem(j).text()),self.tableWidget.item(i, j).text())
                except:
                    previous_value = getattr(data[i],self.model.header[j+1])
                    self.model.md._setItemValue(it,Label(self.model.header[j+1]),previous_value)
                    self.tableWidget.setItem(i, j, QTableWidgetItem(str(previous_value)))
                    #print("error in seeting values for {}! set it to previous value automatically.".format(self.tableWidget.horizontalHeaderItem(j).text()))
        self.model.md.write(star_file)

    def updateMDItem ( self, i, j ):
        try:
            current_value = self.tableWidget.item(i, j).text()
            #self.model.md._setItemValue(self.mnodel.md._data[i],Label(self.model.header[j+1]),current_value)
            #for row,it in enumerate(self.model.md):
            #    print(i,j)
            #    if row == i:
            #        self.model.md._setItemValue(it,Label(self.tableWidget.horizontalHeaderItem(j).text()),self.tableWidget.item(i, j).text())
            self.updateMD()
        except:
            pass
     
    def browseSlot( self , btn ):
        '''
        Called when the user presses the Browse button
        当用户按下浏览按钮时调用
        '''
        lineEdit = self.switch_btn(btn)
        
        pwd = os.getcwd().replace("\\","/")
        
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        
        flt = "All Files (*)"
        if btn == "continue_from":
            flt = "json file (*.json);;All Files (*)"
        if btn == "subtomo_star_refine" or btn == "tomo_star_predict":
            flt = "star file (*.star);;All Files (*)"
        if btn == "pretrain_model_refine" or btn == "pretrain_model_predict":
            flt = "model file (*.h5);;All Files (*)"
            
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "Choose File",
                        "",
                        flt,
                        options=options)
        if fileName:
            #self.model.setFileName( fileName )
            #######
            #cmd = "echo choose file: {} >> log.txt ".format(fileName)
            #os.system(cmd)
            #self.logWindow.append("choose file: {}".format(fileName) )
            simple_name = self.model.sim_path(pwd,fileName)
            lineEdit.setText( simple_name )
            #self.logWindow.moveCursor(QtGui.QTextCursor.End) 
            #######
            #self.refreshAll()
        #self.debugPrint( "Browse button pressed" )

    def browseFolderSlot( self , btn):
        ''' 
            Called when the user presses the Browse folder button
            TODO: add file name filter
            当用户按下“浏览文件夹”按钮时调用
            待办事项：添加文件名过滤器
        '''
        lineEdit = self.switch_btn(btn)
        try:
            pwd = os.getcwd().replace("\\","/")
            dir_path=QtWidgets.QFileDialog.getExistingDirectory(None,"Choose Directory",pwd)
            #self.model.setFolderName( dir_path )
            #cmd = "echo choose folder: {} >> log.txt ".format(dir_path)
            #os.system(cmd)
            #self.logWindow.append("choose folder: {}".format(dir_path) )
            #pwd = os.getcwd().replace("\\","/")

            simple_path = self.model.sim_path(pwd,dir_path)

            lineEdit.setText( simple_path )
            #self.logWindow.moveCursor(QtGui.QTextCursor.End) 
            #self.refreshAll()
        except:
            ##TODO: record to log.
            pass

    def browseSlotTable( self , i, j):
        '''
        Called when the user presses the Browse folder button
        当用户按下“浏览文件夹”按钮时调用
        '''
        if self.model.header[j+1] in ["rlnMicrographName", "rlnMaskBoundary","rlnDeconvTomoName","rlnMaskName"]:
            try:
                options = QtWidgets.QFileDialog.Options()
                options |= QtWidgets.QFileDialog.DontUseNativeDialog
                fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                                None,
                                "Choose File",
                                "",
                                self.file_types(self.model.header[j+1]),
                                options=options)
                if not fileName:
                    fileName = self.tableWidget.item(i, j).text()
                pwd = os.getcwd().replace("\\","/")

                simple_path = self.model.sim_path(pwd,fileName)

                self.tableWidget.setItem(i, j, QTableWidgetItem(simple_path))
            except:
                ##TODO: record to log.
                pass
        else:
            pass
    #对图片进行卷积处理 将图片特征变清晰 本质上就是用isonet.py中 deconvolve方法将图片处理
    def deconvolve( self ):

        tomogram_star = self.model.tomogram_star
        #运行isonet中deconv方法
        cmd = "python isonet.py deconv {} ".format(tomogram_star)
        #获取各个框里面的参数
        #获取deconvolve directory 的值
        if self.lineEdit_deconv_dir.text():
            cmd = "{} --deconv_folder {}".format(cmd, self.lineEdit_deconv_dir.text())
        #获取tomo index的值
        if self.lineEdit_tomo_index_deconv.text():
            cmd = "{} --tomo_idx {}".format(cmd, self.lineEdit_tomo_index_deconv.text())
        #获取ncpu的值
        if self.lineEdit_ncpu.text():
            cmd = "{} --ncpu {}".format(cmd, self.lineEdit_ncpu.text())
        # 获取highpassnyquist的值
        if self.lineEdit_highpassnyquist.text():
            cmd = "{} --highpassnyquist {}".format(cmd, self.lineEdit_highpassnyquist.text())
        #获取chunk size的值
        if self.lineEdit_chunk_size.text():
            cmd = "{} --chunk_size {}".format(cmd, self.lineEdit_chunk_size.text())
        # 获取overlap的值
        if self.lineEdit_overlap.text():
            cmd = "{} --overlap {}".format(cmd, self.lineEdit_overlap.text())
        #保存 获取的值
        self.save_setting()
        # 如果勾选了只打印命令并且按钮文本为 'Deconvolve'，则打印命令
        if self.checkBox_only_print_command_prepare.isChecked() and self.pushButton_deconv.text() == 'Deconvolve':
            print(cmd)
            #logging.info(cmd)
        else:
            self.start_process(cmd,self.pushButton_deconv)

    #设置图片的mask 本质上就是用isonet.py中 make_mask方法设置mask
    def make_mask( self ):
        #print("#####making mask############")

        tomogram_star = self.model.tomogram_star
        #运行isonet mask函数
        cmd = "python isonet.py make_mask {} ".format(tomogram_star)
        #获取各个参数的数值
        if self.lineEdit_mask_dir.text():
            cmd = "{} --mask_folder {}".format(cmd, self.lineEdit_mask_dir.text())
        if self.lineEdit_patch_size.text():
            cmd = "{} --patch_size {}".format(cmd, self.lineEdit_patch_size.text())
        if not self.checkBox_use_deconv_mask.isChecked():
            cmd = "{} --use_deconv_tomo {}".format(cmd, False)
        if self.lineEdit_tomo_index_mask.text():
            cmd = "{} --tomo_idx {}".format(cmd, self.lineEdit_tomo_index_mask.text())
        if self.lineEdit_z_crop.text():
            cmd = "{} --z_crop {}".format(cmd, self.lineEdit_z_crop.text())
        #保存 获取的值
        self.save_setting()
        # 如果勾选了只打印命令并且按钮文本为 'generate mask'，则打印命令
        if self.checkBox_only_print_command_prepare.isChecked() and self.pushButton_generate_mask.text() == 'Generate Mask':
            print(cmd)
        else:
            self.start_process(cmd,self.pushButton_generate_mask)

    #把deconvole ctf 和 generate mask处理过后的数值进行数据提取 本质上就是用isonet.py中 extract_subtomo方法进行提取数据
    def extract_subtomo( self ):
        # 获取 tomogram_star 属性
        tomogram_star = self.model.tomogram_star
        # 运行isonet类中 extract方法
        cmd = "python isonet.py extract {} ".format(tomogram_star)
        # 检查并添加 subtomo_folder 参数
        if self.lineEdit_subtomo_dir.text():
            cmd = "{} --subtomo_folder {}".format(cmd, self.lineEdit_subtomo_dir.text())
        # 检查并添加 subtomo_star 参数
        if self.lineEdit_subtomo_star_extract.text():
            cmd = "{} --subtomo_star {}".format(cmd, self.lineEdit_subtomo_star_extract.text())
        # 检查并添加 cube_size 参数
        if self.lineEdit_cube_size_extract.text():
            cmd = "{} --cube_size {}".format(cmd, self.lineEdit_cube_size_extract.text())
        # 检查并添加 use_deconv_tomo 参数
        if not self.checkBox_use_deconv_extract.isChecked():
            cmd = "{} --use_deconv_tomo {}".format(cmd, False)
        # 检查并添加 tomo_idx 参数
        if self.lineEdit_tomo_index_extract.text():
            cmd = "{} --tomo_idx {}".format(cmd, self.lineEdit_tomo_index_extract.text())
        # 保存设置
        self.save_setting()
        # 如果勾选了只打印命令并且按钮文本为 'Extract'，则打印命令
        if self.checkBox_only_print_command_prepare.isChecked() and self.pushButton_extract.text() == 'Extract':
            print(cmd)
        else:
            # 否则启动进程
            self.start_process(cmd,self.pushButton_extract)

    #将模型进行反复迭代 本质上就是用isonet.py中 refine方法进行预测
    def refine( self ):

        subtomo_star = self.lineEdit_subtomo_star_refine.text() if self.lineEdit_subtomo_star_refine.text() else "subtomo.star"

        cmd = "python isonet.py refine {} ".format(subtomo_star)

        if self.lineEdit_gpuID_refine.text():
            cmd = "{} --gpuID {}".format(cmd, self.lineEdit_gpuID_refine.text())
        if self.lineEdit_pretrain_model_refine.text():
            cmd = "{} --pretrained_model {}".format(cmd, self.lineEdit_pretrain_model_refine.text())
        if self.lineEdit_continue_iter.text():
            cmd = "{} --continue_from {}".format(cmd, self.lineEdit_continue_iter.text())
        if self.lineEdit_result_dir_refine.text():
            cmd = "{} --result_dir {}".format(cmd, self.lineEdit_result_dir_refine.text())
        if self.lineEdit_preprocessing_ncpus.text():
            cmd = "{} --preprocessing_ncpus {}".format(cmd, self.lineEdit_preprocessing_ncpus.text())
            
        if self.lineEdit_iteration.text():
            cmd = "{} --iterations {}".format(cmd, self.lineEdit_iteration.text())
        if self.lineEdit_batch_size.text():
            cmd = "{} --batch_size {}".format(cmd, self.lineEdit_batch_size.text())
        if self.lineEdit_epoch.text():
            cmd = "{} --epochs {}".format(cmd, self.lineEdit_epoch.text()) 
        if self.lineEdit_steps_per_epoch.text():
            cmd = "{} --steps_per_epoch {}".format(cmd, self.lineEdit_steps_per_epoch.text())
        if self.lineEdit_lr.text():
            cmd = "{} --learning_rate {}".format(cmd, self.lineEdit_lr.text())

        if self.lineEdit_noise_level.text():
            cmd = "{} --noise_level {}".format(cmd, self.lineEdit_noise_level.text())
        if self.lineEdit_noise_start_iter.text():
            cmd = "{} --noise_start_iter {}".format(cmd, self.lineEdit_noise_start_iter.text())
        if not self.comboBox_noise_mode.currentText() == "noFilter":
            cmd = "{} --noise_mode {}".format(cmd, self.comboBox_noise_mode.currentText())
        
        if self.lineEdit_drop_out.text():
            cmd = "{} --drop_out {}".format(cmd, self.lineEdit_drop_out.text())
        if self.lineEdit_network_depth.text():
            cmd = "{} --unet_depth {}".format(cmd, self.lineEdit_network_depth.text())
        if self.lineEdit_convs_per_depth.text():
            cmd = "{} --convs_per_depth {}".format(cmd, self.lineEdit_convs_per_depth.text())
        if self.lineEdit_kernel.text():
            cmd = "{} --kernel {}".format(cmd, self.lineEdit_kernel.text())
        if self.lineEdit_filter_base.text():
            cmd = "{} --filter_base {}".format(cmd, self.lineEdit_filter_base.text())
        if self.checkBox_pool.isChecked():
            cmd = "{} --pool {}".format(cmd, True)      


        if not self.checkBox_batch_normalization.isChecked():
            cmd = "{} --batch_normalization {}".format(cmd, False)    
        if not self.checkBox_normalization_percentile.isChecked():
            cmd = "{} --normalize_percentile {}".format(cmd, False)
            
        self.save_setting()
        if self.checkBox_only_print_command_refine.isChecked() and self.pushButton_refine.text() == 'Refine':
            print(cmd)
        else:
            self.start_process(cmd,self.pushButton_refine)
    #对将训练好的模型图片进行重建 本质上就是用isonet.py中 predict方法进行预测
    def predict( self ):
        tomo_star = self.lineEdit_tomo_star_predict.text() if self.lineEdit_tomo_star_predict.text() else "tomograms.star"
        gpuID = self.lineEdit_gpuID_predict.text() if self.lineEdit_gpuID_predict.text() else '0,1,2,3'
        cmd = "python isonet.py predict {}".format(tomo_star)
        
        if self.lineEdit_pretrain_model_predict.text() and self.model.isValid(self.lineEdit_pretrain_model_predict.text()):
            cmd = "{} {}".format(cmd, self.lineEdit_pretrain_model_predict.text())
        else:
            self.warn_window("no trained model detected")
            return
        # if self.lineEdit_gpuID_predict.text():
        #     cmd = "{} --gpuID {}".format(cmd, self.lineEdit_gpuID_predict.text())
        cmd = "{} --gpuID {}".format(cmd,gpuID)
        
        if self.lineEdit_tomo_index_predict.text():
            cmd = "{} --tomo_idx {}".format(cmd, self.lineEdit_tomo_index_predict.text())

        if self.lineEdit_result_dir_predict.text():
            cmd = "{} --output_dir {}".format(cmd, self.lineEdit_result_dir_predict.text())
        
        if self.lineEdit_cube_size_predict.text():
            cmd = "{} --cube_size {}".format(cmd, self.lineEdit_cube_size_predict.text())

        if self.lineEdit_crop_size_predict.text():
            cmd = "{} --crop_size {}".format(cmd, self.lineEdit_crop_size_predict.text())
        
        if not self.checkBox_use_deconv_predict.isChecked():
            cmd = "{} --use_deconv_tomo {}".format(cmd, False)        
                    
        self.save_setting()
        if self.checkBox_only_print_command_predict.isChecked() and self.pushButton_predict.text() == "Predict":
            print(cmd)
        else:
            self.start_process(cmd,self.pushButton_predict)

        
    #查看3d视图但是我现在看不了 本质上就是用在cmd 中调用3dmod方法进行观看图片
    def view_3dmod(self):
        slected_items = self.tableWidget.selectedItems()
        if len(slected_items) > 0:
            cmd = "3dmod"
            model_file=""
            previous_i = -1
            for item in slected_items:
                i = item.row()
                j = item.column()
                if previous_i != -1 and i != previous_i:
                    cmd = "{} {} {}".format(cmd,model_file,"; 3dmod")
                    model_file=""
                item_text = self.tableWidget.item(i, j).text()
                if item_text[-4:] == '.mrc' or item_text[-4:] == '.rec':
                    cmd = "{} {}".format(cmd,item_text)
                if self.model.header[j+1]=="rlnMaskBoundary" and item_text != "None":
                    model_file = "{}".format(item_text)
                previous_i = i
            
            cmd = "{} {}".format(cmd,model_file)

            if cmd != "3dmod":
                os.system(cmd)
            else:
                self.warn_window("selected items are not mrc or rec file(s)")
    #在预测好3dmod查看图片格式
    def view_predict_3dmod(self):
        try:
            result_dir_predict = self.lineEdit_result_dir_predict.text()
            if len(result_dir_predict) < 1:
                result_dir_predict = 'corrected_tomos'
            list_file = os.listdir(result_dir_predict)
            cmd = "3dmod"
            for f in list_file:
                if f[-4:] == ".mrc" or f[-4:] == ".rec":                   
                    cmd = "{} {}/{}".format(cmd,result_dir_predict,f)
            
            if cmd != "3dmod":
                os.system(cmd)  
            else:
                self.warn_window("no mrc or rec file(s) detected in results folder: {}!".format(result_dir_predict))       
        except Exception:
            print('pass')
            
    #选择文件夹在刚开始的时候
    def open_star( self ):#对应open_star按钮 对应的悬着数据集
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "Choose File",
                        "",
                        "Star file (*.star)",
                        options=options)
        if fileName:
            try:
                tomo_file = self.model.sim_path(self.model.pwd, fileName)
                read_result = self.model.read_star_gui(tomo_file)
                if read_result == 1:
                    self.warn_window("The input star file is not legid!")
                else:
                    setTableWidget(self.tableWidget, self.model.md)
            except:
                print("warning")
                pass
    #设置各个按钮的文本信息
    def read_setting(self):
        if os.path.exists(self.setting_file):
            data = {}
            try:
                with open(self.setting_file) as f:
                    for line in f:
                        (k, v) = line.split(":")
                        data[k] = v.strip()
                self.lineEdit_deconv_dir.setText(data['deconv_dir'])
                self.lineEdit_tomo_index_deconv.setText(data['tomo_index_deconv'])
                self.lineEdit_preprocessing_ncpus.setText(data['preprocessing_ncpus'])
                self.lineEdit_chunk_size.setText(data['chunk_size'])
                self.lineEdit_highpassnyquist.setText(data['highpassnyquist'])
                self.lineEdit_overlap.setText(data['overlap'])

                self.lineEdit_mask_dir.setText(data['mask_dir'])
                self.lineEdit_tomo_index_mask.setText(data['tomo_index_mask'])

                self.checkBox_use_deconv_mask.setChecked(data['use_deconv_mask'] == 'True')

                #self.checkBox_use_deconv_mask.setChecked(data['use_deconv_mask'])

                self.lineEdit_patch_size.setText(data['patch_size'])
                self.lineEdit_z_crop.setText(data['z_crop'])

                self.lineEdit_subtomo_dir.setText(data['subtomo_dir'])
                self.lineEdit_subtomo_star_extract.setText(data['subtomo_star_extract'])
                self.checkBox_use_deconv_extract.setChecked(data['use_deconv_extract'] == 'True')
                self.lineEdit_cube_size_extract.setText(data['cube_size_extract'])
                self.lineEdit_tomo_index_extract.setText(data['tomo_index_extract'])

                self.lineEdit_subtomo_star_refine.setText(data['subtomo_star_refine'])
                self.lineEdit_gpuID_refine.setText(data['gpuID_refine'])
                self.lineEdit_pretrain_model_refine.setText(data['pretrain_model_refine'])
                self.lineEdit_continue_iter.setText(data['continue_iter'])
                self.lineEdit_result_dir_refine.setText(data['result_dir_refine'])
                self.lineEdit_ncpu.setText(data['ncpu'])
                
                self.lineEdit_epoch.setText(data['epoch'])
                self.lineEdit_iteration.setText(data['iteration'])
                self.lineEdit_lr.setText(data['lr'])
                self.lineEdit_steps_per_epoch.setText(data['steps_per_epoch'])
                self.lineEdit_batch_size.setText(data['batch_size'])

                self.lineEdit_noise_level.setText(data['noise_level'])
                self.lineEdit_noise_start_iter.setText(data['noise_start_iter'])
                self.comboBox_noise_mode.setCurrentText(data['noise_mode'])

                self.lineEdit_drop_out.setText(data['drop_out'])
                self.lineEdit_network_depth.setText(data['network_depth'])
                self.lineEdit_convs_per_depth.setText(data['convs_per_depth'])
                self.lineEdit_kernel.setText(data['kernel'])
                self.lineEdit_filter_base.setText(data['filter_base'])
                self.checkBox_pool.setChecked(data['pool'] == 'True')
                self.checkBox_batch_normalization.setChecked(data['batch_normalization'] == 'True')
                self.checkBox_normalization_percentile.setChecked(data['normalization_percentile'] == 'True')

                self.lineEdit_tomo_star_predict.setText(data['tomo_star_predict'])
                self.lineEdit_gpuID_predict.setText(data['gpuID_predict'])
                self.lineEdit_tomo_index_predict.setText(data['tomo_index_predict'])
                self.lineEdit_pretrain_model_predict.setText(data['pretrain_model_predict'])
                self.lineEdit_cube_size_predict.setText(data['cube_size_predict'])
                self.lineEdit_result_dir_predict.setText(data['result_dir_predict'])
                self.lineEdit_crop_size_predict.setText(data['crop_size_predict'])
                self.checkBox_use_deconv_predict.setChecked(data['use_deconv_predict'] == 'True')
                
            except:
                print("error reading {}!".format(self.setting_file))
    #每次按钮点击后保存相关参数
    def save_setting(self):
        param = {}
        #保存Deconvolve按钮的参数
        param['deconv_dir'] = self.lineEdit_deconv_dir.text()
        param['tomo_index_deconv'] = self.lineEdit_tomo_index_deconv.text()
        param['preprocessing_ncpus'] = self.lineEdit_preprocessing_ncpus.text()
        param['chunk_size'] = self.lineEdit_chunk_size.text()
        param['highpassnyquist'] = self.lineEdit_highpassnyquist.text()
        param['overlap'] = self.lineEdit_overlap.text()
        #保存Generate Mask按钮的参数
        param['mask_dir'] = self.lineEdit_mask_dir.text()
        param['tomo_index_mask'] = self.lineEdit_tomo_index_mask.text()
        param['use_deconv_mask'] = self.checkBox_use_deconv_mask.isChecked()
        param['patch_size'] = self.lineEdit_patch_size.text()
        param['z_crop'] = self.lineEdit_z_crop.text()
        #保存Extract 按钮的参数
        param['subtomo_dir'] = self.lineEdit_subtomo_dir.text()
        param['subtomo_star_extract'] = self.lineEdit_subtomo_star_extract.text()
        param['use_deconv_extract'] = self.checkBox_use_deconv_extract.isChecked()
        param['cube_size_extract'] = self.lineEdit_cube_size_extract.text()
        param['tomo_index_extract'] = self.lineEdit_tomo_index_extract.text()

        # 保存Refine 按钮的参数
        # General options参数
        param['subtomo_star_refine'] = self.lineEdit_subtomo_star_refine.text()
        param['gpuID_refine'] = self.lineEdit_gpuID_refine.text()
        param['pretrain_model_refine'] = self.lineEdit_pretrain_model_refine.text()
        param['continue_iter'] = self.lineEdit_continue_iter.text()
        param['result_dir_refine'] = self.lineEdit_result_dir_refine.text()
        param['ncpu'] = self.lineEdit_ncpu.text()
        #training setting参数
        param['epoch'] = self.lineEdit_epoch.text()
        param['iteration'] = self.lineEdit_iteration.text()
        param['lr'] = self.lineEdit_lr.text()
        param['steps_per_epoch'] = self.lineEdit_steps_per_epoch.text()
        param['batch_size'] = self.lineEdit_batch_size.text()
        #Denoise setting参数
        param['noise_level'] = self.lineEdit_noise_level.text()
        param['noise_start_iter'] = self.lineEdit_noise_start_iter.text()
        param['noise_mode'] = self.comboBox_noise_mode.currentText()
        #Network setting
        param['drop_out'] = self.lineEdit_drop_out.text()
        param['network_depth'] = self.lineEdit_network_depth.text()
        param['convs_per_depth'] = self.lineEdit_convs_per_depth.text()
        param['kernel'] = self.lineEdit_kernel.text()
        param['filter_base'] = self.lineEdit_filter_base.text()
        param['pool'] = self.checkBox_pool.isChecked()
        param['batch_normalization'] = self.checkBox_batch_normalization.isChecked()
        param['normalization_percentile'] = self.checkBox_normalization_percentile.isChecked()
        #prdiction参数
        #General option参数
        param['tomo_star_predict'] = self.lineEdit_tomo_star_predict.text()
        param['gpuID_predict'] = self.lineEdit_gpuID_predict.text()
        param['tomo_index_predict'] = self.lineEdit_tomo_index_predict.text()
        param['pretrain_model_predict'] = self.lineEdit_pretrain_model_predict.text()
        param['cube_size_predict'] = self.lineEdit_cube_size_predict.text()
        param['result_dir_predict'] = self.lineEdit_result_dir_predict.text()
        param['crop_size_predict'] = self.lineEdit_crop_size_predict.text()
        param['use_deconv_predict'] = self.checkBox_use_deconv_predict.isChecked()


        try:
            with open(self.setting_file, 'w') as f: 
                for key, value in param.items(): 
                    f.write("{}:{}\n".format(key,value))
        except:
            print("error writing {}!".format(self.setting_file))
    #打开github网站
    def openGithub(self):
        import webbrowser
        webbrowser.open(self.model.github_addr)
    #打开警告网站（列如当你用deconv时，然后用gennerate mask）
    def warn_window(self,text):
        msg = QMessageBox()
        msg.setWindowTitle("Warning!")
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setIcon(QMessageBox.Warning)
        msg.exec_()
        
     
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.p = None
    #关闭软件时你看到的情况
    def closeEvent(self, event):
        if self.p:
        
            result = QtWidgets.QMessageBox.question(self,
                          "Confirm Exit...",
                          "Do you want to continue the existing job in the background?",
                          QtWidgets.QMessageBox.Yes| QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
            event.ignore()
            if result == QtWidgets.QMessageBox.Yes:
                event.accept()
            if result == QtWidgets.QMessageBox.No:
                self.p.kill()
                event.accept()
                #kill the old process
        else:
            result = QtWidgets.QMessageBox.question(self,
                          "Confirm Exit...",
                          "Do you want to exit? ",
                          QtWidgets.QMessageBox.Yes| QtWidgets.QMessageBox.No )
            event.ignore()
            if result == QtWidgets.QMessageBox.Yes:
                event.accept()
            if result == QtWidgets.QMessageBox.No:
                pass
                #kill the old process


stylesheet = """


QWidget #tab, #tab_2, #tab_3{
    background-color: rgb(253,247,226)
}

QTabWidget{
    background: rgb(144,160,187)
}

QPushButton {
    background: rgb(239,221,241)
}


"""

def main():
    """
    This is the MAIN ENTRY POINT of our application.  The code at the end
    of the mainwindow.py script will not be executed, since this script is now
    our main program.   We have simply copied the code from mainwindow.py here
    since it was automatically generated by '''pyuic5'''.
    """
    """
    这是我们应用程序的主入口点。
    mainwindow.py脚本末尾的代码将不会被执行,因为这个脚本现在是我们的主程序。
    我们只是将从mainwindow.py复制了代码过来,因为它是由'pyuic5'自动生成的。
    """
    # 创建Qt应用程序实例
    app = QtWidgets.QApplication(sys.argv)
    # 设置应用程序样式表
    app.setStyleSheet(stylesheet)
    # 创建主窗口实例
    MainWindow = MyWindow()
    # MainWindow = QtWidgets.QMainWindow()
    # 创建UI实例并设置到主窗口
    ui = MainWindowUIClass()
    ui.setupUi(MainWindow)
    # 显示主窗口
    MainWindow.show()
    # 进入Qt应用程序的事件循环
    sys.exit(app.exec_())
# 调用main函数，启动应用程序
